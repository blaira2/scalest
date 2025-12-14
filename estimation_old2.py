# estimation.py

import cv2 as cv
import numpy as np
import glob
import os
import random
import pandas as pd
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import matplotlib.pyplot as plt


class Estimator:
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: torch.device | None = None ):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model_name = model_name

        # Load segmentation model & processor once
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(
            device
        )
        self.model.eval()


    # Keypoints / Matching

    def resize_image(self, ImOriginal, scale_factor=0.125):
        Im = cv.resize(
            ImOriginal, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA
        )
        return Im

    def compute_keypoints_and_descriptors(self, leftIm, rightIm, features=5000):
        orb = cv.ORB_create(nfeatures=features)
        kL, dL = orb.detectAndCompute(leftIm, None)
        kR, dR = orb.detectAndCompute(rightIm, None)

        print(f"Number of left descriptors: {len(dL)}; right descriptors: {len(dR)}")
        return kL, dL, kR, dR

    def save_keypoint_images(self, leftIm, rightIm, kL, kR, output_path):
        keypointedImLeft = cv.drawKeypoints(
            leftIm, kL, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv.imwrite(os.path.join(output_path, "image_left_keypoints.jpg"), keypointedImLeft)

        keypointedImRight = cv.drawKeypoints(
            rightIm, kR, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv.imwrite(os.path.join(output_path, "image_right_keypoints.jpg"), keypointedImRight)

    def find_matches(self, dL, dR, kL, kR, l_ratio=0.8, h_ratio=0.9):
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        knn_matches = bf.knnMatch(dL, dR, k=2)

        matches = []
        for m, n in knn_matches:
            if m.distance < l_ratio * n.distance:
                matches.append(m)

        good = sorted(matches, key=lambda m: m.distance)
        keep = int(len(good) * h_ratio)
        good = good[: max(keep, 1)]

        matched_pts = []
        for m in good:
            ptL = tuple(map(int, kL[m.queryIdx].pt))
            ptR = tuple(map(int, kR[m.trainIdx].pt))
            matched_pts.append((ptL, ptR))

        print("Number of matching points:", len(matched_pts))
        return matched_pts

    def draw_matches_image(self, leftIm, rightIm, matched, output_path, filename):
        hRight, wRight = rightIm.shape[:2]
        hLeft, wLeft = leftIm.shape[:2]
        combinedWidths = wLeft + wRight
        maxHeight = max(hLeft, hRight)

        imMatches = np.zeros((maxHeight, combinedWidths, 3), dtype="uint8")
        imMatches[0:hRight, wLeft:] = rightIm
        imMatches[0:hLeft, 0:wLeft] = leftIm

        for (leftPoints, rightPoints) in matched:
            pointsL = leftPoints
            pointsR = (wLeft + rightPoints[0], rightPoints[1])
            blue = (255, 0, 0)
            green = (0, 255, 0)
            red = (0, 0, 255)
            cv.circle(imMatches, pointsL, 3, blue, 1)
            cv.circle(imMatches, pointsR, 3, green, 1)
            cv.line(imMatches, pointsL, pointsR, red, 1)

        cv.imwrite(os.path.join(output_path, filename), imMatches)

    def ransac_homography(self, matched, num_iterations=7000, inlier_threshold=5.0):
        leftPoints = []
        rightPoints = []
        for leftPoint, rightPoint in matched:
            leftPoints.append(list(leftPoint))
            rightPoints.append(list(rightPoint))

        leftPoints = np.array(leftPoints)
        rightPoints = np.array(rightPoints)

        Best_H = None
        inlierMax = 0

        for _ in range(num_iterations):
            if len(matched) < 4:
                break

            sampleInd = random.sample(range(len(matched)), 4)
            A = []

            for r in range(len(rightPoints[sampleInd])):
                x_r, y_r = rightPoints[sampleInd][r]
                x_l, y_l = leftPoints[sampleInd][r]

                A.append(
                    [-x_r, -y_r, -1, 0, 0, 0, x_r * x_l, y_r * x_l, x_l]
                )
                A.append(
                    [0, 0, 0, -x_r, -y_r, -1, x_r * y_l, y_r * y_l, y_l]
                )

            U, SIGMA, VT = np.linalg.svd(A)
            H = np.reshape(VT[8], (3, 3))
            H = (1.0 / H.item(8)) * H

            inlierCount = 0
            for i in range(len(matched)):
                if i in sampleInd:
                    continue
                withZaxis = np.hstack((rightPoints[i], [1]))
                transformedPoint = H @ withZaxis.T
                if transformedPoint[2] <= 1e-7:
                    continue
                transformedPoint = transformedPoint / transformedPoint[2]
                if (
                    np.linalg.norm(transformedPoint[:2] - leftPoints[i])
                    < inlier_threshold
                ):
                    inlierCount += 1

            if inlierCount > inlierMax:
                Best_H = H
                inlierMax = inlierCount

        print("The Number of Maximum Inliers:", inlierMax)
        return Best_H, inlierMax

    def warp_and_stitch(self, leftIm, rightIm, H):
        (hr, wr) = rightIm.shape[:2]
        (hl, wl) = leftIm.shape[:2]
        print("Left image size: (", hl, "*", wl, ")")
        print("Right image size: (", hr, "*", wr, ")")

        combinedImages = np.zeros((max(hl, hr), wl + wr, 3), dtype="int")
        combinedImages[:hl, :wl] = leftIm

        invH = np.linalg.inv(H)

        for i in range(combinedImages.shape[0]):
            for j in range(combinedImages.shape[1]):
                cRightIm = invH @ np.array([j, i, 1])
                cRightIm /= cRightIm[2]
                y, x = int(round(cRightIm[0])), int(round(cRightIm[1]))
                if x < 0 or x >= hr or y < 0 or y >= wr:
                    continue
                combinedImages[i, j] = rightIm[x, y]

        return combinedImages

    def remove_black_border(self, image):
        nonZeroRows = np.any(np.any(0 < image, axis=2), axis=1)
        nonZeroCols = np.any(np.any(0 < image, axis=2), axis=0)
        stitch_img = image[nonZeroRows][:, nonZeroCols]
        return stitch_img

    def find_displacement(self, matched, baseline, focal_length_pixels):
        displacements_pixels = []
        displacements_meters = []
        depths_meters = []
        for (leftPoints, rightPoints) in matched:
            xL, yL = leftPoints
            xR, yR = rightPoints
            pixel_displacement = np.sqrt(
                (xL - xR) ** 2 + (yL - yR) ** 2
            )
            displacements_pixels.append(pixel_displacement)
            disparity = abs(xL - xR)
            if disparity > 0:
                depth = (focal_length_pixels * baseline) / disparity
                depths_meters.append(depth)
                meter_displacement = (
                    pixel_displacement / focal_length_pixels
                ) * depth
                displacements_meters.append(meter_displacement)
            else:
                depths_meters.append(0)
                displacements_meters.append(0)

        if displacements_pixels and depths_meters:
            avg_displacement_pixels = np.mean(displacements_pixels)
            min_displacement_pixels = np.min(displacements_pixels)
            max_displacement_pixels = np.max(displacements_pixels)
            positive_m = [d for d in displacements_meters if d > 0]
            positive_z = [d for d in depths_meters if d > 0]
            avg_displacement_meters = np.mean(positive_m) if positive_m else 0
            min_displacement_meters = np.min(positive_m) if positive_m else 0
            max_displacement_meters = np.max(positive_m) if positive_m else 0
            avg_depth_meters = np.mean(positive_z) if positive_z else 0
            min_depth_meters = np.min(positive_z) if positive_z else 0
            max_depth_meters = np.max(positive_z) if positive_z else 0

            print(f"Number of matched keypoints: {len(displacements_pixels)}")
            print(f"Average displacement: {avg_displacement_pixels:.2f} pixels")
            print(f"Minimum displacement: {min_displacement_pixels:.2f} pixels")
            print(f"Maximum displacement: {max_displacement_pixels:.2f} pixels")
            print(f"Average displacement: {avg_displacement_meters:.4f} meters")
            print(f"Minimum displacement: {min_displacement_meters:.4f} meters")
            print(f"Maximum displacement: {max_displacement_meters:.4f} meters")
            print(f"Average depth: {avg_depth_meters:.2f} meters")
            print(f"Minimum depth: {min_depth_meters:.2f} meters")
            print(f"Maximum depth: {max_depth_meters:.2f} meters")
        else:
            print("No matched keypoints found.")
        return depths_meters, displacements_meters

    # ---------------------------------------------------------------------
    # Segmentation: buildings & ground
    # ---------------------------------------------------------------------

    def get_building_mask(self, rgb_small):
        building_words = [
            "building",
            "house",
            "skyscraper",
            "tower",
            "castle",
            "palace",
            "apartment",
        ]
        height, width = rgb_small.shape[:2]
        pil_img = Image.fromarray(rgb_small)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        pred = logits.argmax(dim=1)[0].cpu().numpy()
        id2label = self.model.config.id2label
        building_ids = [
            i
            for i, name in id2label.items()
            if any(word in name.lower() for word in building_words)
        ]

        building_mask = np.isin(pred, building_ids).astype("uint8") * 255
        return building_mask

    def building_edges(self, building_mask, rgb_image, low_thresh=100, high_thresh=200):
        gray = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
        edges = cv.Canny(gray, low_thresh, high_thresh)
        masked_edges = cv.bitwise_and(edges, edges, mask=building_mask)
        return masked_edges

    def unmasked_matches(self, matched, mask):
        ptsL = []
        ptsR = []
        height, width = mask.shape
        allowed_labels = {0}
        filtered_matches = []

        for (cL, cR) in matched:
            uL, vL = cL
            if not (0 <= vL < height and 0 <= uL < width):
                continue
            label = mask[vL, uL]
            if label in allowed_labels:
                filtered_matches.append([cL, cR])
        return filtered_matches

    # ---------------------------------------------------------------------
    # Triangulation / building match selection
    # ---------------------------------------------------------------------

    def triangulate_points(self, pts1, pts2, focal_length_pixels, baseline_m, image_shape):
        pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
        pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)

        height, width = image_shape[:2]
        cx = width / 2.0
        cy = height / 2.0

        uL = pts1[:, 0]
        vL = pts1[:, 1]
        uR = pts2[:, 0]

        disparity = uL - uR

        eps = 1e-6
        valid_mask = disparity > eps
        disparity_valid = disparity[valid_mask]
        uL_valid = uL[valid_mask]
        vL_valid = vL[valid_mask]

        Z = (focal_length_pixels * baseline_m) / disparity_valid
        X = (uL_valid - cx) * Z / focal_length_pixels
        Y = (vL_valid - cy) * Z / focal_length_pixels

        pts3d = np.zeros((pts1.shape[0], 3), dtype=np.float64)
        pts3d[valid_mask, 0] = X
        pts3d[valid_mask, 1] = Y
        pts3d[valid_mask, 2] = Z

        return pts3d, valid_mask

    def building_matches(self, matched, building_mask):
        height, width = building_mask.shape[:2]
        filtered = []
        for (cL, cR) in matched:
            uL, vL = cL
            if 0 <= vL < height and 0 <= uL < width:
                if building_mask[vL, uL] > 0:
                    filtered.append((cL, cR))
        return filtered

    # ---------------------------------------------------------------------
    # Height estimation
    # ---------------------------------------------------------------------
    """
    Height estimation Plan:
    1. Pick a vertical line within the building mask
    2. Estimate the depth of the line (distance from camera) by comparing nearby points
    3. Calculate a ground plane that intersects with the base of the building.
    4. Get the projected height of the line
    """

    def vertical_line_through_point(self, building_mask, u0, v0, min_run=10):
        height, width = building_mask.shape[:2]
        u0 = int(round(u0))
        v0 = int(round(v0))

        if not (0 <= u0 < width and 0 <= v0 < height):
            return None, None
        if building_mask[v0, u0] == 0:
            return None, None

        v_top = v0
        while v_top - 1 >= 0 and building_mask[v_top - 1, u0] > 0:
            v_top -= 1

        v_bottom = v0
        while v_bottom + 1 < height and building_mask[v_bottom + 1, u0] > 0:
            v_bottom += 1

        if (v_bottom - v_top) < min_run:
            return None, None

        return v_top, v_bottom

    def pick_known_point(self, pts2d, pts3d, building_mask, horizontal_target=None, horiz_radius=20):
        pts2d = np.asarray(pts2d, dtype=np.float64)
        pts3d = np.asarray(pts3d, dtype=np.float64)
        num_points = len(pts2d)

        height, width = building_mask.shape[:2]

        if horizontal_target is None:
            horizontal_target = width * 0.5

        ys, xs = np.where(building_mask > 0)
        if ys.size == 0:
            return None, None

        v_center = np.median(ys)
        candidates = []

        for i in range(num_points):
            u, v = pts2d[i]
            ui = int(round(u))
            vi = int(round(v))
            if not (0 <= ui < width and 0 <= vi < height):
                continue
            if building_mask[vi, ui] == 0:
                continue
            if abs(ui - horizontal_target) > horiz_radius:
                continue

            score = abs(vi - v_center)
            score += 0.3 * abs(ui - horizontal_target)
            candidates.append((score, i))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[0])
        _, best_idx = candidates[0]
        return pts2d[best_idx], pts3d[best_idx]

    def project_point(self, point, focal_length_pixels, image_shape):
        height, width = image_shape[:2]
        fx = fy = float(focal_length_pixels)
        cx = width / 2.0
        cy = height / 2.0

        x, y, z = point
        if z <= 1e-6:
            return None

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        if 0 <= u < width and 0 <= v < height:
            return int(round(u)), int(round(v))
        else:
            return None

    def ground_plane_intersect(self, known_3d, plane_normal, plane_d, focal_length_pixels, image_shape):
        X0 = np.asarray(known_3d, dtype=np.float64)
        n = np.asarray(plane_normal, dtype=np.float64)
        n /= (np.linalg.norm(n) + 1e-12)

        dist = np.dot(n, X0) + plane_d
        X_ground = X0 - dist * n

        return self.project_point(X_ground, focal_length_pixels, image_shape)

    def estimate_heights(self, building_mask,
        pts2d_valid,
        pts3d_valid,
        focal_length_pixels,
        plane_normal,
        plane_d,
        horiz_radius=25):
        
        height, width = building_mask.shape[:2]

        pts2d_valid = np.asarray(pts2d_valid, dtype=np.float64)
        pts3d_valid = np.asarray(pts3d_valid, dtype=np.float64)
        num_points = min(len(pts2d_valid), len(pts3d_valid))
        pts2d_valid = pts2d_valid[:num_points]
        pts3d_valid = pts3d_valid[:num_points]

        fractions = [0.2, 0.4, 0.5, 0.6, 0.8]
        lines = []

        for frac in fractions:
            horizontal_target = width * frac

            known_2d, known_3d = self.pick_known_point(
                pts2d_valid,
                pts3d_valid,
                building_mask,
                horizontal_target=horizontal_target,
                horiz_radius=horiz_radius,
            )
            if known_2d is None:
                continue

            u0, v0 = known_2d
            X0, Y0, Z0 = known_3d


            v_top, v_bottom = self.vertical_line_through_point(
                building_mask, u0, v0, min_run=10 )
            if v_top is None:
                continue

            bottom = self.ground_plane_intersect(
                known_3d,
                plane_normal,
                plane_d,
                focal_length_pixels,
                image_shape=building_mask.shape,
            )

            u_bottom, v_bottom = bottom
            
            pixel_dist = abs(v_bottom - v_top)

            #plane_normal_arr = np.asarray(plane_normal, dtype=np.float64)
            #plane_normal_arr /= np.linalg.norm(plane_normal_arr) + 1e-12
            #height_m = abs(np.dot(plane_normal_arr, known_3d) + plane_d)
            height_m = (pixel_dist / float(focal_length_pixels)) * float(Z0)

            top_pt = (int(round(u0)), int(v_top))
            bottom_pt = (int(round(u0)), int(v_bottom))
            lines.append((height_m, top_pt, bottom_pt))

        return lines

    def draw_building_height_lines(
        self, image_bgr, lines, line_color=(0, 0, 255), thickness=2
    ):
        img = image_bgr.copy()
        for (height_m, top_pt, bottom_pt) in lines:
            cv.line(img, top_pt, bottom_pt, line_color, thickness)
            cv.circle(img, top_pt, 5, (255, 0, 0), -1)
            cv.circle(img, bottom_pt, 5, (0, 255, 0), -1)
            cv.putText(
                img,
                f"{height_m:.2f} m",
                (top_pt[0] + 5, top_pt[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                line_color,
                2,
                cv.LINE_AA,
            )
        return img

    # ---------------------------------------------------------------------
    # Ground segmentation & plane estimation
    # ---------------------------------------------------------------------

    def get_ground_mask(self, rgb_small):
        ground_words = [
            "road",
            "street",
            "sidewalk",
            "pavement",
            "ground",
            "earth",
            "grass",
            "field",
            "path",
        ]

        height, width = rgb_small.shape[:2]
        pil_img = Image.fromarray(rgb_small)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        pred = logits.argmax(dim=1)[0].cpu().numpy()
        id2label = self.model.config.id2label

        ground_ids = [
            i
            for i, name in id2label.items()
            if any(word in name.lower() for word in ground_words)
        ]

        ground_mask = np.isin(pred, ground_ids).astype("uint8") * 255
        return ground_mask

    def ground_matches(self, matched, ground_mask):
        height, width = ground_mask.shape[:2]
        filtered = []
        for (cL, cR) in matched:
            uL, vL = cL
            if 0 <= vL < height and 0 <= uL < width:
                if ground_mask[vL, uL] > 0:
                    filtered.append((cL, cR))
        return filtered

    #synthetic keypoints
    def generate_grid_keypoints(self, img_shape, grid_rows=60, grid_cols=60, mask=None ):
    
        height, width =  img_shape[:2]
 
        cell_h = height / grid_rows
        cell_w = width / grid_cols

        keypoints = []

        rng = np.random.default_rng()

        for gi in range(grid_rows):
            for gj in range(grid_cols):
                # Center of this grid cell
                v = (gi + 0.5) * cell_h
                u = (gj + 0.5) * cell_w

                # Clamp to image bounds
                u = np.clip(u, 0, width - 1)
                v = np.clip(v, 0, height - 1)

                # If we have a mask, skip if outside region
                if mask is not None:
                    if mask[int(round(v)), int(round(u))] == 0:
                        continue

                kp = cv.KeyPoint(float(u), float(v), size=8)
                keypoints.append(kp)

        return keypoints

    def fit_plane_ransac(self, points, num_iterations=1000, min_sample_dist=0.4, distance_thresh=0.05):
        points = np.asarray(points, dtype=np.float64)
        N = points.shape[0]
        if N < 3:
            raise ValueError("Not enough points")

        best_inliers = None
        best_count = 0
        best_n = None
        best_d = None

        yplus = np.array([0.0, 1.0, 0.0])

        for _ in range(num_iterations):
            good_sample = False
            for _ in range(10):
                idx = np.random.choice(N, 3, replace=False)
                p1, p2, p3 = points[idx]
                
                 # pairwise distances
                d12 = np.linalg.norm(p1 - p2)
                d13 = np.linalg.norm(p1 - p3)
                d23 = np.linalg.norm(p2 - p3)
                min_d = min(d12, d13, d23)

                if min_d < min_sample_dist:
                    # too clustered → try again
                    continue
                good_sample = True

            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-6:
                continue
            n = n / norm_n

            cos_angle = np.clip(np.dot(n, yplus), -1.0, 1.0)
            angle = np.degrees(np.arccos(abs(cos_angle)))
            if angle > 40:
                continue
            if not good_sample:
                continue

            d = -np.dot(n, p1)

            dist = np.abs(points @ n + d)
            inliers = dist < distance_thresh
            count = np.count_nonzero(inliers)

            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_n = n
                best_d = d

        if best_inliers is None:
            raise RuntimeError("RANSAC failed to find a plane")

        P_in = points[best_inliers]
        centroid = P_in.mean(axis=0)
        Q = P_in - centroid
        U, S, Vt = np.linalg.svd(Q)
        n_refined = Vt[-1]
        n_refined = n_refined / np.linalg.norm(n_refined)
        d_refined = -np.dot(n_refined, centroid)

        return n_refined, d_refined, best_inliers

    def ground_specific_keypoints_and_descriptors(self,
        leftIm, rightIm,
        ground_mask,
        max_corners=500,
        quality=0.005,
        min_distance=5,
        orb_features=3000 ):
        
        grayL = cv.cvtColor(leftIm, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(rightIm, cv.COLOR_BGR2GRAY)
        hL, wL = grayL.shape
        hR, wR = grayR.shape

        ground_mask_left = ground_mask
        
        ground_mask_right = np.zeros((hR, wR), dtype=np.uint8)
        ground_mask_right[hR // 2 :, :] = 255


        cornersL = cv.goodFeaturesToTrack(grayL,
            maxCorners=max_corners,
            qualityLevel=quality,
            minDistance=min_distance,
            mask=ground_mask_left )
            
        if cornersL is not None:
            cornersL = cornersL.reshape(-1, 2)
            kL = [cv.KeyPoint(float(u), float(v), 16) for (u, v) in cornersL]
        
        synthetic_kL = self.generate_grid_keypoints( grayL.shape, mask=ground_mask_left )
            
        kL.extend(synthetic_kL)
        
        cornersR = cv.goodFeaturesToTrack(grayR,
            maxCorners=max_corners,
            qualityLevel=quality,
            minDistance=min_distance,
            mask=ground_mask_right )

        if cornersR is not None:
            cornersR = cornersR.reshape(-1, 2)
            kR = [cv.KeyPoint(float(u), float(v), 16) for (u, v) in cornersR]
        
        synthetic_kR = self.generate_grid_keypoints( grayR.shape, mask=ground_mask_right )
            
        kR.extend(synthetic_kR)

        orb = cv.ORB_create(nfeatures=orb_features)

        kL_final, dL = orb.compute(grayL, kL)
        kR_final, dR = orb.compute(grayR, kR)

        print(f"Ground; Left descriptors: {len(dL)}; Right descriptors: {len(dR)}")

        return kL_final, dL, kR_final, dR

    def estimate_ground_plane(self, leftIm, rightIm,
        ground_mask,
        baseline_m,
        focal_length_pixels,
        max_corners=3000,
        quality=0.05,
        min_distance=5,
        ransac_thresh=0.05 ):
        
        kL, dL, kR, dR = self.ground_specific_keypoints_and_descriptors(
            leftIm, rightIm,
            ground_mask,
            max_corners=max_corners,
            quality=quality,
            min_distance=min_distance,
            orb_features=3000 )

        g_matched = self.find_matches(dL, dR, kL, kR)
        ground_matched = self.ground_matches(g_matched, ground_mask)

        if len(ground_matched) < 3:
            raise ValueError(f"Not enough ground matches after masking: {len(ground_matched)}")

        pts1_ground = np.array([m[0] for m in ground_matched], dtype=np.float64)
        pts2_ground = np.array([m[1] for m in ground_matched], dtype=np.float64)

        pts3d_ground, valid_ground = self.triangulate_points(
            pts1_ground,
            pts2_ground,
            focal_length_pixels=focal_length_pixels,
            baseline_m=baseline_m,
            image_shape=leftIm.shape,
        )

        pts3d_ground_valid = pts3d_ground[valid_ground]
        pts2d_ground_valid = pts1_ground[valid_ground]

        if pts3d_ground_valid.shape[0] < 3:
            raise ValueError(
                f"Not enough valid 3D ground points: {pts3d_ground_valid.shape[0]}"
            )

        z = pts3d_ground_valid[:, 2]
        depth_mask = (z > 1.0) & (z < 40.0)
        pts3d_ground_valid = pts3d_ground_valid[depth_mask]
        pts2d_ground_valid = pts2d_ground_valid[depth_mask]

        n_ground, d_ground, inliers_ground = self.fit_plane_ransac(
            pts3d_ground_valid, distance_thresh=ransac_thresh
        )

        return n_ground, d_ground, inliers_ground, pts2d_ground_valid, pts3d_ground_valid


    def visualize_ground_plane_on_image(self,
        image_bgr,
        pts2d_ground_valid,
        pts3d_ground_valid,
        inlier_mask,
        plane_normal,
        plane_d,
        focal_length_pixels,
        grid_extent_m=5.0,
        grid_step_m=1.0 ):
        
        img = image_bgr.copy()
        h, w = img.shape[:2]

        fx = fy = float(focal_length_pixels)
        cx = w / 2.0
        cy = h / 2.0

        pts2d_ground_valid = np.asarray(pts2d_ground_valid, dtype=np.float64)
        pts3d_ground_valid = np.asarray(pts3d_ground_valid, dtype=np.float64)
        inlier_mask = np.asarray(inlier_mask, dtype=bool)

        inlier_indices = np.where(inlier_mask)[0]

        for idx in inlier_indices:
            u, v = pts2d_ground_valid[idx]
            u_i = int(round(u))
            v_i = int(round(v))
            if 0 <= u_i < w and 0 <= v_i < h:
                cv.circle(img, (u_i, v_i), 4, (0, 0, 255), -1)

        if len(inlier_indices) < 3:
            cv.putText(
                img,
                "Not enough inliers to draw ground plane grid",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            return img

        plane_normal = np.asarray(plane_normal, dtype=np.float64)
        n_hat = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)

        P_inliers = pts3d_ground_valid[inlier_mask]
        p0 = P_inliers.mean(axis=0)

        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, n_hat)) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        u_dir = np.cross(n_hat, tmp)
        u_dir /= np.linalg.norm(u_dir) + 1e-12
        v_dir = np.cross(n_hat, u_dir)
        v_dir /= np.linalg.norm(v_dir) + 1e-12

        s_vals = np.arange(-grid_extent_m, grid_extent_m + 1e-9, grid_step_m)
        t_vals = np.arange(-grid_extent_m, grid_extent_m + 1e-9, grid_step_m)

        def project_point(X):
            X = np.asarray(X, dtype=np.float64)
            Xx, Xy, Xz = X
            if Xz <= 1e-6:
                return None
            u = fx * (Xx / Xz) + cx
            v = fy * (Xy / Xz) + cy
            if 0 <= u < w and 0 <= v < h:
                return int(round(u)), int(round(v))
            return None

        for s in s_vals:
            pts_line = []
            for t in t_vals:
                X = p0 + s * u_dir + t * v_dir
                uv = project_point(X)
                if uv is not None:
                    pts_line.append(uv)

            if len(pts_line) >= 2:
                pts_np = np.array(pts_line, dtype=np.int32).reshape(-1, 1, 2)
                cv.polylines(img, [pts_np], False, (0, 255, 0), 1, cv.LINE_AA)

        for t in t_vals:
            pts_line = []
            for s in s_vals:
                X = p0 + s * u_dir + t * v_dir
                uv = project_point(X)
                if uv is not None:
                    pts_line.append(uv)

            if len(pts_line) >= 2:
                pts_np = np.array(pts_line, dtype=np.int32).reshape(-1, 1, 2)
                cv.polylines(img, [pts_np], False, (0, 255, 255), 1, cv.LINE_AA)

        cv.putText(
            img,
            f"Ground inliers: {len(inlier_indices)}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        return img

    def debug_plane_orientation(self, n_ground):
        n_ground = np.asarray(n_ground, dtype=np.float64)
        n_ground /= np.linalg.norm(n_ground) + 1e-12

        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cos_angle = np.clip(np.dot(n_ground, up), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(abs(cos_angle)))
        print("Plane normal:", n_ground)
        print("Angle between plane normal and up-axis:", angle_deg, "deg")
