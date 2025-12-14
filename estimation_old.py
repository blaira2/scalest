
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


# Point matching

def resize_image(ImOriginal, scale_factor=0.125):
    Im = cv.resize(ImOriginal, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
    return Im



def compute_keypoints_and_descriptors(leftIm, rightIm, features=5000):
    orb = cv.ORB_create(nfeatures=features)
    kL, dL = orb.detectAndCompute(leftIm, None)
    kR, dR = orb.detectAndCompute(rightIm, None)

    print(f"Number of left descriptors: {len(dL)}; right descriptors: {len(dR)}")
    #numDescsToPrint = 1
    #print("First left and right descriptors (128-dimensional vectors):")
    #print(dL[:numDescsToPrint])
    #print(dR[:numDescsToPrint])

    return kL, dL, kR, dR


def save_keypoint_images(leftIm, rightIm, kL, kR, output_path):
    keypointedImLeft = cv.drawKeypoints(
        leftIm, kL, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv.imwrite(os.path.join(output_path, "image_left_keypoints.jpg"), keypointedImLeft)

    keypointedImRight = cv.drawKeypoints(
        rightIm, kR, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv.imwrite(os.path.join(output_path, "image_right_keypoints.jpg"), keypointedImRight)



def find_matches(dL, dR, kL, kR, l_ratio=0.8, h_ratio=.9):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(dL, dR, k=2)

    #select on Lowe's ratio
    matches = []
    for m, n in knn_matches:
        if m.distance < l_ratio * n.distance: 
            matches.append(m)
            
    # filter points on Hamming distance
    good = sorted(matches, key=lambda m: m.distance)
    keep = int(len(good) * h_ratio)
    good = good[:max(keep, 1)]

    matched_pts = []
    for m in good:
        ptL = tuple(map(int, kL[m.queryIdx].pt))
        ptR = tuple(map(int, kR[m.trainIdx].pt))
        matched_pts.append((ptL, ptR))

    print("Number of matching points:", len(matched_pts))
    return matched_pts


def draw_matches_image(leftIm, rightIm, matched, output_path, filename):
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



def ransac_homography(matched, num_iterations=7000, inlier_threshold=5.0):
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

            A.append([-x_r, -y_r, -1, 0, 0, 0,
                      x_r * x_l, y_r * x_l, x_l])
            A.append([0, 0, 0, -x_r, -y_r, -1,
                      x_r * y_l, y_r * y_l, y_l])

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
            if np.linalg.norm(transformedPoint[:2] - leftPoints[i]) < inlier_threshold:
                inlierCount += 1

        if inlierCount > inlierMax:
            Best_H = H
            inlierMax = inlierCount

    print("The Number of Maximum Inliers:", inlierMax)
    return Best_H, inlierMax



def warp_and_stitch(leftIm, rightIm, H):
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


def remove_black_border(image):
    nonZeroRows = np.any(np.any(0 < image, axis=2), axis=1)
    nonZeroCols = np.any(np.any(0 < image, axis=2), axis=0)
    stitch_img = image[nonZeroRows][:, nonZeroCols]
    return stitch_img


def find_displacement(matched, baseline, focal_length_pixels):
  displacements_pixels = []
  displacements_meters = []
  depths_meters = []
  for (leftPoints, rightPoints) in matched:
    xL, yL = leftPoints
    xR, yR = rightPoints
    #Calculate pixel displacement (Euclidean distance)
    pixel_displacement = np.sqrt((xL - xR) ** 2 + (yL - yR) ** 2)
    displacements_pixels.append(pixel_displacement)
    #Calculate disparity (horizontal displacement)
    disparity = abs(xL - xR)
    if disparity > 0:  # Avoid division by zero
        # Depth: Z = (f*B)/d
        depth = (focal_length_pixels * baseline) / disparity
        depths_meters.append(depth)
        #meter displacement: (pixel_displacement / f) * depth
        meter_displacement = (pixel_displacement / focal_length_pixels) * depth
        displacements_meters.append(meter_displacement)
    else:
        depths_meters.append(0)  # Handle zero disparity
        displacements_meters.append(0)
  # Analyze results
  if displacements_pixels and depths_meters:
      avg_displacement_pixels = np.mean(displacements_pixels)
      min_displacement_pixels = np.min(displacements_pixels)
      max_displacement_pixels = np.max(displacements_pixels)
      avg_displacement_meters = np.mean([d for d in displacements_meters if d > 0])
      min_displacement_meters = np.min([d for d in displacements_meters if d > 0])
      max_displacement_meters = np.max([d for d in displacements_meters if d > 0])
      avg_depth_meters = np.mean([d for d in depths_meters if d > 0])
      min_depth_meters = np.min([d for d in depths_meters if d > 0])
      max_depth_meters = np.max([d for d in depths_meters if d > 0])

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


# Object Recognition/Segmentation
# https://huggingface.co/docs/transformers/en/model_doc/segformer

def get_building_mask(processor, rgb_small):
    building_words = ["building", "house", "skyscraper", "tower", "castle", "palace", "apartment"]
    height, width = rgb_small.shape[:2]
    pil_img = Image.fromarray(rgb_small)

    # Preprocess
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits  # [1, num_classes, height, width]

    logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),  
        mode="bilinear",
        align_corners=False
    )

    # Class prediction per pixel
    pred = logits.argmax(dim=1)[0].cpu().numpy()
    id2label = model.config.id2label
    # Find all class IDs that look like buildings
    building_ids = [
        i for i, name in id2label.items()
        if any(word in name.lower() for word in building_words)
    ]

    building_mask = np.isin(pred, building_ids).astype("uint8") * 255

    return building_mask


def building_edges(building_mask, rgb_image, low_thresh=100, high_thresh=200):
    gray = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, low_thresh, high_thresh)

    # Restrict edges to building mask pixels
    masked_edges = cv.bitwise_and(edges, edges, mask=building_mask)
    return masked_edges

def unmasked_matches(matched, mask):
    #match mask against left image
    ptsL = []
    ptsR = []
    height, width = mask.shape
    allowed_labels = {0}
    filtered_matches = []

    for (cL, cR) in matched:
        uL, vL = cL

        # Ensure pixel is inside mask bounds
        if not (0 <= vL < height and 0 <= uL < width):
            continue  # skip out-of-bounds
        label = mask[vL, uL]
        if label in allowed_labels:
            filtered_matches.append([cL, cR])
    return filtered_matches



def triangulate_points(pts1, pts2, focal_length_pixels, baseline_m, image_shape):
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

    # Z = f * B / d
    Z = (focal_length_pixels * baseline_m) / disparity_valid

    # X and Y back-projection
    X = (uL_valid - cx) * Z / focal_length_pixels
    Y = (vL_valid - cy) * Z / focal_length_pixels

    pts3d = np.zeros((pts1.shape[0], 3), dtype=np.float64)
    pts3d[valid_mask, 0] = X
    pts3d[valid_mask, 1] = Y
    pts3d[valid_mask, 2] = Z

    return pts3d, valid_mask

#get points within the mask
def building_matches(matched, building_mask):
    height, width = building_mask.shape[:2]
    filtered = []
    for (cL, cR) in matched:
        uL, vL = cL
        # ensure in bounds
        if 0 <= vL < height and 0 <= uL < width:
            if building_mask[vL, uL] > 0:  # inside building
                filtered.append((cL, cR))

    return filtered





#Height Estimation

"""
Height estimation Plan:
1. Pick a vertical line within the building mask
2. Estimate the depth of the line (distance from camera) by comparing nearby points
3. Calculate a ground plane that intersects with the base of the building.
4. Get the projected height of the line
"""
# Attempted PCA on building, was not an accurate way to get verticality

def vertical_line_through_point(building_mask, u0, v0, min_run=10):
    height, width = building_mask.shape[:2]
    u0 = int(round(u0))
    v0 = int(round(v0))

    if not (0 <= u0 < width and 0 <= v0 < height):
        return None, None
    if building_mask[v0, u0] == 0:
        return None, None

    # move up
    v_top = v0
    while v_top - 1 >= 0 and building_mask[v_top - 1, u0] > 0:
        v_top -= 1

    # move down
    v_bottom = v0
    while v_bottom + 1 < height and building_mask[v_bottom + 1, u0] > 0:
        v_bottom += 1

    if (v_bottom - v_top) < min_run:
        return None, None

    return v_top, v_bottom

#find point candidates near horizontal position
def pick_known_point(pts2d, pts3d, building_mask, horizontal_target=None, horiz_radius=20):
    pts2d = np.asarray(pts2d, dtype=np.float64)
    pts3d = np.asarray(pts3d, dtype=np.float64)
    num_points = len(pts2d)
    
    height, width = building_mask.shape[:2]

    if horizontal_target is None:
        horizontal_target = width *.5

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

        # primary score: vertical distance to center
        score = abs(vi - v_center)
        # small horizontal penalty
        score += 0.3 * abs(ui - horizontal_target)
        candidates.append((score, i))

    if not candidates:
        return None, None

    # choose the candidate closest to vertical center
    candidates.sort(key=lambda x: x[0])
    _, best_idx = candidates[0]
    return pts2d[best_idx], pts3d[best_idx]


def project_point(point, focal_length_pixels, image_shape):

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

#an alternate method of finding the building bottom in 3d space
def ground_plane_intersect(known_3d, plane_normal, plane_d, focal_length_pixels, image_shape):
    
    X0 = np.asarray(known_3d, dtype=np.float64)
    n = np.asarray(plane_normal, dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-12)

    # signed distance from point to plane (plane normal assumed "up")
    dist = np.dot(n, X0) + plane_d

    # foot of perpendicular on plane
    X_ground = X0 - dist * n

    return project_point(X_ground, focal_length_pixels, image_shape)

#estimate points in vertical slice of mask
def estimate_heights( building_mask, pts2d_valid, pts3d_valid,
    focal_length_pixels,
    plane_normal,
    plane_d,
    horiz_radius=25,
    min_run=10 ):

    height, width = building_mask.shape[:2]

    pts2d_valid = np.asarray(pts2d_valid, dtype=np.float64)
    pts3d_valid = np.asarray(pts3d_valid, dtype=np.float64)
    num_points = min(len(pts2d_valid), len(pts3d_valid))
    pts2d_valid = pts2d_valid[:num_points]
    pts3d_valid = pts3d_valid[:num_points]

    fractions = [.2, .4, .5, .6, .8]
    lines = []

    for frac in fractions:
        horizontal_target = width * frac

        # pick a known point near this horizontal position
        known_2d, known_3d = pick_known_point(
            pts2d_valid, pts3d_valid,
            building_mask,
            horizontal_target=horizontal_target,
            horiz_radius=horiz_radius,
        )
        if known_2d is None:
            continue  # no suitable point found is ok

        u0, v0 = known_2d
        X0, Y0, Z0 = known_3d

        if Z0 <= 0:
            continue  # invalid depth

        # get vertical line through that point in the mask
        v_top, v_bottom = vertical_line_through_point(building_mask,
            u0, v0, min_run=min_run )
        if v_top is None:
            continue

        #get ground intersect instead of using v_bottom
        bottom = ground_plane_intersect(known_3d,
            plane_normal,
            plane_d,
            focal_length_pixels,
            image_shape=building_mask.shape)

        u_bottom, v_bottom = bottom

        # compute height above plane
        plane_normal = np.asarray(plane_normal, dtype=np.float64)
        plane_normal /= (np.linalg.norm(plane_normal) + 1e-12)
        height_m = abs(np.dot(plane_normal, known_3d) + plane_d)

        # store: height, top point, bottom point
        top_pt = (int(round(u0)), int(v_top))
        bottom_pt = (int(round(u0)), int(v_bottom))
        lines.append((height_m, top_pt, bottom_pt))

    return lines

def draw_building_height_lines(image_bgr, lines, line_color=(0, 0, 255), thickness=2 ):

    img = image_bgr.copy()
    for (height_m, top_pt, bottom_pt) in lines:
        cv.line(img, top_pt, bottom_pt, line_color, thickness)
        cv.circle(img, top_pt, 5, (255, 0, 0), -1)
        cv.circle(img, bottom_pt, 5, (0, 255, 0), -1)
        cv.putText(img, f"{height_m:.2f} m", (top_pt[0] + 5, top_pt[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX, 1,  line_color, 2,cv.LINE_AA)
    return img

#Estimate ground plane

def get_ground_mask(processor, rgb_small):

    ground_words = [ "road", "street", "sidewalk", "pavement", "ground", "earth", "grass", "field", "path" ]

    height, width = rgb_small.shape[:2]
    pil_img = Image.fromarray(rgb_small)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    # Class prediction per pixel
    pred = logits.argmax(dim=1)[0].cpu().numpy()
    id2label = model.config.id2label

    ground_ids = [
        i for i, name in id2label.items()
        if any(word in name.lower() for word in ground_words)
    ]

    ground_mask = np.isin(pred, ground_ids).astype("uint8") * 255
    return ground_mask

def ground_matches(matched, ground_mask):
    height, width = ground_mask.shape[:2]
    filtered = []
    for (cL, cR) in matched:
        uL, vL = cL
        if 0 <= vL < height and 0 <= uL < width:
            if ground_mask[vL, uL] > 0:  # inside ground region
                filtered.append((cL, cR))
    return filtered

#fit for horizontal plane
def fit_plane_ransac(points, num_iterations=1000, distance_thresh=0.05):

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
        # sample 3 points
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]

        # compute normal
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            continue
        n = n / norm_n

        #ignore vertical results
        cos_angle = np.clip(np.dot(n, yplus), -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(cos_angle)))
        if angle > 40:
            continue

        d = -np.dot(n, p1)

        # distances
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

    # least-squares on all inliers
    P_in = points[best_inliers]
    centroid = P_in.mean(axis=0)
    Q = P_in - centroid
    U, S, Vt = np.linalg.svd(Q)
    n_refined = Vt[-1]  # last singular vector
    n_refined = n_refined / np.linalg.norm(n_refined)
    d_refined = -np.dot(n_refined, centroid)

    return n_refined, d_refined, best_inliers

# better keypoints for ground-like areas
def ground_specific_keypoints_and_descriptors(leftIm, rightIm,
        max_corners=3000, quality=0.01, min_distance=5, orb_features=3000):

    grayL = cv.cvtColor(leftIm, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(rightIm, cv.COLOR_BGR2GRAY)
    hL, wL = grayL.shape
    hR, wR = grayR.shape

    # limit to bottom half of image
    ground_mask_left  = np.zeros((hL, wL), dtype=np.uint8)
    ground_mask_right = np.zeros((hR, wR), dtype=np.uint8)

    ground_mask_left[hL//2 :, :]  = 255
    ground_mask_right[hR//2:, :] = 255

    kL_features = []
    kR_features = []

    cornersL = cv.goodFeaturesToTrack(grayL,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        mask=ground_mask_left )
    if cornersL is not None:
        cornersL = cornersL.reshape(-1, 2)
        kL_features = [cv.KeyPoint(float(u), float(v), 16) for (u, v) in cornersL]

    cornersR = cv.goodFeaturesToTrack(grayR,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        mask=ground_mask_right )
    if cornersR is not None:
        cornersR = cornersR.reshape(-1, 2)
        kR_features = [cv.KeyPoint(float(u), float(v), 16) for (u, v) in cornersR]

    # get ORB descriptors
    orb = cv.ORB_create(nfeatures=orb_features)

    kL_final, dL = orb.compute(grayL, kL_features)
    kR_final, dR = orb.compute(grayR, kR_features)

    print(f"Ground; Left descriptors: {len(dL)}; Right descriptors: {len(dR)}")

    return kL_final, dL, kR_final, dR

def estimate_ground_plane_from_mask(leftIm,rightIm,
    ground_mask, baseline_m,
    focal_length_pixels,
    max_corners=3000,
    quality=0.01,
    min_distance=5,
    ransac_thresh=0.05 ):

    kL, dL, kR, dR = ground_specific_keypoints_and_descriptors(leftIm,rightIm,
        max_corners=max_corners, quality=quality, min_distance=min_distance, orb_features=3000)

    g_matched = find_matches(dL, dR, kL, kR)
    ground_matched = ground_matches(g_matched, ground_mask)

    if len(ground_matched) < 3:
        raise ValueError(f"Not enough ground matches after masking: {len(ground_matched)}")

    # Build arrays of corresponding 2D points
    pts1_ground = np.array([m[0] for m in ground_matched], dtype=np.float64)  # left
    pts2_ground = np.array([m[1] for m in ground_matched], dtype=np.float64)  # right

    pts3d_ground, valid_ground = triangulate_points(pts1_ground, pts2_ground,
        focal_length_pixels=focal_length_pixels, baseline_m=baseline_m, image_shape=leftIm.shape )

    #filter points
    pts3d_ground_valid = pts3d_ground[valid_ground]
    pts2d_ground_valid = pts1_ground[valid_ground]

    if pts3d_ground_valid.shape[0] < 3:
        raise ValueError(f"Not enough valid 3D ground points: {pts3d_ground_valid.shape[0]}")

    # try depth filtering 
    z = pts3d_ground_valid[:, 2]
    depth_mask = (z > 1.0) & (z < 40.0)
    pts3d_ground_valid = pts3d_ground_valid[depth_mask]
    pts2d_ground_valid = pts2d_ground_valid[depth_mask]

    n_ground, d_ground, inliers_ground = fit_plane_ransac(pts3d_ground_valid, distance_thresh=ransac_thresh )

    return n_ground, d_ground, inliers_ground, pts2d_ground_valid, pts3d_ground_valid


def visualize_ground_plane_on_image(
    image_bgr,
    pts2d_ground_valid,
    pts3d_ground_valid,
    inlier_mask,
    plane_normal,
    plane_d,
    focal_length_pixels,
    grid_extent_m=5.0,
    grid_step_m=1.0,
):
    """
    Overlay:
      1) All inlier ground points used by RANSAC.
      2) A 3D grid that lies on the discovered ground plane,
         projected into the original image.

    Args:
        image_bgr: (H,W,3) original left image (BGR).
        pts2d_ground_valid: (N,2) 2D pixels for ground points (left image).
        pts3d_ground_valid: (N,3) 3D points (meters) in camera coordinates.
        inlier_mask: (N,) bool array – which points are inliers to the plane.
        plane_normal: (3,) plane normal n in n·X + d = 0.
        plane_d: scalar d.
        focal_length_pixels: fx = fy (pixels).
        grid_extent_m: half-size of grid in plane local coords (meters).
        grid_step_m: spacing between grid lines (meters).

    Returns:
        img_out: BGR image with overlays.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]

    fx = fy = float(focal_length_pixels)
    cx = w / 2.0
    cy = h / 2.0

    pts2d_ground_valid = np.asarray(pts2d_ground_valid, dtype=np.float64)
    pts3d_ground_valid = np.asarray(pts3d_ground_valid, dtype=np.float64)
    inlier_mask = np.asarray(inlier_mask, dtype=bool)

    # ------------------------------------------------------------------
    # 1) Draw all inlier ground points on the image
    # ------------------------------------------------------------------
    inlier_indices = np.where(inlier_mask)[0]

    for idx in inlier_indices:
        u, v = pts2d_ground_valid[idx]
        u_i = int(round(u))
        v_i = int(round(v))
        if 0 <= u_i < w and 0 <= v_i < h:
            # red blobs = inliers used for ground plane
            cv.circle(img, (u_i, v_i), 4, (0, 0, 255), -1)

    # If not enough inliers, just return the image with points
    if len(inlier_indices) < 3:
        cv.putText(img, "Not enough inliers to draw ground plane grid",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255), 2, cv.LINE_AA)
        return img

    # ------------------------------------------------------------------
    # 2) Build a 2D grid on the plane and project it
    # ------------------------------------------------------------------
    plane_normal = np.asarray(plane_normal, dtype=np.float64)
    n_hat = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)

    # Use the centroid of inlier 3D points as grid center (p0)
    P_inliers = pts3d_ground_valid[inlier_mask]
    p0 = P_inliers.mean(axis=0)

    # Construct an orthonormal basis (u_dir, v_dir) lying in the plane
    # Pick an arbitrary vector not parallel to n_hat
    tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(np.dot(tmp, n_hat)) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    u_dir = np.cross(n_hat, tmp)
    u_dir /= np.linalg.norm(u_dir) + 1e-12
    v_dir = np.cross(n_hat, u_dir)
    v_dir /= np.linalg.norm(v_dir) + 1e-12

    # Grid coordinates in plane local (s,t)
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

    # Draw grid lines of constant s
    for s in s_vals:
        pts_line = []
        for t in t_vals:
            X = p0 + s * u_dir + t * v_dir  # point on plane
            uv = project_point(X)
            if uv is not None:
                pts_line.append(uv)

        if len(pts_line) >= 2:
            pts_np = np.array(pts_line, dtype=np.int32).reshape(-1, 1, 2)
            cv.polylines(img, [pts_np], False, (0, 255, 0), 1, cv.LINE_AA)

    # Draw grid lines of constant t
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

    # Optional: annotate number of inliers
    cv.putText(img,
               f"Ground inliers: {len(inlier_indices)}",
               (10, 30),
               cv.FONT_HERSHEY_SIMPLEX,
               0.7,
               (255, 255, 255),
               2,
               cv.LINE_AA)

    return img

def debug_plane_orientation(n_ground):
    n_ground = np.asarray(n_ground, dtype=np.float64)
    n_ground /= np.linalg.norm(n_ground) + 1e-12

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # camera "up" axis in your triangulation
    cos_angle = np.clip(np.dot(n_ground, up), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(abs(cos_angle)))
    print("Plane normal:", n_ground)
    print("Angle between plane normal and up-axis:", angle_deg, "deg")
