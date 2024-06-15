import cv2
import numpy as np
import pandas as pd
from rgb_comparator import compare_rgb
import random


# def compare_rgb(moved_piece, target_rgb, region_path):
#     bg_sample_path = os.path.join(region_path, 'piece_colored_regions.png')
#     bg_sample = cv2.imread(bg_sample_path, cv2.IMREAD_COLOR)
#
#     if bg_sample is None:
#         raise ValueError(f"Image at path {bg_sample_path} could not be loaded.")
#
#     if moved_piece.shape != bg_sample.shape:
#         raise ValueError("Piece image and full image must have the same dimensions")
#
#     mask = cv2.inRange(moved_piece, np.array([1, 1, 1]), np.array([255, 255, 255]))
#     filled_piece = moved_piece.copy()
#     filled_piece[mask > 0] = target_rgb
#
#     piece_rgb = cv2.bitwise_and(filled_piece, filled_piece, mask=mask)
#     full_rgb = cv2.bitwise_and(bg_sample, bg_sample, mask=mask)
#
#     intersection = np.logical_and(np.all(piece_rgb == target_rgb, axis=-1), np.all(full_rgb == target_rgb, axis=-1))
#     intersection_count = np.sum(intersection)
#
#     union = np.logical_or(np.all(piece_rgb == target_rgb, axis=-1), np.all(full_rgb == target_rgb, axis=-1))
#     union_count = np.sum(union)
#
#     iou = intersection_count / union_count if union_count != 0 else 0
#
#     return iou

def inserting_puzzle(piece_pts, piece_kp, gt_pts, gt_kp, good_matches, piece, bg, idx, region_path):
    dst_temp = cv2.drawMatches(piece, piece_kp, bg, gt_kp, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    RT = 3.0  # RANSAC Threshold
    M, inliers = cv2.estimateAffinePartial2D(piece_pts, gt_pts, method=cv2.RANSAC, ransacReprojThreshold=RT)

    if M is None:
        print("Transformation matrix could not be estimated")
        centroid = np.mean(gt_pts, axis=0)
        return bg, dst_temp, centroid, 0

    inliers = list(inliers.reshape(-1)) if inliers is not None else []

    if piece_pts.ndim < 2 or gt_pts.ndim < 2:
        print("Insufficient dimensions in points array")
        centroid = np.mean(gt_pts, axis=0)
        return bg, dst_temp, centroid, 0

    if len(piece_pts) < 3:
        print('Not enough key points or matching points')
        centroid = np.mean(gt_pts, axis=0)
        return bg, dst_temp, centroid, 0

    good_matches_np = np.array(good_matches)
    ransac_matches = good_matches_np[inliers]
    ransac_matches = ransac_matches.tolist()

    dst = cv2.drawMatches(piece, piece_kp, bg, gt_kp, ransac_matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    solved_piece = cv2.warpAffine(piece, M, (bg.shape[1], bg.shape[0]))

    area_pts = np.where(solved_piece > 0)
    centroid = [np.mean(area_pts[0]), np.mean(area_pts[1])]

    groud_truth = pd.read_csv(region_path + 'piece_puzzle_colors.csv')

    color = groud_truth.loc[idx, ['B', 'G', 'R']].tolist()
    iou = compare_rgb(solved_piece, color, region_path)

    print(f"Piece {idx + 1} IoU: {iou * 100:.2f}%")

    bg = cv2.add(bg, solved_piece)  # Use cv2.add for better image blending
    return bg, dst, centroid, iou

# def inserting_puzzle(piece_pts, gt_pts, piece, bg):
#     if (len(piece_pts) < 3):
#         print('Not enough key points or matching points')
#         no_matching = np.array([-1, -1])
#         return bg, no_matching
#
#     M = cv2.getAffineTransform(piece_pts[:3], gt_pts[:3])
#     if M.all() == True:
#         predict_pts_avg = np.mean(gt_pts[:3], axis=0)
#
#     while (M.all() == False):
#         pts_list = list(np.arange(len(piece_pts)))
#         random_pts = random.sample(pts_list, 3)
#         M = cv2.getAffineTransform(piece_pts[random_pts], gt_pts[random_pts])
#         predict_pts_avg = np.mean(gt_pts[random_pts], axis=0)
#
#     transformed_piece = cv2.warpAffine(piece, M, (bg.shape[1], bg.shape[0]))
#     bg = bg + transformed_piece
#
#     return bg, predict_pts_avg


# def mapping(predict_pts_avg, H_list, W_list):
#     if predict_pts_avg[0] == -1:
#         return -1, -1
#
#     for idx, H in enumerate(H_list):
#         if idx == 0:
#             continue
#         if predict_pts_avg[1] < H and predict_pts_avg[1] >= H_list[idx - 1]:
#             row = idx - 1
#             break
#
#     for idx, W in enumerate(W_list):
#         if idx == 0:
#             continue
#         if predict_pts_avg[0] < W and predict_pts_avg[0] >= W_list[idx - 1]:
#             col = idx - 1
#             break
#
#     return row, col
