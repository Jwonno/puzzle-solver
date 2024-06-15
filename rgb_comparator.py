import cv2
import numpy as np
import os

def compare_rgb(moved_piece, target_rgb, region_path):
    bg_sample_path = os.path.join(region_path, 'piece_colored_regions.png')
    bg_sample = cv2.imread(bg_sample_path, cv2.IMREAD_COLOR)

    if bg_sample is None:
        raise ValueError(f"Image at path {bg_sample_path} could not be loaded.")

    # 이미지가 동일한 크기인지 확인
    if moved_piece.shape != bg_sample.shape:
        raise ValueError("Piece image and full image must have the same dimensions")

    # 조각 이미지에서 빈 공간(예: 검정색 배경)이 아닌 실제 색상이 있는 영역을 마스크로 생성
    mask = cv2.inRange(moved_piece, np.array([1, 1, 1]), np.array([255, 255, 255]))

    # 실제 색상이 있는 영역을 target_rgb로 채움
    filled_piece = moved_piece.copy()
    filled_piece[mask > 0] = target_rgb

    # 마스크를 사용하여 조각 이미지와 전체 이미지의 해당 영역을 비교
    piece_rgb = cv2.bitwise_and(filled_piece, filled_piece, mask=mask)
    full_rgb = cv2.bitwise_and(bg_sample, bg_sample, mask=mask)

    # 교집합 영역 계산
    intersection = np.logical_and(np.all(piece_rgb == target_rgb, axis=-1), np.all(full_rgb == target_rgb, axis=-1))
    intersection_count = np.sum(intersection)

    # 합집합 영역 계산
    union = np.logical_or(np.all(piece_rgb == target_rgb, axis=-1), np.all(full_rgb == target_rgb, axis=-1))
    union_count = np.sum(union)

    # IoU 계산
    iou = intersection_count / union_count if union_count != 0 else 0

    return iou