import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from image_helper import image_loader
from puzzle_insert import inserting_puzzle
import random


def mapping(centroid, H_list, W_list):
    # 중점의 좌표가 범위를 벗어난 경우
    row = 0
    col = -1

    if centroid[0] == -1:
        return -1, -1

    for idx, H in enumerate(H_list):
        if centroid[0] <= H:
            row = idx - 1
            break

    for idx, W in enumerate(W_list):
        if centroid[1] <= W:
            col = idx - 1
            break

    return row, col

def solving(detector, matcher, img_name, size, gt_path, demo):
    """
    특징 추출기 함수
    주어진 이미지에서 특징을 추출하고 매칭을 수행합니다.
    """
    pieces, pieces_gray, gt, gt_gray, region_path = image_loader(img_name, size, gt_path)
    bg = np.zeros_like(gt)

    H, W = gt.shape[0], gt.shape[1]
    H_list = np.linspace(0, H, size+1)
    W_list = np.linspace(0, W, size+1)

    start = time.time()
    pieces_kp, pieces_des = [], []
    if detector == "SIFT":
        siftF = cv2.xfeatures2d.SIFT_create()
    elif detector == "ORB":
        siftF = cv2.ORB_create()
    elif detector == "BRISK":
        siftF = cv2.BRISK_create()
    elif detector == "KAZE":
        siftF = cv2.KAZE_create()
    else:
        print("Not supported Detector:", detector)

    for i in range(size ** 2):
        kp, des = siftF.detectAndCompute(pieces_gray[i], None)
        pieces_kp.append(kp)
        pieces_des.append(des)
    gt_kp, gt_des = siftF.detectAndCompute(gt_gray, None)
    print("Detector and Descriptor:", time.time() - start)

    start = time.time()
    pieces_matches = []

    # Matcher 선택 및 적용
    if matcher == "BF":
        bf = cv2.BFMatcher()
        for i in range(size ** 2):
            matches = bf.knnMatch(pieces_des[i], gt_des, k=2)
            pieces_matches.append(matches)
    elif matcher == "BF_L1":
        bf = cv2.BFMatcher(cv2.NORM_L1)
        for i in range(size ** 2):
            matches = bf.knnMatch(pieces_des[i], gt_des, k=2)
            pieces_matches.append(matches)
    elif matcher == "BF_L2":
        bf = cv2.BFMatcher(cv2.NORM_L2)
        for i in range(size ** 2):
            matches = bf.knnMatch(pieces_des[i], gt_des, k=2)
            pieces_matches.append(matches)
    elif matcher == "FLANN":
        if detector in ["ORB", "BRISK"]:
            raise ValueError("FLANN is not supported for binary descriptors with ORB or BRISK.")
        else:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            for i in range(size ** 2):
                matches = flann.knnMatch(pieces_des[i], gt_des, k=2)
                pieces_matches.append(matches)
    else:
        raise ValueError(f"Not supported Matcher: {matcher}")

    print("Matching:", time.time() - start)

    distance_ratio = 0.5
    pieces_dst = []
    pts = {}
    pts['piece'], pts['gt'] = [], []

    predict = []
    iou_scores = []

    # 랜덤으로 퍼즐 조각을 샘플링하여 맞추기 위한 퍼즐 조각 순서 정답 리스트
    label = list(np.arange(size**2))
    random.shuffle(label)

    for idx in label:
        matches_p = pieces_matches[idx]
        good_matches = [f1 for f1, f2 in matches_p if f1.distance < distance_ratio * f2.distance]
        good_matches.sort(key=lambda DMatch: DMatch.distance)

        if len(good_matches) == 0:
            print(f"No good matches for piece {idx}")
            predict.append(-1)
            iou_scores.append(0)
            continue

        piece_pts = np.float32([pieces_kp[idx][m.queryIdx].pt for m in good_matches])
        gt_pts = np.float32([gt_kp[m.trainIdx].pt for m in good_matches])

        if piece_pts.size == 0 or gt_pts.size == 0:
            print(f"Empty point arrays for piece {idx}")
            predict.append(-1)
            iou_scores.append(0)
            continue

        pts['piece'].append(piece_pts)
        pts['gt'].append(gt_pts)

        # 퍼즐 조각을 배경 이미지에 결합
        bg, dst, centroid, iou = inserting_puzzle(piece_pts, pieces_kp[idx], gt_pts, gt_kp, good_matches, pieces[idx], bg, idx, region_path)
        predict_row, predict_col = mapping(centroid, H_list, W_list)
        predict_num = predict_row*size + predict_col
        predict.append(predict_num)
        iou_scores.append(iou)

        # 매칭 결과 시각화
        #dst = cv2.drawMatches(pieces[idx], pieces_kp[idx], bg, gt_kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # 매칭점 시각화
        pieces_dst.append(dst)
        if demo:
            plt.figure(figsize=(20, 15))
            plt.imshow(dst[:, :, ::-1])
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

            # 퍼즐 조각이 맞춰진 배경 이미지 출력
            plt.figure(figsize=(20, 15))
            plt.imshow(bg[:, :, ::-1])
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

    if demo:
        plt.figure(figsize=(20,15))
        plt.imshow(bg[:, :, ::-1])
        plt.axis('off')
        plt.title('Final Result')
        plt.show()

    print('label: ', label)
    print('predict: ', predict)

    return label, predict, iou_scores
