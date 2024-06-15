import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import os
import imutils


def transform_piece(img):
    '''
    퍼즐 조각 이미지를 변형하는 함수
    이미지에 0.5~1 의 범위에서 무작위 값으로 스케일링을 적용하고
    0~180도의 범위에서 무작위 값으로 회전을 적용하여 변형시킨다.
    
    Parameters:
    - img: 퍼즐 조각 이미지
    
    Returns:
    - result_img: 변환이 적용된 이미지
    '''
    height, width = img.shape[:2]

    # 스케일링 적용
    scale_factor = random.uniform(0.5, 1)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    scaled_img = cv2.resize(img, (new_width, new_height))

    # 회전 변환 적용
    angle = random.uniform(0, 180)
    rotated_img = imutils.rotate_bound(scaled_img, angle)
    result_img = rotated_img

    return result_img


def dataset_comp(name, size):
    '''
    변형된 퍼즐 조각 이미지 데이터셋을 구성하는 함수
    
    Parameters:
    - name: 데이터셋을 구성할 이미지의 이름
    - size: 정사각행렬의 퍼즐로 분할된 영역의 행 또는 열의 길이 ex) 3x3->3, 5x5->5,...
    '''
    save_dir = './test_pieces/{0}_{1}/'.format(name, size)

    for idx in range(size ** 2):
        row, col = int(idx / size), int(idx % size)
        piece_path = './test_pieces/{}_{}/piece_{}_{}.png'.format(name, size, row, col)
        alpha = Image.open(piece_path).convert('RGBA')
        bgr = cv2.imread(piece_path)

        piece_alpha = np.array(alpha)
        piece_im = np.zeros_like(bgr)

        piece_mask = (piece_alpha[:, :, 3] != 0) * 255
        piece_mask = piece_mask[:, :, np.newaxis]
        piece_mask = np.repeat(piece_mask, 3, 2)

        piece_im = np.bitwise_and(piece_mask, bgr)

        piece_im = piece_im.astype(np.uint8)

        result_img = transform_piece(piece_im)

        save_path = save_dir + 'piece_{}.png'.format(idx)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imwrite(save_path, result_img)


def image_loader(name, size, gt_path):
    gt = cv2.imread(gt_path)
    piece_imgs, pieces_gray = [], []
    region_path = './test_pieces/{}_{}/'.format(name, size)

    for i in range(size ** 2):
        piece_path = './test_pieces/{}_{}/piece_{}.png'.format(name, size, i)
        piece_im = cv2.imread(piece_path)
        piece_gray = cv2.cvtColor(piece_im, cv2.COLOR_BGR2GRAY)
        piece_imgs.append(piece_im)
        pieces_gray.append(piece_gray)

    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    return piece_imgs, pieces_gray, gt, gt_gray, region_path
