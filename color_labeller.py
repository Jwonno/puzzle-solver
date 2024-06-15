import cv2
import numpy as np
import csv

def label_and_color_pieces(outline_path, color_output_path, label_output_path, csv_output_path):
    # 기준 이미지 로드
    outline_img = cv2.imread(outline_path, cv2.IMREAD_GRAYSCALE)

    # 이진화하여 흰색과 검정색으로 분리
    _, binary_img = cv2.threshold(outline_img, 128, 255, cv2.THRESH_BINARY)

    # 외곽선 검출
    _, contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컬러 이미지를 생성하여 각 영역을 채움
    color_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    label_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)

    # 각 퍼즐 조각의 중심 좌표와 컨투어를 저장할 리스트
    centroids_contours = []

    # 각 외곽선 내부를 무작위 색상으로 채움 및 중심 좌표 계산
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids_contours.append(((cX, cY), contour))

    # 중심 좌표를 기준으로 정렬 (왼쪽 상단부터 오른쪽 하단 순서대로)
    centroids_contours.sort(key=lambda x: (x[0][1] // 50, x[0][0] // 50))

    labels = []
    label_colors = []

    # 라벨링 및 컬러링
    for label, (centroid, contour) in enumerate(centroids_contours, 1):
        color = np.random.randint(0, 255, size=3).tolist()
        labels.append(label)
        label_colors.append(color)
        cv2.drawContours(color_img, [contour], -1, color, thickness=cv2.FILLED)
        cv2.putText(label_img, str(label), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 결과 이미지 표시
    combined_img = cv2.addWeighted(color_img, 0.7, label_img, 0.3, 0)

    cv2.imwrite(color_output_path, color_img)
    cv2.imwrite(label_output_path, combined_img)

    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title('Colored Regions')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title('Labeled Regions')
    # plt.tight_layout()
    # plt.show()

    # CSV 파일로 RGB 값 저장
    with open(csv_output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "B", "G", "R"])
        for label, color in zip(labels, label_colors):
            writer.writerow([label] + color)

    print(f"CSV 파일이 저장되었습니다: {csv_output_path}")