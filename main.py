import argparse
import os
import pandas as pd
from puzzle_solver import solving
from utils import calculate_map, calculate_f1_score, calculate_ap_and_map


def main():
    """
    메인 함수
    명령 줄 인수로부터 입력을 받아 퍼즐 조각을 결합하는 과정을 수행합니다.
    """
    parser = argparse.ArgumentParser(description="Puzzle Solver")

    subparsers = parser.add_subparsers(dest='mode', help="Mode of operation")

    # demo 모드
    demo_parser = subparsers.add_parser('demo', help="Enable demo mode")
    demo_parser.add_argument('gt_path', type=str, help="Path to the ground truth image")
    demo_parser.add_argument('size', type=int, help="Size of the puzzle (number of pieces along one dimension)")

    # eval 모드
    eval_parser = subparsers.add_parser('eval', help="Enable eval mode")
    eval_parser.add_argument('folder_path', type=str, help="Path to the folder containing images")

    args = parser.parse_args()

    opt_detector = "ORB"  # 특징 검출기로 SIFT 선택
    opt_matcher = "BF"  # 특징 매칭기로 BFMatcher 선택

    if args.mode == 'demo':
        gt_path, size = args.gt_path, args.size
        img_name = gt_path.split('/')[-1].split('.')[0]
        solving(opt_detector, opt_matcher, img_name, size, gt_path, demo=True)

    if args.mode == 'eval':
        folder_path = args.folder_path
        print(f"Eval mode with folder: {folder_path}")

        results = []

        # 이미지 파일들 읽기
        for img_file in os.listdir(folder_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_name, img_ext = os.path.splitext(img_file)
                img_name = img_name.split('.')[0]  # 파일 이름에서 확장자 제거

                # 해당 이미지의 폴더 찾기
                gt_path = os.path.join(folder_path, img_file)
                sizes = []
                for subfolder in os.listdir(folder_path.replace("puzzle_images", "test_pieces")):
                    if subfolder.startswith(img_name):
                        try:
                            size = int(subfolder.split('_')[-1])
                            sizes.append(size)
                        except ValueError:
                            continue

                if gt_path and sizes:
                    for size in sizes:
                        print(img_name, size)
                        label, predict, iou_scores = solving(opt_detector, opt_matcher, img_name, size, gt_path,
                                                             demo=False)
                        map_value = calculate_ap_and_map(iou_scores, 0.85, size*size)
                        f1_value = calculate_f1_score(label, predict)

                        # 결과를 리스트에 추가
                        results.append({
                            'Image_Name': img_name,
                            'The Number of Puzzles': size * size,
                            'F1-Score': round(f1_value, 2),
                        #    'AP' : round(ap, 2),
                            'mAP@0.85': round(map_value, 2)
                        })
                else:
                    print(f"Could not find matching folder for image {img_file}")

        # 결과를 DataFrame으로 변환하고 CSV 파일로 저장
        df = pd.DataFrame(results)
        df.to_csv('puzzle_results.csv', index=False)
        print("Results saved to puzzle_results.csv")


if __name__ == '__main__':
    main()
