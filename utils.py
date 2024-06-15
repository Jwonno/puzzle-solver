import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

def calculate_precision_recall_at_threshold(iou_values, threshold):
    iou_values = np.array(iou_values)
    tp = np.sum(iou_values >= threshold)
    fp = np.sum(iou_values < threshold)
    fn = len(iou_values) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall

def compute_ap(precisions, recalls):
    precisions = np.concatenate(([0.], precisions, [0.]))
    recalls = np.concatenate(([0.], recalls, [1.]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def calculate_map(iou_list_per_image):
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []

    for threshold in iou_thresholds:
        all_precisions = []
        all_recalls = []

        for iou_values in iou_list_per_image:
            if not isinstance(iou_values, (list, np.ndarray)):
                iou_values = [iou_values]
            precision, recall = calculate_precision_recall_at_threshold(iou_values, threshold)
            all_precisions.append(precision)
            all_recalls.append(recall)

        ap = compute_ap(np.array(all_precisions), np.array(all_recalls))
        aps.append(ap)

    mAP = np.mean(aps)
    return mAP

def calculate_f1_score(label, predict):
    # TP, FP, FN을 계산하기 위한 초기화
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    filtered_labels = []
    filtered_predictions = []

    for lbl, pred in zip(label, predict):
        if pred != -1:
            filtered_labels.append(lbl)
            filtered_predictions.append(pred)

            if lbl == pred:
                true_positives += 1
            else:
                false_positives += 1
        else:
            false_negatives += 1

    # scikit-learn의 F1-Score 계산 (micro 평균 사용)
    f1 = f1_score(filtered_labels, filtered_predictions, average='micro', zero_division=0)

    return f1


# def calculate_ap_and_map(iou_values, threshold):
#     """
#     주어진 IoU 값들과 threshold를 기반으로 AP와 mAP를 계산하는 함수
#
#     Parameters:
#     iou_values (list): IoU 값들의 리스트
#     threshold (float): 양성 예측으로 간주하기 위한 IoU threshold 값
#
#     Returns:
#     tuple: AP와 mAP 값
#     """
#     # TP, FP, FN 계산
#     tp = np.sum(np.array(iou_values) >= threshold)
#     fp = np.sum(np.array(iou_values) < threshold)
#     fn = 0  # ground truth가 주어지지 않았으므로 FN은 0으로 간주
#
#     # Precision, Recall 계산
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # fn은 0이므로 recall은 1
#
#     # AP는 Precision과 Recall이 동일한 경우의 값을 그대로 사용
#     ap = precision
#
#     # mAP 계산
#     map_value = ap  # 단일 클래스이므로 AP가 mAP와 동일
#
#     return ap, map_value


def calculate_ap_and_map(iou_values, threshold, num_classes):
    """
    주어진 IoU 값들과 threshold를 기반으로 AP와 mAP를 계산하는 함수

    Parameters:
    iou_values (list): 각 클래스별 IoU 값들의 리스트
    threshold (float): 양성 예측으로 간주하기 위한 IoU threshold 값
    num_classes (int): 클래스의 총 개수

    Returns:
    float: mAP 값
    """
    aps = []

    for i in range(num_classes):
        iou = iou_values[i] if i < len(iou_values) else 0

        tp = 1 if iou >= threshold else 0
        fp = 1 if iou < threshold else 0
        fn = 0  # ground truth가 주어지지 않았으므로 FN은 0으로 간주

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        ap = precision
        aps.append(ap)

    map_value = np.mean(aps)  # 모든 클래스의 AP 평균

    return map_value