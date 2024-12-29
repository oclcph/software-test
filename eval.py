import logging
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')  # 使用适合你的YOLO模型
# 禁用所有日志输出
logging.disable(logging.CRITICAL)
# 获取类别名称
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# annotations_path = Path('E:/LoLI-Street/data/dataset/LoLI-Street Dataset/YOLO Annotations/Val/YOLO Annotations (high)/Labels')

# 设置验证数据集路径

def calculate_iou(pred_bbox, true_bbox, image_width, image_height):
    # 将预测框坐标从像素转换为相对坐标
    pred_x_center = pred_bbox[0]
    pred_y_center = pred_bbox[1]
    pred_width = pred_bbox[2]
    pred_height = pred_bbox[3]

    true_x_center = true_bbox[0]
    true_y_center = true_bbox[1]
    true_width = true_bbox[2]
    true_height = true_bbox[3]

    # 计算预测框和真实框的四个角的坐标（归一化）
    pred_x1 = pred_x_center - pred_width / 2
    pred_y1 = pred_y_center - pred_height / 2
    pred_x2 = pred_x_center + pred_width / 2
    pred_y2 = pred_y_center + pred_height / 2

    true_x1 = true_x_center - true_width / 2
    true_y1 = true_y_center - true_height / 2
    true_x2 = true_x_center + true_width / 2
    true_y2 = true_y_center + true_height / 2

    # 计算交集
    inter_x1 = max(pred_x1, true_x1)
    inter_y1 = max(pred_y1, true_y1)
    inter_x2 = min(pred_x2, true_x2)
    inter_y2 = min(pred_y2, true_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    true_area = (true_x2 - true_x1) * (true_y2 - true_y1)

    union_area = pred_area + true_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def compare_results(image_file):
    image = cv2.imread(str(image_file))
    image_height, image_width, _ = image.shape  # 获取图像的宽度和高度

    # 进行YOLO推理
    results = model(image)

    # 获取YOLO推理的结果
    preds = results[0].boxes  # 获得框的位置和类别信息
    pred_labels = preds.cls.cpu().numpy().astype(int)
    pred_bboxes = preds.xywh.cpu().numpy()  # 获取相对坐标: [x_center, y_center, width, height]

    # 获取标注文件路径
    label_file = annotations_path / (image_file.stem + '.txt')  # 使用 Path 对象进行路径拼接
    if not label_file.exists():
        return None  # 如果没有对应的标签文件，跳过

    # 读取标注
    true_labels = []
    true_bboxes = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            true_labels.append(int(parts[0]))  # 类别标签
            true_bboxes.append(list(map(float, parts[1:])))  # 边界框位置（归一化的）

    # 计算IOU并判断预测结果是否正确
    correct = 0
    total_pred = len(pred_labels)
    total_true = len(true_labels)

    for i in range(min(total_pred, total_true)):  # 比较最小数量的预测框和标注框
        pred_label = pred_labels[i]
        pred_bbox = pred_bboxes[i]
        true_label = true_labels[i]
        true_bbox = true_bboxes[i]

        # 检查是否类别匹配，且计算IOU
        if pred_label == true_label:
            iou = calculate_iou(pred_bbox, true_bbox, image_width, image_height)
            # print(f'Predicted IoU: {iou:.4f}, Predicted: {pred_bbox}, True: {true_bbox}')
            if iou > 0.5:
                correct += 1

    return correct / total_pred if total_pred > 0 else 0  # 返回正确率

if __name__ == '__main__':
    val_path = 'output'  # 低光照验证集路径
    image_files = list(Path(val_path).glob('*.jpg'))
    print(f"Total number of images in the validation set: {len(image_files)}")
# 设置标注文件路径
    accuracies = []
    for image_file in image_files:
        accuracy = compare_results(image_file)
        if accuracy is not None:
            accuracies.append(accuracy)


# 输出平均准确率
    if accuracies:
        print(f"Average accuracy: {np.mean(accuracies)}")
    else:
        print("No valid predictions to evaluate.")