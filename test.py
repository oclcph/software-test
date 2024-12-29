import csv

import cv2
import numpy as np
from pathlib import Path
from eval import model, calculate_iou
from blur import input_folder, output_base_folder

def inference(image_file, model):
    image = cv2.imread(str(image_file))
    image_height, image_width, _ = image.shape  # 获取图像的宽度和高度

    results = model(image)

    # 获取YOLO推理的结果
    preds = results[0].boxes  # 获得框的位置和类别信息
    pred_labels = preds.cls.cpu().numpy().astype(int)
    pred_bboxes = preds.xywh.cpu().numpy()  # 获取相对坐标: [x_center, y_center, width, height]

    return pred_labels, pred_bboxes, image_height, image_width

def cal(pred, org):
    pred_labels, pred_bboxes, image_height, image_width = pred
    true_labels, true_bboxes,_, _ = org
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

    image_files = list(Path(input_folder).glob('*.jpg'))
    print(f"Total number of images in the validation set: {len(image_files)}")
    # 设置标注文件路径
    acc = {}
    accuracies_ori = []
    for image_file in image_files:
        origin_result = inference(image_file, model)

        origin_result2 = inference(image_file, model)
        accuracy2 = cal(origin_result, origin_result2)
        if accuracy2 is not None:
            accuracies_ori.append(accuracy2)
    std_mean_accuracy = np.mean(accuracies_ori)

    acc['std'] = std_mean_accuracy
    print(std_mean_accuracy)

    ops = ['blur_3_3', 'blur_5_5', 'blur_7_7', 'blur_9_9', 'blur_11_11', 'blur_15_15', 'brighter', 'motion_blur', 'blacker',
           'noise', 'occlusion_circle', 'occlusion_rectangle', 'resize', 'jpeg_compressed']
    for op in ops:
        acc_op = []
        for image_file in image_files:
            filepath = output_base_folder + '/' + op + '/' + image_file.parts[-1]
            result = inference(filepath, model)
            origin_result = inference(image_file, model)
            acc_op.append(cal(result, origin_result))
        acc[op] = np.mean(acc_op)
        print(acc[op])
    print(acc)

    with open('res.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(acc.keys())
        # 写入数据
        writer.writerow(acc.values())
