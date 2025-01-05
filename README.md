# 测试报告：YOLO模型在正常光照与低光照条件下的物体识别性能对比

---

## 测试目的
本次测试旨在评估YOLO模型在**特定场景**下的物体识别性能差异，主要通过Precision、Recall和F1-Score三个评估指标进行对比分析。

## 测试模型
本次测试使用了YOLOv8n作为测试模型。

## 测试数据
使用了LOFI_LoLI-Street Dataset中的部分数据进行测试,共9000张图像。
测试数据包含多个类的物体，涵盖了常见的交通工具、日常物品等对象。

实验过程主要有两个：
1. 测试分别在**正常光照**和**低光照环境**下进行，以评估YOLO模型在不同光照条件下的**鲁棒性**和**准确性**。
2. 利用扰动方法对test数据集进行扰动，包括添加模糊、添加遮挡、进行压缩等方法，计算相应的IOU，评估YOLO模型的**鲁棒性**。
---

## 测试结果

### 1. 正常光照条件下的识别结果

| Class         | Precision | Recall  | F1-Score |
|---------------|-----------|---------|----------|
| person        | 0.987860  | 0.995413 | 0.991622 |
| bicycle       | 1.000000  | 1.000000 | 1.000000 |
| car           | 0.995233  | 0.997515 | 0.996373 |
| motorcycle    | 0.983709  | 0.992415 | 0.988043 |
| airplane      | 1.000000  | 1.000000 | 1.000000 |
| bus           | 0.989437  | 0.989437 | 0.989437 |
| train         | 1.000000  | 1.000000 | 1.000000 |
| truck         | 0.997067  | 0.998532 | 0.997799 |
| boat          | 1.000000  | 1.000000 | 1.000000 |
| traffic light | 1.000000  | 1.000000 | 1.000000 |
| fire hydrant  | 1.000000  | 1.000000 | 1.000000 |
| stop sign     | 1.000000  | 1.000000 | 1.000000 |
| bench         | 1.000000  | 1.000000 | 1.000000 |
| backpack      | 1.000000  | 1.000000 | 1.000000 |
| umbrella      | 1.000000  | 1.000000 | 1.000000 |
| handbag       | 1.000000  | 1.000000 | 1.000000 |
| tie           | 1.000000  | 1.000000 | 1.000000 |
| sports ball   | 1.000000  | 1.000000 | 1.000000 |
| kite          | 1.000000  | 1.000000 | 1.000000 |
| tennis racket | 1.000000  | 1.000000 | 1.000000 |
| banana        | 1.000000  | 1.000000 | 1.000000 |
| chair         | 1.000000  | 1.000000 | 1.000000 |
| potted plant  | 1.000000  | 1.000000 | 1.000000 |
| tv            | 1.000000  | 1.000000 | 1.000000 |
| clock         | 1.000000  | 1.000000 | 1.000000 |

### 2. 低光照条件下的识别结果

| Class         | Precision | Recall  | F1-Score |
|---------------|-----------|---------|----------|
| person        | 0.988451  | 0.991093 | 0.989770 |
| bicycle       | 0.972727  | 0.990741 | 0.981651 |
| car           | 0.993814  | 0.998428 | 0.996116 |
| motorcycle    | 0.980288  | 0.987124 | 0.983694 |
| airplane      | 1.000000  | 1.000000 | 1.000000 |
| bus           | 0.995050  | 0.993820 | 0.994434 |
| train         | 0.960784  | 0.942308 | 0.951456 |
| truck         | 0.994872  | 0.995893 | 0.995382 |
| boat          | 1.000000  | 1.000000 | 1.000000 |
| traffic light | 0.990751  | 0.990751 | 0.990751 |
| fire hydrant  | 1.000000  | 1.000000 | 1.000000 |
| stop sign     | 0.958333  | 0.958333 | 0.958333 |
| bench         | 1.000000  | 1.000000 | 1.000000 |
| backpack      | 0.952381  | 0.952381 | 0.952381 |
| umbrella      | 1.000000  | 1.000000 | 1.000000 |
| handbag       | 1.000000  | 1.000000 | 1.000000 |
| tie           | 1.000000  | 1.000000 | 1.000000 |
| sports ball   | 1.000000  | 1.000000 | 1.000000 |
| kite          | 1.000000  | 1.000000 | 1.000000 |
| skateboard    | 1.000000  | 1.000000 | 1.000000 |
| tennis racket | 1.000000  | 1.000000 | 1.000000 |
| banana        | 1.000000  | 1.000000 | 1.000000 |
| chair         | 1.000000  | 1.000000 | 1.000000 |
| potted plant  | 0.966667  | 0.966667 | 0.966667 |
| tv            | 1.000000  | 1.000000 | 1.000000 |
| laptop        | 1.000000  | 1.000000 | 1.000000 |
| clock         | 1.000000  | 1.000000 | 1.000000 |


### 3. 不同扰动方式下的识别结果

| Disturbance mode     | acc   |
|:---------------------|:------|
| no-disturbance (std) | 0.987 |
| blur_3_3             | 0.489 |
| blur_5_5             | 0.432 |
| blur_7_7             | 0.376 |
| blur_9_9             | 0.371 |
| blur_11_11           | 0.355 |
| blur_15_15           | 0.343 |
| brighter             | 0.501 |
| darker               | 0.283 |
| motion_blur          | 0.373 |
| occlusion_circle     | 0.642 |
| occlusion_rectangle  | 0.600 |
| jpeg_compresses      | 0.389 |
| noise                | 0.770 |

---

## 分析

### 1. 精确度（Precision）

- 在正常光照条件下，大多数类别的Precision都为1，表明模型在这些类别的预测结果中几乎没有误报。
- 在低光照条件下，Precision略有下降，尤其是对于bicycle、train、stop sign、backpack和potted plant等类别。这表明，低光照环境下模型对部分类别的误报有所增加。

### 2. 召回率（Recall）

- 在正常光照条件下，召回率普遍很高，接近1，表明模型在大多数类别中能够识别出几乎所有的真实物体。
- 在低光照条件下，召回率同样保持在较高水平，但某些类别如train和stop sign的召回率有所下降，尤其是train类别，降幅较为明显。

### 3. F1-Score

- F1-Score在正常光照下普遍为1，表明模型在这些类别中表现完美，Precision与Recall的平衡非常好。
- 在低光照条件下，虽然大部分类别的F1-Score仍然保持较高值，但bicycle、train和stop sign等类别的F1-Score有所下降，反映了低光照对模型性能的影响。

---

## 结论

从测试结果可以看出，YOLO模型在正常光照条件下表现优秀，几乎所有类别的精度、召回率和F1-Score都接近1。而在低光照条件下，模型的性能略有下降，尤其是在部分类别（如bicycle、train和stop sign）的识别上，精度和召回率有所减少。

这表明，尽管YOLO模型在低光照条件下仍然具有较强的鲁棒性，但在某些极端条件下，尤其是在较难识别的物体类别上，模型的表现受到一定程度的影响。因此，在低光照环境下进一步优化模型，可能是提高整体识别准确性的一个方向。

---

## 建议

1. **数据增强**：为提高YOLO模型在低光照环境下的表现，可以考虑在训练阶段引入低光照图像进行数据增强。
2. **模型优化**：可以尝试调整模型的参数，或使用其他光照不敏感的特征来增强低光照条件下的识别能力。
3. **后处理优化**：在低光照条件下，可以通过图像预处理（如图像增强、噪声去除等）来改善模型的输入数据，从而提高识别精度。


