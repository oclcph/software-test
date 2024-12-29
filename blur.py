import cv2
import os
import numpy as np
import random

# 文件夹路径
input_folder = 'D:/AI/data/LoLI-Street/LoLI-Street Dataset/Test'
output_base_folder = 'output'  # 存储所有处理图像的根文件夹

# 定义不同的模糊程度（不同核大小）
blur_strengths = [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11), (15, 15)]  # 核大小（宽度, 高度）

# 创建根文件夹
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

# 亮度和对比度调整函数
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness  # 增加亮度的值
    alpha = contrast / 100.0 + 1.0  # 对比度调整值，默认为1.0
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

# 添加噪声的函数
def add_noise(image, mean=0, sigma=25):
    """给图像添加噪声"""
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, gaussian_noise)
    return noisy_image

# 运动模糊的函数
def motion_blur(image, kernel_size=15):
    """添加运动模糊"""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    motion_blurred = cv2.filter2D(image, -1, kernel)
    return motion_blurred

# 调整分辨率的函数
def resize_image(image, size=(416, 416)):
    """调整图像分辨率"""
    resized_image = cv2.resize(image, size)
    return resized_image

# 遮挡图像的函数
def add_occlusion(image, occlusion_type="rectangle", size=(100, 100), color=(0, 0, 0)):
    """给图像添加遮挡物"""
    h, w, _ = image.shape
    if occlusion_type == "rectangle":
        # 随机选择一个位置，填充为矩形遮挡
        top_left = (random.randint(0, w - size[0]), random.randint(0, h - size[1]))
        bottom_right = (top_left[0] + size[0], top_left[1] + size[1])
        cv2.rectangle(image, top_left, bottom_right, color, -1)
    elif occlusion_type == "circle":
        # 随机选择圆形遮挡
        center = (random.randint(0, w), random.randint(0, h))
        radius = random.randint(30, 50)
        cv2.circle(image, center, radius, color, -1)
    return image

if __name__ == '__main__':
    # 读取文件夹中的所有图像
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 检查文件扩展名
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # 调亮
            brightness_contrast_folder = os.path.join(output_base_folder, 'brighter')
            if not os.path.exists(brightness_contrast_folder):
                os.makedirs(brightness_contrast_folder)
            adjusted_image = adjust_brightness_contrast(image, brightness=50, contrast=30)
            save_path = os.path.join(brightness_contrast_folder, filename)
            cv2.imwrite(save_path, adjusted_image)

            # 调暗
            brightness_contrast_folder = os.path.join(output_base_folder, 'blacker')
            if not os.path.exists(brightness_contrast_folder):
                os.makedirs(brightness_contrast_folder)
            adjusted_image = adjust_brightness_contrast(image, brightness=-50, contrast=0)
            save_path = os.path.join(brightness_contrast_folder, filename)
            cv2.imwrite(save_path, adjusted_image)

            # 应用不同程度的高斯模糊
            for blur_strength in blur_strengths:
                blur_folder = os.path.join(output_base_folder, f"blur_{blur_strength[0]}_{blur_strength[1]}")
                if not os.path.exists(blur_folder):
                    os.makedirs(blur_folder)
                blurred_image = cv2.GaussianBlur(image, blur_strength, 0)
                save_path = os.path.join(blur_folder, filename)
                cv2.imwrite(save_path, blurred_image)

            # 添加噪声
            noise_folder = os.path.join(output_base_folder, 'noise')
            if not os.path.exists(noise_folder):
                os.makedirs(noise_folder)
            noisy_image = add_noise(image)
            save_path = os.path.join(noise_folder, filename)
            cv2.imwrite(save_path, noisy_image)

            # 添加运动模糊
            motion_blur_folder = os.path.join(output_base_folder, 'motion_blur')
            if not os.path.exists(motion_blur_folder):
                os.makedirs(motion_blur_folder)
            motion_blurred_image = motion_blur(image, 5)
            save_path = os.path.join(motion_blur_folder, filename)
            cv2.imwrite(save_path, motion_blurred_image)

            # 改变分辨率
            resize_folder = os.path.join(output_base_folder, 'resize')
            if not os.path.exists(resize_folder):
                os.makedirs(resize_folder)
            resized_image = resize_image(image, size=(224, 224))  # 改为较低的分辨率
            save_path = os.path.join(resize_folder, filename)
            cv2.imwrite(save_path, resized_image)

            # 添加遮挡（矩形遮挡）
            occlusion_folder = os.path.join(output_base_folder, 'occlusion_rectangle')
            if not os.path.exists(occlusion_folder):
                os.makedirs(occlusion_folder)
            image_with_occlusion = add_occlusion(image.copy(), occlusion_type="rectangle", size=(100, 100),
                                                 color=(0, 0, 0))
            save_path = os.path.join(occlusion_folder, filename)
            cv2.imwrite(save_path, image_with_occlusion)

            # 添加圆形遮挡
            occlusion_circle_folder = os.path.join(output_base_folder, 'occlusion_circle')
            if not os.path.exists(occlusion_circle_folder):
                os.makedirs(occlusion_circle_folder)
            image_with_circle_occlusion = add_occlusion(image.copy(), occlusion_type="circle", size=(0, 0),
                                                        color=(0, 0, 0))
            save_path = os.path.join(occlusion_circle_folder, filename)
            cv2.imwrite(save_path, image_with_circle_occlusion)

            # 压缩处理
            jpeg_compressed_folder = os.path.join(output_base_folder, 'jpeg_compressed')
            if not os.path.exists(jpeg_compressed_folder):
                os.makedirs(jpeg_compressed_folder)
            compression_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
            # 保存压缩后的图像
            save_path = os.path.join(jpeg_compressed_folder, filename)
            cv2.imwrite(save_path, image, compression_params)

            print(f"Processed {filename}")

    print("All images processed and saved to respective folders.")


