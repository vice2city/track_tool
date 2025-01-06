import cv2

def crop_center(img, cropx, cropy):
    """
    裁剪图像中心部分。

    Args:
        img: 要裁剪的图像。
        cropx: 裁剪后的宽度。
        cropy: 裁剪后的高度。

    Returns:
        裁剪后的图像。
    """
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def concat_videos(video1_path, video2_path, output_path):
    """
    读取两段视频，拼接两个视频。

    Args:
      video1_path: 第一段视频的路径。
      video2_path: 第二段视频的路径。
      output_path: 输出视频的路径。
    """

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 裁剪后的尺寸
    crop_size = 600

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
    out = cv2.VideoWriter(output_path, fourcc, 30, (crop_size, crop_size))

    # 处理第一个视频
    while cap1.isOpened():
        ret, frame = cap1.read()
        if not ret:
            break

        # 裁剪第一个视频
        frame = crop_center(frame, crop_size, crop_size)

        out.write(frame)

    # 处理第二个视频
    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            break

        # 裁剪第二个视频
        frame = crop_center(frame, crop_size, crop_size)
        # 旋转第二个视频90度 (顺时针)
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        out.write(rotated_frame)

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()
    print(f"Video saved to {output_path}")

# 使用示例
concat_videos('/code/data/demo/demo1.mp4', '/code/data/demo/demo2.mp4', '/code/data/demo19.mp4')