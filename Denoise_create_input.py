import cv2
import numpy as np

def add_noise_to_video(input_video, output_video, noise_level=25, resize_dim=(1920, 1280)):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS), resize_dim)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 调整尺寸并添加噪声
        frame = cv2.resize(frame, resize_dim)
        
        # 将帧转换为浮点数以避免溢出问题
        frame = frame.astype(np.float32)
        
        # 生成噪音并添加到帧中
        noise = np.random.normal(0, noise_level, frame.shape).astype(np.float32)
        noisy_frame = frame + noise
        
        # 将像素值剪裁到 [0, 255] 并转换回 uint8 类型
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        
        # 将处理后的帧写入输出视频
        out.write(noisy_frame)

    cap.release()
    out.release()
    print(f"Processed and saved noisy video: {output_video}")

# 使用适度的噪音级别
input_video = r"C:\Users\user\Desktop\mp4\bus.mp4"
output_video = r"C:\Users\user\Desktop\bus_orig.avi"
add_noise_to_video(input_video, output_video, noise_level=0)  # 噪音水平设为10
print("Creation Done!")