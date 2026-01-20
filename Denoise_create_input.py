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
    
        frame = cv2.resize(frame, resize_dim)
    
        frame = frame.astype(np.float32)
        
        noise = np.random.normal(0, noise_level, frame.shape).astype(np.float32)
        noisy_frame = frame + noise
        
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        
        out.write(noisy_frame)

    cap.release()
    out.release()
    print(f"Processed and saved noisy video: {output_video}")

input_video = r"C:\Users\user\Desktop\mp4\bus.mp4"
output_video = r"C:\Users\user\Desktop\bus_orig.avi"
add_noise_to_video(input_video, output_video, noise_level=0)  

print("Creation Done!")
