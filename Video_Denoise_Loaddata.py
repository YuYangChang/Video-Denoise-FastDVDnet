import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

# Video datasets type definition
class VideoDataset(Dataset):
    def __init__(self, clean_video_path, noisy_video_path, resize_dim=(512, 512), transform=None):
        self.clean_frames = self._extract_frames(clean_video_path, resize_dim)
        self.noisy_frames = self._extract_frames(noisy_video_path, resize_dim)
        self.transform = transform

    def _extract_frames(self, video_path, resize_dim):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize_dim)
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        return min(len(self.clean_frames), len(self.noisy_frames))

    def __getitem__(self, idx):
        clean_frame = self.clean_frames[idx]
        noisy_frame = self.noisy_frames[idx]

        if self.transform:
            clean_frame = self.transform(clean_frame)
            noisy_frame = self.transform(noisy_frame)

        return noisy_frame, clean_frame

# Input data to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load input data
clean_video_path = r"C:\Users\user\Desktop\U Yang 統整\ISP專題\Video Denoise\original_512512.avi"
noisy_video_path = r"C:\Users\user\Desktop\U Yang 統整\ISP專題\Video Denoise\noisy_512512.avi"
save_dir = r"C:\Users\user\Desktop\20241017_denoising_results"
os.makedirs(save_dir, exist_ok=True)

# Initialization for training sets (reshape input data)
dataset = VideoDataset(clean_video_path, noisy_video_path, resize_dim=(512, 512), transform=transform)

# Training: Validation = 4:1
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print("Load data done!")

# Define MSE, PSNR Function
def calculate_mse_psnr(clean_frame, denoised_frame):
    mse = np.mean((clean_frame - denoised_frame) ** 2)
    if mse == 0:
        psnr = 100  # Set Upper Limit 
    else:
        pixel_max = 255.0  # Gray Scale Maximum
        psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    return mse, psnr

# Plot MSE, PSNR
def plot_metrics(frame_indices, mse_list, psnr_list, mse_list_denoise, psnr_list_denoise, output_path):
    plt.figure(figsize=(10, 8))
    
    # MSE
    plt.subplot(2, 1, 1)
    plt.plot(frame_indices, mse_list, label='Original MSE', color='blue', marker='o')
    plt.plot(frame_indices, mse_list_denoise, label='Denoised MSE', color='orange', marker='x')
    plt.xlabel('Frame Index')
    plt.ylabel('MSE')
    plt.title('MSE Comparison over Frames')
    plt.legend()
    
    # PSNR
    plt.subplot(2, 1, 2)
    plt.plot(frame_indices, psnr_list, label='Original PSNR', color='blue', marker='o')
    plt.plot(frame_indices, psnr_list_denoise, label='Denoised PSNR', color='orange', marker='x')
    plt.xlabel('Frame Index')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Comparison over Frames')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved MSE and PSNR plot to {output_path}")

# Save denoise video and calculate MSE, PSNR
def save_denoised_video_with_metrics(input_video_path, output_video_path, model, resize_dim=(512, 512), frame_interval=5):
    cap = cv2.VideoCapture(input_video_path)
    
    # Video for AVI format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI Format
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), resize_dim)

    mse_list = []
    psnr_list = []
    mse_list_denoise = []
    psnr_list_denoise = []
    frame_indices = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Training per frame_interval 
        if frame_idx % frame_interval == 0:
            frame_indices.append(frame_idx)

            # Turn frame into Tensor
            frame = cv2.resize(frame, resize_dim)
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

            # Denoise...
            with torch.no_grad():
                denoised_tensor = model(frame_tensor)

            # Tensor to Numpy format and spanned to [0, 255]
            denoised_frame = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            denoised_frame = np.clip(denoised_frame, 0, 255).astype(np.uint8)

            # Calculate PSNR, MSE for denoise frame and noisy frame
            mse, psnr = calculate_mse_psnr(frame, denoised_frame)
            mse_list_denoise.append(mse)
            psnr_list_denoise.append(psnr)

            # Calculate PSNR, MSE for initial input frame (If necessary)
            mse_orig, psnr_orig = calculate_mse_psnr(frame, frame)  
            mse_list.append(mse_orig)
            psnr_list.append(psnr_orig)

            # Save denoise frame
            out.write(denoised_frame)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved denoised video in AVI format: {output_video_path}")
    
    # Plot and save MSE, PSNR figure
    plot_metrics(frame_indices, mse_list, psnr_list, mse_list_denoise, psnr_list_denoise, os.path.join(save_dir, 'mse_psnr_testing.png'))

# Model implementation
class FastDVDnet(nn.Module):
    def __init__(self):
        super(FastDVDnet, self).__init__()
        self.denoise_block1 = self._denoise_block()
        self.denoise_block2 = self._denoise_block()

    def _denoise_block(self):
        layers = []
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(4):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        output_block1 = self.denoise_block1(x)
        output_block2 = self.denoise_block2(output_block1)
        return output_block2

# Training, Validation
def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0

        for i, (noisy_frame, clean_frame) in enumerate(dataloader_train):
            noisy_frame, clean_frame = noisy_frame.cuda(), clean_frame.cuda()

            optimizer.zero_grad()
            outputs = model(noisy_frame)
            loss = criterion(outputs, clean_frame)

            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()

        avg_train_loss = running_loss_train / len(dataloader_train)
        train_losses.append(avg_train_loss)

        # Validation...
        model.eval()
        running_loss_val = 0.0
        with torch.no_grad():
            for i, (noisy_frame, clean_frame) in enumerate(dataloader_val):
                noisy_frame, clean_frame = noisy_frame.cuda(), clean_frame.cuda()

                outputs = model(noisy_frame)
                loss = criterion(outputs, clean_frame)
                running_loss_val += loss.item()

        avg_val_loss = running_loss_val / len(dataloader_val)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Plot loss curve for training, validation
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # Save the model weights
    torch.save(model.state_dict(), os.path.join(save_dir, 'denoising_model_weights.pth'))
    print(f"Saved model weights to {os.path.join(save_dir, 'denoising_model_weights.pth')}")

# Main process
if __name__ == "__main__":
    # Device settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FastDVDnet().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)

    # Save denoised video with metrics
    output_video_path = os.path.join(save_dir, 'denoised_output.avi')
    save_denoised_video_with_metrics(noisy_video_path, output_video_path, model, resize_dim=(512, 512), frame_interval=1) 