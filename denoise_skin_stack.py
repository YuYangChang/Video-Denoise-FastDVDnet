import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

# 匯入自己的影像
image_path = r'C:\Users\user\Desktop\SB_denoise_test.png' 
astro = img_as_float(io.imread(image_path))

# =============================================================================
# # 加入隨機噪聲 (可選, 若影像已經是含噪影像可跳過這一步)
# sigma = 0.08
# noisy = random_noise(astro, var=sigma**2)
# =============================================================================

# 估計影像的噪聲標準差
sigma_est = np.mean(estimate_sigma(astro, channel_axis=-1))
print(f'estimated noise standard deviation = {sigma_est}')

# 設定補丁參數
patch_kw = dict(
    patch_size=5, patch_distance=6, channel_axis=-1  # 5x5 patches
)

# 使用非局部均值去噪演算法
denoise = denoise_nl_means(astro, h=1.15 * sigma_est, fast_mode=False, **patch_kw)

# 使用快速非局部均值去噪演算法
denoise_fast = denoise_nl_means(astro, h=0.8 * sigma_est, fast_mode=True, **patch_kw)

# 顯示原始影像、含噪影像與去噪後影像
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharex=True, sharey=True)

ax[0].imshow(astro)
ax[0].axis('off')
ax[0].set_title('Original\n(no noise)')

ax[0].imshow(denoise)
ax[0].axis('off')
ax[0].set_title('Denoised\n(NLM)')
# =============================================================================
# ax[1].imshow(noisy)
# ax[1].axis('off')
# ax[1].set_title('Noisy')
# 
# =============================================================================
ax[2].imshow(denoise_fast)
ax[2].axis('off')
ax[2].set_title('Denoised\n(fast NLM)')

fig.tight_layout()

# 計算各種影像的 PSNR 值
psnr_noisy = peak_signal_noise_ratio(astro, denoise)
psnr_denoise_fast = peak_signal_noise_ratio(astro, denoise_fast)

print(f'PSNR (noisy) = {psnr_noisy:0.2f}')
print(f'PSNR (fast denoised) = {psnr_denoise_fast:0.2f}')

plt.show()
