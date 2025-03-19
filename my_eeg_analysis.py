import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# 1. 设置文件夹路径
folder_path = r"D:\文档：）\学习文件\eeg\files"  # 你的 EEG 数据文件夹路径

# 2. 遍历文件夹，读取所有 .edf 文件
edf_files = []
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".edf"):
            edf_files.append(os.path.join(root, file))

# 3. 读取每个 .edf 文件并打印基本信息
for edf_file in edf_files:
    print(f"正在读取: {edf_file}")
    raw = mne.io.read_raw_edf(edf_file, preload=True)  # 读取文件
    print(raw.info)  # 打印基本信息
    print(raw.ch_names)  # 打印通道名称

# 4. 计算 Power Spectral Density (PSD)
psd_info = raw.compute_psd(fmin=0, fmax=40, n_fft=2048)  # 限制频率范围到 0-40 Hz
psd = psd_info.get_data()  # 获取 PSD 数据
freqs = psd_info.freqs  # 获取频率信息

# 打印一下 PSD 和 freqs 的形状，确保它们正确
print("PSD shape:", psd.shape)
print("Frequencies:", freqs[:10])  # 打印前10个频率

# 5. 提取 α 波和 β 波频段并计算每个通道的均值
alpha_band = (freqs >= 8) & (freqs <= 13)  # α 波频段：8-13 Hz
beta_band = (freqs >= 14) & (freqs <= 30)  # β 波频段：14-30 Hz

alpha_psd = psd[:, alpha_band].mean(axis=1)  # 对每个通道在 α 波范围内求平均
beta_psd = psd[:, beta_band].mean(axis=1)  # 对每个通道在 β 波范围内求平均

# 打印 α 波和 β 波的前五个通道的均值
print("Alpha PSD (mean):", alpha_psd[:5])
print("Beta PSD (mean):", beta_psd[:5])

# 6. 使用对数尺度绘制 PSD 图（每个通道的 PSD）
plt.figure(figsize=(10, 6))
for i in range(5):  # 只画前5个通道的 PSD（可以根据需要调整）
    plt.plot(freqs, np.log10(psd[i, :]), label=f'Channel {i+1}', alpha=0.7)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Log Power Spectral Density (dB)")
plt.title("Log Power Spectral Density of Selected Channels (0-40 Hz)")
plt.legend(loc="upper right", fontsize=8)
plt.show()

# 7. 绘制 α 波和 β 波的均值对比图
plt.figure(figsize=(10, 6))
plt.plot(alpha_psd, label="Alpha PSD", marker="o", linestyle="--", color="b")
plt.plot(beta_psd, label="Beta PSD", marker="s", linestyle=":", color="r")
plt.xlabel("Channel Index")
plt.ylabel("Mean Power Spectral Density")
plt.title("Alpha and Beta Band Power Across Channels")
plt.legend()
plt.show()

# 8. 绘制频谱热图（频率与通道的关系）
plt.figure(figsize=(10, 6))
plt.imshow(np.log10(psd[:10, :]), aspect='auto', origin='lower', cmap='viridis', extent=[freqs[0], freqs[-1], 0, 10])  # 只显示前10个通道的热图
plt.colorbar(label="Log Power Spectral Density (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Channel Index")
plt.title("Power Spectral Density Heatmap (Log Scale)")
plt.show()
