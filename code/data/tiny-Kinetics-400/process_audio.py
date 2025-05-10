"""
import csv
import os

import numpy as np
import torchaudio
import torch

## save path of processed spectrogram
save_path = './tiny-Kinetics-400-audio/train_spec'

## file path of wav files
audio_path='./tiny-Kinetics-400-audio/test_set'

## the list of all wav files
csv_file = './tiny-Kinetics-400-audio/tinyk_train_real.txt'


data = []
with open(csv_file) as f:
  for line in f:
      item = line.split("\n")[0].split(" ")
      name = item[0][:-4]

      if os.path.exists(audio_path + '/' + name + '.wav'):
        data.append(name)
        # print(name)
        # exit(0)

for name in data:
  waveform, sr = torchaudio.load(audio_path + '/'+ name + '.wav')
  waveform = waveform - waveform.mean()
  norm_mean = -4.503877
  norm_std = 5.141276

  fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
  
  target_length = 1024
  n_frames = fbank.shape[0]
  # print(n_frames)
  p = target_length - n_frames

  # cut and pad
  if p > 0:
      m = torch.nn.ZeroPad2d((0, 0, 0, p))
      fbank = m(fbank)
  elif p < 0:
      fbank = fbank[0:target_length, :]
  fbank = (fbank - norm_mean) / (norm_std * 2)

  print(fbank.shape)
  np.save(save_path + '/'+ name + '.npy',fbank)
"""

import os
import csv
import numpy as np
import librosa

# 设置路径
# save_path = 'G:\\research\paper_repeat\\Valuate-and-Enhance-Multimodal-Cooperation\\code\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-audio\\train_spec'
# audio_path = 'G:\\research\paper_repeat\\Valuate-and-Enhance-Multimodal-Cooperation\\code\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-audio\\train_set'
# csv_file = 'G:\\research\paper_repeat\\Valuate-and-Enhance-Multimodal-Cooperation\\code\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-audio\\tinyk_train_real.txt'
save_path = 'G:\\research\paper_repeat\\Valuate-and-Enhance-Multimodal-Cooperation\\code\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-audio\\test_spec'
audio_path = 'G:\\research\paper_repeat\\Valuate-and-Enhance-Multimodal-Cooperation\\code\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-audio\\test_set'
csv_file = 'G:\\research\paper_repeat\\Valuate-and-Enhance-Multimodal-Cooperation\\code\\data\\tiny-Kinetics-400\\tiny-Kinetics-400-audio\\tinyk_test_real.txt'

# 创建保存目录
os.makedirs(save_path, exist_ok=True)

# 读取文件列表
data = []
with open(csv_file, 'r') as f:
    for name in f:
        data.append(name.strip())
print(f"Total valid files to process: {len(data)}")

# 处理每个音频文件
for name in data:
    wav_path = os.path.join(audio_path, name)

    # 加载音频文件
    waveform, sr = librosa.load(wav_path, sr=None, mono=True)

    # 计算 Mel Spectrogram（参数与原脚本对齐）
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=400,
        hop_length=160,
        win_length=400,
        window='hanning',
        n_mels=128,
        power=1.0,
        htk=True
    )

    log_mel_spec = np.log(mel_spec + 1e-6)  # 取 log 避免数值问题

    # 转置成 [T, F] 格式（时间帧 x 特征维度）
    log_mel_spec = log_mel_spec.T

    # Padding 或截断到目标长度
    target_length = 1024
    n_frames = log_mel_spec.shape[0]
    p = target_length - n_frames

    if p > 0:
        pad_data = np.tile(log_mel_spec[-1:, :], (p, 1))  # 用最后一行填充
        log_mel_spec = np.vstack((log_mel_spec, pad_data))
    elif p < 0:
        log_mel_spec = log_mel_spec[:target_length, :]

    # 归一化处理（使用你原来的均值和标准差）
    norm_mean = -4.503877
    norm_std = 5.141276
    log_mel_spec = (log_mel_spec - norm_mean) / (norm_std * 2)

    print(f"Processed {name}, shape: {log_mel_spec.shape}")

    # 保存为 .npy 文件
    save_file = os.path.join(save_path, name + '.npy')
    np.save(save_file, log_mel_spec)
