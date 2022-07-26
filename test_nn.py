from pathlib import Path
import os

import soundfile as sf
import torch
from torch.utils.data import DataLoader
# from torchaudio.functional import angle
from torch import angle
from torch import istft

from dataloader.wav_dataset import WAVDataset
from model.nsnet_model import NSNetModel
import sys

path_to_weights = sys.arv[1]

model = NSNetModel.load_from_checkpoint(Path(path_to_weights))
# train_dir = './WAVs/dataset/training'
# val_dir = './WAVs/dataset/validation'

# hparams = {'train_dir': train_dir,
#            'val_dir': val_dir,
#            'batch_size': 64,
#            'n_fft': 512,
#            'n_gru_layers': 3,
#            'gru_dropout': 0.2,
#            'alpha': 0.35}

# model = NSNetModel()
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

testing_dir = Path('/datasets/wav/val')
n_fft = 512
dataset = WAVDataset(dir=testing_dir, n_fft=n_fft, test=True)
dataloader = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=True)
noisy_waveform, clean_waveform, x_stft, _, x_lps, x_ms, _, _ = next(iter(dataloader))

#  enable eval mode
model.zero_grad()
model.eval()
model.freeze()

# disable gradients to save memory
torch.set_grad_enabled(False)

print('model forward')
gain_mask = model(x_lps)
y_spectrogram_hat = x_ms * gain_mask

# print('\n', y_spectrogram_hat.shape, x_stft.shape, torch.angle(torch.view_as_complex(x_stft)).shape, '\n')
print('vies as complex')
x_stft = torch.view_as_complex(x_stft)
print('stack')
y_stft_hat = torch.stack([y_spectrogram_hat * torch.cos(angle(x_stft)),
                          y_spectrogram_hat * torch.sin(angle(x_stft))], dim=-1)
print('hamming window')
window = torch.hamming_window(n_fft)
print('istft')
y_waveform_hat = istft(y_stft_hat, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft, window=window, length=clean_waveform.shape[-1])
print('writing')
for i, waveform in enumerate(y_waveform_hat.numpy()):
    sf.write('denoised/denoised' + str(i) + '.wav', waveform, 16000)
