from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from vocoder.env import AttrDict
from vocoder.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from vocoder.models import Generator
# from env import AttrDict
# from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
# from models import Generator

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def get_mel_AGAIN(x, module):
    return mel_spectrogram(x, module.config['n_fft'], module.config['n_mels'], module.config['sample_rate'], module.config['hop_length'], module.config['win_length'], module.config['f_min'], module.config['f_max'])


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = librosa.resample(wav.astype(float), sr, 22050) ###########################################################
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)




import numpy as np
import librosa

def wav2mel(y):
        D = np.abs(librosa.stft(y, n_fft=h.n_fft,
            hop_length=h.hop_size, win_length=h.win_size)**2)
        D = np.sqrt(D)
        mel_basis = librosa.filters.mel(h.sampling_rate, h.n_fft,
                fmin=h.fmin, fmax=h.fmax,
                n_mels=h.num_mels)
        S = np.dot(mel_basis, D)
        log_S = np.log10(S)
        return log_S

def main():

    # print('Initializing HiFi-GAN Inference Process..')
    # config_file = os.path.join(os.path.split("vocoder/VCTK_V1/generator_v1")[0], 'config.json')
    # with open(config_file) as f:
    #     data = f.read()

    # global h
    # json_config = json.loads(data)
    # h = AttrDict(json_config)

    # torch.manual_seed(h.seed)
    # global device
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(h.seed)
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')

    # generator = Generator(h).to(device)

    # state_dict_g = load_checkpoint("vocoder/VCTK_V1/generator_v1", device)
    # generator.load_state_dict(state_dict_g['generator'])

    # # filelist = os.listdir(a.input_mels_dir)

    # os.makedirs("data/generated/wav_hifi_gan/", exist_ok=True)

    # generator.eval()
    # generator.remove_weight_norm()
    # with torch.no_grad():

    #     wav, sr = load_wav('data/wav48/p225/p225_004.wav')
    #     wav = librosa.resample(wav.astype(float), sr, 22050)
    #     # print(f"sample rate: {sr}")
    #     wav_hifi = wav / MAX_WAV_VALUE
    #     wav_hifi = torch.FloatTensor(wav_hifi).to(device)
    #     spec_hifi = get_mel(wav_hifi.unsqueeze(0))

    #     # print(f"type:{type(wav)}")
    #     spec_mel = wav2mel(wav)
    #     spec_mel = torch.FloatTensor(spec_mel).to(device).unsqueeze(0)

    #     print(f"hifi spec: {spec_hifi.shape}")
    #     print(f"mel spec: {spec_mel.shape}")

    #     y_g_hat_hifi = generator(spec_hifi)
    #     audio_hifi = y_g_hat_hifi.squeeze()
    #     audio_hifi = audio_hifi * MAX_WAV_VALUE
    #     audio_hifi = audio_hifi.cpu().numpy().astype('int16')

    #     y_g_hat_mel = generator(spec_mel)
    #     audio_mel = y_g_hat_mel.squeeze()
    #     audio_mel = audio_mel * MAX_WAV_VALUE
    #     audio_mel = audio_mel.cpu().numpy().astype('int16')

    #     output_file = os.path.join("data/generated/wav_hifi_gan/",  'test_hifi' + '.wav')
    #     write(output_file, h.sampling_rate, audio_hifi)
    #     output_file = os.path.join("data/generated/wav_hifi_gan/",  'test_mel' + '.wav')
    #     write(output_file, h.sampling_rate, audio_mel)

    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

