from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from vocoder.env import AttrDict
from vocoder.meldataset import MAX_WAV_VALUE
from vocoder.models import Generator


h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


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

    filelist = os.listdir(a.input_mels_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            x = np.load(os.path.join(a.input_mels_dir, filname))
            x = torch.FloatTensor(x).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)



def hifi_gan_mel2wav(mel, device, output_name):
    print('Initializing HiFi-GAN Inference Process..')
    config_file = os.path.join(os.path.split("vocoder/VCTK_V1/generator_v1")[0], 'config.json')
    # config_file = os.path.join(os.path.split("VCTK_V1/generator_v1")[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    # global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint("vocoder/VCTK_V1/generator_v1", device)
    # state_dict_g = load_checkpoint("VCTK_V1/generator_v1", device)
    generator.load_state_dict(state_dict_g['generator'])

    # filelist = os.listdir(a.input_mels_dir)

    os.makedirs("data/generated/wav_hifi_gan/", exist_ok=True)
    # os.makedirs("../data/generated/wav_hifi_gan/", exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()


    with torch.no_grad():
        # for i, filname in enumerate(filelist):
        # x = np.load(os.path.join(a.input_mels_dir, filname))
        # x = torch.FloatTensor(x).to(device)
        y_g_hat = generator(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        output_file = os.path.join("data/generated/wav_hifi_gan/",  output_name + '.wav')
        # output_file = os.path.join("../data/generated/wav_hifi_gan/",  output_name + '.wav')
        write(output_file, h.sampling_rate, audio)
        print(output_file)



# import librosa
# from meldataset import mel_spectrogram
# from env import AttrDict
# from meldataset import MAX_WAV_VALUE, load_wav
# from models import Generator
def main():
    print('Initializing Inference Process..')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_mels_dir', default='test_mel_files')
    # parser.add_argument('--output_dir', default='generated_files_from_mel')
    # parser.add_argument('--checkpoint_file', required=True)
    # a = parser.parse_args()

    # # config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
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

    # inference(a)

    path = "../data/wav48/p225/p225_003.wav"
    # y, sr = librosa.load(path, sr=22050)
    y, sr = load_wav(path)
    y = librosa.resample(y.astype(float), sr, 22050)
    if type(20) is int:
        y, _ = librosa.effects.trim(y, top_db=20)
    # y = np.clip(y, -1.0, 1.0)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    MAX_WAV_VALUE = 32768.0
    y = y / MAX_WAV_VALUE
    y = torch.FloatTensor(y).to(device)

    mel = mel_spectrogram(y.unsqueeze(0), 1024, 80,22050,256,1024,0,11025)

    hifi_gan_mel2wav(mel, device, 'test2')



if __name__ == '__main__':
    main()

