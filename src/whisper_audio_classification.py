#!/usr/bin/python3
# Apply Whisper for zero-shot Audio Classification
# Rao Ma, 2023-12-06
import argparse
from general import check_output_dir
import numpy as np
import torch
from torch import nn
from pytorch_lightning import seed_everything
import torchaudio
import torchaudio.transforms as at
import os
import numpy
import whisper


def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wave_path, normalize=True)
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform


def load_audio_file_list(pair_list_path):
    audio_transcript_pair_list = []
    for line in open(pair_list_path):
        line_sp = line.strip().split(None, 2)
        if len(line_sp) == 2:
            line_sp.append('')
        assert len(line_sp) == 3
        audio_transcript_pair_list.append(line_sp)
    return audio_transcript_pair_list


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, audio=None, n_mels=80) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.audio = audio
        self.n_mels = n_mels

    def __len__(self):
        return len(self.audio_info_list)
    
    def __getitem__(self, id):
        _, audio_path, text = self.audio_info_list[id]
        if self.audio is not None:
            audio = self.audio
        else:
            audio = load_wave(audio_path, sample_rate=self.sample_rate)

        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, self.n_mels)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text,
        }


class WhisperDataCollatorWhithPadding:
    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50256) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id, TODO: multilingual models?

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch


def audio_classification(args):
    out_dir = f"{args.out_dir}/{args.model}/{args.eval_list_file.split('/')[1]}"
    check_output_dir(out_dir)
    out_fname = f"{out_dir}/{args.eval_list_file.split('/')[-1]}"
    if args.ilm:
        if args.ilm == 'gaussian':
            out_fname += '_ilm_gaussian_' + str(args.std)
        elif args.ilm == 'zero':
            out_fname += '_ilm_zeros'

    eval_datalist = load_audio_file_list(args.eval_list_file)
    wmodel = whisper.load_model(args.model, device=args.device, download_root=os.path.expanduser(args.model_dir))
    wtokenizer = whisper.tokenizer.get_tokenizer(not args.model.endswith('.en'), language=args.language, task='transcribe', num_languages=wmodel.num_languages)

    if args.ilm == 'gaussian':
        audio = torch.from_numpy(numpy.float32(numpy.random.normal(0, args.std, size=int(args.avg_len * args.sample_rate))))
    elif args.ilm == 'zero':
        audio = torch.zeros(int(args.avg_len * args.sample_rate))
    else:
        audio = None

    dataset = SpeechDataset(eval_datalist, wtokenizer, args.sample_rate, audio=audio, n_mels=wmodel.dims.n_mels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=WhisperDataCollatorWhithPadding())
    wmodel.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    with open(out_fname, 'w') as fout:
        idx = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].long().to(args.device)
            dec_input_ids = batch["dec_input_ids"].long().to(args.device)

            with torch.no_grad():
                audio_features = wmodel.encoder(input_ids)
                out = wmodel.decoder(dec_input_ids, audio_features)
                loss = loss_fn(out.reshape(-1, out.size(-1)), labels.view(-1)).view(dec_input_ids.size(0), -1)
                loss = loss.sum(dim=-1)

            for l, _ in zip(loss, labels):
                audio_id, audio_path, text = eval_datalist[idx]
                fout.write(f'{audio_id} {audio_path} {float(l)} {text}\n')
                idx = idx + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform zero-shot audio classification with Whisper')
    parser.add_argument('--model', type=str, default='large-v2', help='version of the pretrained Whisper model')
    parser.add_argument('--model_dir', type=str, default='~/.cache/whisper', help='path to load model')
    parser.add_argument('--seed', type=int, default=1031, help='random seed')
    parser.add_argument('--eval_list_file', type=str, default='', help='filepath containing all prompted text sequences to be evaluated')
    parser.add_argument('--out_dir', type=str, default='exp')
    parser.add_argument('--ilm', type=str, default='', help='used for null-input calibration, could be set to zero or gaussian')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--avg_len', type=float, default=0, help='used in null-input calibration. The average length of each dataset. ESC-50-master: 5; UrbanSound8K: 3.6; TUT2017_eval: 10; vocal: 5; ravdess: 3.7; CREMA-D: 5.0; GTZAN_genres: 30.0 LibriCount: 5.0')
    parser.add_argument('--std', type=float, default=1.0, help='the std value used in gaussian-noise null-input calibration')
    parser.add_argument('--language', type=str, default='en')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed_everything(args.seed, workers=True)
    audio_classification(args)
