import sys
from collections import defaultdict
import torch
import numpy as np
from scipy.special import softmax
import re
import os

numbers = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
'six' : 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}


def compute_scores(rst_dict, ref_dict, labels):

    correct = 0
    count = 0

    preds = defaultdict(int)
    corrs = defaultdict(int)

    for audio_id in rst_dict:
        target = ref_dict[audio_id]
        if target not in labels:
            assert 0
        results = rst_dict[audio_id]
        sorted_results = sorted(results, key=lambda x:x[0])
        hyp = sorted_results[0][1].lower()
        count = count + 1
        if re.search(r'\b' + re.escape(target) + r'\b', hyp.strip().lower()):
            correct = correct + 1

    sorted_preds = sorted(preds.items(), key=lambda x:-x[1])
    for ii, (name, cnt) in enumerate(sorted_preds):
        p = corrs[name]/cnt
        r = corrs[name]/(len(ref_dict) / len(labels))
        if p or r:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0
        if len(sys.argv) > 2:
            print(f'{ii+1} & {name} & {cnt} & {corrs[name]} & {round(p, 2)} & {round(r, 2)} & {round(f1, 2)} \\\\')

    return correct / count


scores = []
dataset = sys.argv[1].split('/')[2]

############################# Uncalibrated ## Score #####################################

with open(sys.argv[1]) as fhyp, open(f'data/{dataset}/audio_ref_pair_list') as fref, open(f'data/{dataset}/labels') as flabel:
    ref_dict = dict()
    for line in fref:
        audio_id, wav_path, text = line.strip().split(None, 2)
        ref_dict[audio_id] = text

    labels = dict()
    for ii, line in enumerate(flabel):
        labels[line.strip()] = ii
    print('Classes:', len(labels))

    rst_dict = defaultdict(list)
    for line in fhyp:
        audio_id, wav_path, score, text = line.strip().split(None, 3)
        rst_dict[audio_id].append([float(score), text])
        scores.append(float(score))

print(f'Uncalibrated score: {compute_scores(rst_dict, ref_dict, labels)}')


############################# Prior-matched ## Score ###########################################
norm_scores_pt = torch.FloatTensor(scores).view(-1, len(labels))

nll_tmp = norm_scores_pt.clone().cpu().numpy()
nll = norm_scores_pt.clone().cpu().numpy()

N = nll.shape[-1]
weights = np.zeros((1, N))
uniform = np.ones((N))/N

step = 1
lr = 0.1
prior = np.mean(softmax(-1*nll, axis=-1), axis=0)

while np.sum(np.abs(prior - uniform)) > 0.001:
    new_nll = nll + weights
    prior = np.mean(softmax(-1*new_nll, axis=-1), axis=0)
    weights -= lr*(prior<uniform)

    if step % 1000 == 0:
        lr /= 2

    step += 1
weights -= np.min(weights)
new_nll = nll + weights

rst_dict = defaultdict(list)
with open(sys.argv[1]) as fhyp, open(sys.argv[1] + '_norm', 'w') as fout:
    for norm_score, line in zip(new_nll.flatten(), fhyp):
        audio_id, wav_path, score, text = line.strip().split(None, 3)
        fout.write(f'{audio_id} {wav_path} {norm_score} {text}\n')
        rst_dict[audio_id].append([norm_score, text])

print(f'Prior-matched score: {compute_scores(rst_dict, ref_dict, labels)}')


################################## Zero-input ## Score ######################################

ilm_dict = dict()
if os.path.exists(sys.argv[1] + '_ilm_zeros'):
    with open(sys.argv[1] + '_ilm_zeros') as film:
        for line in film:
            audio_id, wav_path, score, text = line.strip().split(None, 3)
            ilm_dict[text] = float(score)

    rst_dict = defaultdict(list)
    with open(sys.argv[1]) as fhyp:
        for line in fhyp:
            audio_id, wav_path, score, text = line.strip().split(None, 3)
            rst_dict[audio_id].append([float(score)-ilm_dict[text], text])
    print(f'Zero-input score: {compute_scores(rst_dict, ref_dict, labels)}')


################################### Gaussian-noise ## Score #####################################
std = 1.0
ilm_dict = dict()
base_dir = os.path.split(sys.argv[1])[0]
if os.path.exists(sys.argv[1] + '_ilm_gaussian_' + str(std)):
    with open(sys.argv[1] + '_ilm_gaussian_' + str(std)) as film:
        for line in film:
            audio_id, wav_path, score, text = line.strip().split(None, 3)
            ilm_dict[text] = float(score)

    rst_dict = defaultdict(list)
    with open(sys.argv[1]) as fhyp:
        for line in fhyp:
            audio_id, wav_path, score, text = line.strip().split(None, 3)
            rst_dict[audio_id].append([float(score)-ilm_dict[text], text])
    print(f'Gassian-noise score: {compute_scores(rst_dict, ref_dict, labels)}')
