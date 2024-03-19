# Whisper_audio_classification
-----------------------------------------
This is the official implementation of our paper "[Investigating the Emergent Audio Classification Ability of ASR Foundation Models](https://arxiv.org/abs/2311.09363)".

> Authors: Rao Ma*, Adian Liusie*, Mark Gales, Kate Knill\
> Abstract: Text and vision foundation models can perform many tasks in a zero-shot setting, a desirable property that enables these systems to be applied in general and low-resource settings. However, there has been significantly less work on the zero-shot abilities of ASR foundation models, with these systems typically fine-tuned to specific tasks or constrained to applications that match their training criterion and data annotation. In this work we investigate the ability of [Whisper](https://github.com/openai/whisper) and [MMS](https://ai.meta.com/blog/multilingual-model-speech-recognition), ASR foundation models trained primarily for speech recognition, to perform zero-shot audio classification. We use **simple template-based text prompts** at the decoder and use the resulting decoding probabilities to generate zero-shot predictions. Without training the model on extra data or adding any new parameters, we demonstrate that **Whisper shows promising zero-shot classification performance** on a range of **8 audio-classification datasets**, outperforming existing state-of-the-art zero-shot baseline's accuracy by an average of 9%. One important step to unlock the emergent ability is **debiasing**, where a simple unsupervised reweighting method of the class probabilities yields consistent significant performance gains. We further show that performance increases with model size, implying that as ASR foundation models scale up, they may exhibit improved zero-shot performance.


Overview
-----------------------------------------
1. `data`: The processed datafiles where each line contains audio_id, audio_path, and the filled prompt text associated with a class label. Model performance of using different prompt templates are compared in the original paper, as listed in Appendix D.
2. `src/whisper_audio_classification.py`: Compute the zero-shot ASR classification probability $P_\theta(t(w_k)|s)$ for each class $w_k$ given the audio $s$, as specified in Section 3.1.
3. `src/scoring.py`: Calculate the classification accuracy with different types of debiasing methods introduced in Section 3.2.

Requirements
-----------------------------------------
- python 3.10
- torch 1.12.1
- pytorch_lightning 2.0.2
- openai-whisper


Data Preparation
-----------------------------------------
In our paper we used 8 audio classification datasets and 1 audio question answering dataset.

Please download the original wav data from the test split using the following links and modify the audio paths for files under the `data/` directory accordingly.

- Sound Event Classification:
    - ESC-50: https://github.com/karolpiczak/ESC-50
    - UrbanSound8K: https://zenodo.org/records/1203745
- Acoustic Scene Classification:
    - TUT2017: https://zenodo.org/records/1040168
- Vocal Sound Classification:
    - Vocal sound: https://github.com/YuanGongND/vocalsound
- Emotion Recognition:
    - RAVDESS: https://zenodo.org/records/1188976
    - CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
- Music Genre Classification:
    - GTZAN: https://github.com/chittalpatel/Music-Genre-Classification-GTZAN
- Speaker Counting:
    - LibriCount: https://zenodo.org/records/1216072
- Audio Question Answering:
    - Clotho-AQA: https://zenodo.org/records/6473207


Instruction
-----------------------------------------
Here we use the ESC-50 dataset as an example to illustrate how to prompt the emergent audio classification ability of Whisper:

Run the zero-shot audio classification:

    python3 src/whisper_audio_classification.py --eval_list_file data/ESC50/prompt0

Run the zero-shot audio classification with zero input:

    python3 src/whisper_audio_classification.py \
        --eval_list_file data/ESC50/prompt0 \
        --ilm zero --avg_len 5.0

Run the zero-shot audio classification with Gaussian noise:

    python3 src/whisper_audio_classification.py \
        --eval_list_file data/ESC50/prompt0 \
        --ilm gaussian --avg_len 5.0

Calculate the classification accuracy with different calibration methods:

    python3 src/scoring.py exp/large-v2/ESC50/prompt0 

Results using this repository
-----------------------------------------

We list the classification accuracy using Whisper large-v2 obtained with prompt0, please refer to our paper for more detailed results.

| Model | ESC50 | US8K  | TUT | Vocal | RAVDESS | CREMA-D | GTZAN | LibriCount | **Avg.** |
|-------|-------|-------|-----|-------|---------|---------|-------|-----------|---------------| 
| Random | 2.0 | 10.0 | 6.7 | 16.7 | 12.5 | 16.7 | 10.0 | 9.1 | 10.4 |
| [AudioCLIP](https://arxiv.org/abs/2106.13043) | 69.4 | 65.3 | - | - | - | - | - | - | - |
| [CLAP](https://arxiv.org/abs/2206.04769)  | 82.6 | 73.2 | 29.6 | 49.4 | 16.0 | 17.8 | 25.2 | 17.9 | 39.0 |
|               |       |       |      |       |         |         |       |            |          |
| Ours, Uncalibrated | 38.9 | 50.5 | 7.7 | 60.1 | 15.1 | 20.2 | 38.2 | 9.2 | 30.0 |
| Ours, Zero-input | 35.9 | 52.1 | 18.0 | 57.5 | 29.4 | 26.5 | 45.8 | 13.6 | 34.9 |
| Ours, Prior-matched | 65.4 | 60.4 | 26.0 | 84.9 | 41.7 | 28.8 | 60.9 | 17.3 | **48.2** |


Citation
-----------------------------------------
    @article{ma2023investigating,
        title={Investigating the Emergent Audio Classification Ability of ASR Foundation Models},
        author={Ma, Rao and Liusie, Adian and Gales, Mark JF and Knill, Kate M},
        journal={arXiv preprint arXiv:2311.09363},
        year={2023}
    }

About Me
-----------------------------------------
https://julirao.github.io
