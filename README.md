# Detecting Incongruity in PyTorch
PyTorch implementation of Attentive Hierarchical Dual Encoder model from the following paper:

**Detecting Incongruity Between News Headline and Body Text via a Deep Hierarchical Encoder**, AAAI-19, [paper](https://arxiv.org/abs/1811.07066)

Original Tensorflow implementation can be found [here](https://github.com/david-yoon/detecting-incongruity) [1]
## Requirements
 - Python 3.6 or greater
 - PyTorch 1.2.0

## Installation
 `pip install -r requirements.txt`

## Download Dataset
Follow instructions from the original Tensorflow repo [1].

## Train
### NELA 2017
```
python main.py --data-dir <PATH_TO_NELA_2017_DATA> \
               --max-headline-len 25 \
               --max-para-len 200 \
               --max-num-para 50 \
               --headline-rnn-hidden-dim 200 \
               --word-level-rnn-hidden-dim 200 \
               --paragraph-level-rnn-hidden-dim 100 \
               --lr 0.001 \
               --batch-size 64 \
               --evaluate-test-after-train

```

## References
[1] https://github.com/david-yoon/detecting-incongruity

## Cite
Please cite our paper, when you use our code | dataset | model

> @inproceedings{yoon2019detecting,<br>
> title={Detecting Incongruity between News Headline and Body Text via a Deep Hierarchical Encoder},<br>
> author={Yoon, Seunghyun and Park, Kunwoo and Shin, Joongbo and Lim, Hongjun and Won, Seungpil and Cha, Meeyoung and Jung, Kyomin},<br>
>  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},<br>
>  volume={33},<br>
>  pages={791--800},<br>
>  year={2019}<br>
> }