# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pickle

import numpy as np
import torch

import dataloader
from model import AttnHrDualEncoderModel

DATA_DIR = Path(os.environ.get('DATA_DIR'))
PROCESSED_DATA_DIR =  DATA_DIR / 'nela-17/whole/processed'
PROCESSED_GLOVE_PATH = DATA_DIR / 'nela-17/whole/W_embedding.npy'


def train(model, data):
    pass


def main():
    # Load data
    print("Loading processed data...")
    with open(PROCESSED_DATA_DIR / 'train_set.pkl', 'rb') as f:
        train_set = pickle.load(f, encoding='latin1')
    print("Loading done!")

    batch_size = 64
    encoder_size = 200
    context_size = 50
    encoderR_size = 25
    batch = dataloader.get_batch(train_set, batch_size, encoder_size, context_size, encoderR_size, is_test=False)
    raw_encoder_inputs, raw_encoderR_inputs, raw_encoder_seq, raw_context_seq, raw_encoderR_seq, raw_target_label = batch
    input_body = torch.tensor(raw_encoder_inputs)
    input_headline = torch.tensor(raw_encoderR_inputs)
    labels = torch.tensor(raw_target_label)

    # Initialize model
    hidden_dim = 300
    glove_embeds = torch.tensor(np.load(open(PROCESSED_GLOVE_PATH, 'rb')))
    vocab_size, embedding_dim = glove_embeds.shape
    model = AttnHrDualEncoderModel(hidden_dim, vocab_size, embedding_dim, pretrained_embeds=glove_embeds)
    preds = model((input_headline, input_body))
    print(preds.shape)
    print(labels.shape)


if __name__ == "__main__":
    main()
