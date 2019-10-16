# -*- coding: utf-8 -*-

import dataloader
import os
from pathlib import Path
import pickle

PROCESSED_DATA_DIR = Path(os.environ.get('DATA_DIR')) / 'nela-17/whole/processed'


def main():
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


if __name__ == "__main__":
    main()
