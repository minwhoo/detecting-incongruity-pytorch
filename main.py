# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

import data
from model import AttnHrDualEncoderModel

DATA_DIR = Path(os.environ.get('DATA_DIR'))
PROCESSED_DATA_DIR =  DATA_DIR / 'nela-17/whole/processed'
PROCESSED_GLOVE_PATH = DATA_DIR / 'nela-17/whole/W_embedding.npy'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def iter_batch(dataset, batch_size, is_test=False):
    encoder_size = 200
    context_size = 50
    encoderR_size = 25
    for i in range(0, len(dataset), batch_size):
        # Call batch generator
        batch = data.get_batch(dataset, batch_size, encoder_size, context_size, encoderR_size, start_index=i, is_test=is_test)
        raw_encoder_inputs, raw_encoderR_inputs, raw_encoder_seq, raw_context_seq, raw_encoderR_seq, raw_target_label = batch

        # Convert data to torch tensors
        input_headline = torch.tensor(raw_encoderR_inputs)
        input_body = torch.tensor(raw_encoder_inputs)
        labels = torch.tensor(raw_target_label, dtype=torch.float32)

        # Yield data
        yield (input_headline, input_body), labels


def evaluate(model, data, criterion):
    with torch.no_grad():
        model.eval()

        batch_size = 64
        num_correct = 0
        num_total = 0
        loss_sum = 0
        for inputs, labels in iter_batch(data, batch_size, is_test=True):
            inputs, labels = tuple(x.to(device) for x in inputs), labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)
            loss_sum += len(labels) * loss
            num_correct += torch.eq(torch.round(preds), labels).sum().item()
            num_total += len(labels)

        loss = loss_sum / num_total
        acc = num_correct / num_total

        return loss, acc


def train(model, train_data, val_data):
    num_epoch = 10
    batch_size = 64

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    num_iterations_per_epoch = len(train_data) / batch_size
    val_eval_freq = int(0.1 * num_iterations_per_epoch)
    print(f"Val set evaluated every {val_eval_freq:,} steps (approx. 0.1 epoch)")

    initial_time = time.time()

    global_step = 0
    for i in range(num_epoch):
        print(f"EPOCH {i+1}")

        # Train single epoch
        for inputs, labels in iter_batch(train_data, batch_size):
            inputs, labels = tuple(x.to(device) for x in inputs), labels.to(device)
            optimizer.zero_grad()

            preds = model(inputs)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            
            global_step += 1
            if global_step % val_eval_freq == 0:
                val_loss, val_acc = evaluate(model, val_data, criterion)
                end_time = time.time()
                print(f"STEP: {global_step:7} | TIME: {int((end_time - initial_time)/60):4}min | VAL LOSS: {val_loss:.4f} | VAL ACC: {val_acc:.4f}")
                model.train()


def main():
    print(f"Using device: {device}")

    # Load data
    print("Loading processed data...")
    with open(PROCESSED_DATA_DIR / 'train_set.pkl', 'rb') as f:
        train_set = pickle.load(f, encoding='latin1')
        print(f"Train set size: {len(train_set):9,}")
    with open(PROCESSED_DATA_DIR / 'valid_set.pkl', 'rb') as f:
        valid_set = pickle.load(f, encoding='latin1')
        print(f"Valid set size: {len(valid_set):9,}")
    print("Loading done!")

    # Initialize model
    hidden_dim = 300
    glove_embeds = torch.tensor(np.load(open(PROCESSED_GLOVE_PATH, 'rb')))
    vocab_size, embedding_dim = glove_embeds.shape

    print("Initializing model...")
    model = AttnHrDualEncoderModel(hidden_dim, vocab_size, embedding_dim, pretrained_embeds=glove_embeds)
    model.to(device)
    print("Initialization done!")

    # Train
    train(model, train_set, valid_set)


if __name__ == "__main__":
    main()
