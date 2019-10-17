# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import data
from model import AttnHrDualEncoderModel
import utils

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


def evaluate(model, data):
    with torch.no_grad():
        model.eval()

        criterion = nn.BCEWithLogitsLoss()

        batch_size = 64
        val_dataloader = DataLoader(data, batch_size=batch_size)

        loss_sum = 0
        preds_all = []
        labels_all = []
        for batch in val_dataloader:
            headlines, headline_lengths, bodys, para_lengths, labels = tuple(b.to(device) for b in batch)

            preds = model(headlines, headline_lengths, bodys, para_lengths)

            loss = criterion(preds, labels)
            loss_sum += len(labels) * loss

            preds_all.append(preds)
            labels_all.append(labels)

        preds_all = torch.cat(preds_all)
        labels_all = torch.cat(labels_all)

        num_correct = torch.eq((preds_all > 0).int(), labels_all.int()).sum().item()
        num_total = len(preds_all)

        loss = loss_sum / num_total
        acc = num_correct / num_total
        auc = roc_auc_score(labels_all.tolist(), preds_all.tolist())

        return loss, acc, auc


def train(model, train_data, val_data):
    min_iterations = 2000
    max_iterations = 100000
    batch_size = 64

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    num_iterations_per_epoch = len(train_data) / batch_size
    val_eval_freq = int(0.1 * num_iterations_per_epoch)
    print(f"Val set evaluated every {val_eval_freq:,} steps (approx. 0.1 epoch)")

    es = utils.EarlyStopping(10)
    initial_time = time.time()

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    global_step = 0
    epoch_no = 0
    while True:
        print(f"EPOCH #{epoch_no+1}")

        # Train single epoch
        for batch in train_dataloader:
            headlines, headline_lengths, bodys, para_lengths, labels = tuple(b.to(device) for b in batch)
            optimizer.zero_grad()

            preds = model(headlines, headline_lengths, bodys, para_lengths)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            
            global_step += 1

            if global_step % val_eval_freq == 0:
                # Evaluate on validation set
                val_loss, val_acc, val_auc = evaluate(model, val_data)
                end_time = time.time()
                minutes_elapsed = int((end_time - initial_time)/60)
                print("STEP: {:7} | TIME: {:4}min | VAL LOSS: {:.4f} | VAL ACC: {:.4f} | VAL AUROC: {:.4f}".format(
                    global_step, minutes_elapsed, val_loss, val_acc, val_auc
                ))
                model.train()

                # Check early stopping
                if global_step >= min_iterations:
                    es.record_loss(val_loss, model)

                if es.should_stop():
                    print(f"Early stopping at STEP: {global_step}...")
                    return

            if global_step == max_iterations:
                print(f"Stopping after reaching max iterations({global_step})...")
                return
        epoch_no += 1


def main():
    print(f"Using device: {device}")

    # Load data
    print("Loading processed data...")
    max_headline_len = 25
    max_para_len = 200
    max_num_para = 50

    with open(PROCESSED_DATA_DIR / 'train_set.pkl', 'rb') as f:
        train_data = pickle.load(f, encoding='latin1')
        print(f"Train set size: {len(train_data):9,}")
        train_dataset = data.load_dataset(train_data, max_headline_len, max_para_len, max_num_para)
    with open(PROCESSED_DATA_DIR / 'valid_set.pkl', 'rb') as f:
        val_data = pickle.load(f, encoding='latin1')
        print(f"Valid set size: {len(val_data):9,}")
        val_dataset = data.load_dataset(val_data, max_headline_len, max_para_len, max_num_para)
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
    train(model, train_dataset, val_dataset)

    # Evaluate test
    evaluate_test_after_train = True
    if evaluate_test_after_train:
        print("Loading test set...")
        with open(PROCESSED_DATA_DIR / 'test_set.pkl', 'rb') as f:
            test_data = pickle.load(f, encoding='latin1')
            print(f"Test set size: {len(test_data):9,}")
            test_dataset = data.load_dataset(test_data, max_headline_len, max_para_len, max_num_para)
        print("Loading done!")

        if utils.CHECKPOINT_SAVE_PATH.exists():
            print(f"Loading model checkpoint with min val loss...")
            model.load_state_dict(torch.load(utils.CHECKPOINT_SAVE_PATH))
            print(f"Loading done!")

        _, test_acc, test_auc = evaluate(model, test_dataset)

        print(f"TEST ACC: {test_acc:.4f} | TEST AUROC: {test_auc:.4f}")

        if utils.CHECKPOINT_SAVE_PATH.exists():
            utils.CHECKPOINT_SAVE_PATH.unlink()  # delete model


if __name__ == "__main__":
    main()
