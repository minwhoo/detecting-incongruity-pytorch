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

DATASET_DIR = Path(os.environ.get('DATA_DIR')) / 'nela-18'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
    max_headline_len = 25
    max_para_len = 200
    max_num_para = 50

    print("Loading train dataset...")
    train_dataset = data.load_dataset(DATASET_DIR, 'train', max_headline_len, max_para_len, max_num_para, cache=True)
    print(f"Train dataset size: {len(train_dataset):9,}")
    print("Loading val dataset...")
    val_dataset = data.load_dataset(DATASET_DIR, 'dev', max_headline_len, max_para_len, max_num_para, cache=True)
    print(f"Val dataset size: {len(val_dataset):9,}")

    glove_embeds = data.load_glove(DATASET_DIR)

    # Initialize model
    hidden_dim = 300
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
        print("Loading test dataset...")
        test_dataset = data.load_dataset(DATASET_DIR, 'test', max_headline_len, max_para_len, max_num_para, cache=True)
        print(f"Test dataset size: {len(test_dataset):9,}")

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
