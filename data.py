# -*- coding: utf-8 -*-

from pathlib import Path
import pickle
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from tqdm import trange, tqdm


def load_dataset(data_path, dataset_type, max_headline_len, max_para_len, max_num_para, cache, cache_dir):
    if cache:
        cache_file_name = "{}_{}_Lh{}_Lp{}_P{}.pt".format(
            data_path.stem, dataset_type, max_headline_len, max_para_len, max_num_para)
        cache_path = cache_dir / cache_file_name
        if cache_path.exists():
            print("Cached dataset found!")
            return torch.load(cache_path)
        else:
            print("Cached dataset not found. Dataset will be cached after loading")

    assert dataset_type in ['train', 'dev', 'test', 'debug']

    numpy_data = {}
    numpy_data['c'] = np.load(data_path / "whole/{}/{}_title.npy".format(dataset_type, dataset_type))
    numpy_data['r'] = np.load(data_path / "whole/{}/{}_body.npy".format(dataset_type , dataset_type))
    numpy_data['y'] = np.load(data_path / "whole/{}/{}_label.npy".format(dataset_type , dataset_type))
    with open(data_path / "whole/dic_mincutN.pkl", "rb") as f:
        voca = pickle.load(f, encoding='latin1')
        eop_voca = voca['<EOP>']
    
    processed_data = _process_numpy_data(numpy_data, eop_voca)

    dataset = _create_tensor_dataset(processed_data, max_headline_len, max_para_len, max_num_para)

    if cache:
        print("Caching dataset...")
        torch.save(dataset, cache_path)
        print(f"Caching done! Located at {cache_path}")
    return dataset


def load_glove(data_path):
    return torch.tensor(np.load(open(data_path / "whole/W_embedding.npy", 'rb')))


def _process_numpy_data(input_data, eop_voca):
    """
    Basically same functionality as
      ``src_whole.AHDE_process_data.ProcessData.create_data_set``
    from david-yoon/detecting-incongruity
    """
    output_set = []
    
    data_len = len(input_data['c'])
    for index in trange(data_len, desc='STEP #1/2: Split body by paragraph'):
        delimiter = ' ' +  str(eop_voca) + ' '
        # last == padding
        turn =[x.strip() for x in (' '.join(str(e) for e in input_data['r'][index])).split(delimiter)[:-1] ]
        turn = [ x for x in turn if len(x) >1]
        
        tmp_ids = [x.split(' ') for x in turn]
        target_ids = []
        for sent in tmp_ids:
            target_ids.append( [ int(x) for x in sent]  )

        source_ids = input_data['c'][index]
        
        label = float(input_data['y'][index])
        
        output_set.append( [source_ids, target_ids, label] )
    
    return output_set


def create_padded_tensor(input_tensor, seq_length, dtype, value=0):
    output = torch.tensor(input_tensor[:seq_length], dtype=dtype)
    pad_len = seq_length - len(output)
    return F.pad(output, (0, pad_len), value=value)


def _create_tensor_dataset(data, max_headline_len, max_para_len, max_num_para):
    samples = []
    headline_tensors = []
    headline_lengths = []
    body_tensors = []
    para_length_tensors = []
    labels = []
    count_no_body = 0
    for d in tqdm(data, desc="STEP #2/2: Pad and convert to tensor"):
        # headline
        headline_tensor = create_padded_tensor(d[0], max_headline_len, dtype=torch.long)
        headline_tensors.append(headline_tensor)

        headline_lengths.append(min(len(np.nonzero(d[0])[0]), max_headline_len))

        # body
        if len(d[1]) == 0:
            count_no_body += 1
            headline_tensors.pop()
            headline_lengths.pop()
            continue

        paragraphs = d[1][:max_num_para]
        empty_paragraphs = [[]] * (max_num_para - len(paragraphs))
        para_tensors = []
        for para in paragraphs + empty_paragraphs:
            para_tensor = create_padded_tensor(para, max_para_len, dtype=torch.long)
            para_tensors.append(para_tensor)
        body_tensor = torch.stack(para_tensors, dim=0)
        body_tensors.append(body_tensor)

        para_length_tensor = create_padded_tensor([len(p[:max_para_len]) for p in paragraphs], 
                                                  max_num_para, dtype=torch.int)
        para_length_tensors.append(para_length_tensor)

        # label
        labels.append(int(d[2]))
    print(count_no_body)

    return TensorDataset(torch.stack(headline_tensors), 
                         torch.tensor(headline_lengths, dtype=torch.int), 
                         torch.stack(body_tensors), 
                         torch.stack(para_length_tensors), 
                         torch.tensor(labels, dtype=torch.float))
