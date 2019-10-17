# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import random
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F


def create_padded_tensor(input_tensor, seq_length, dtype, value=0):
    output = torch.tensor(input_tensor[:seq_length], dtype=dtype)
    pad_len = seq_length - len(output)
    return F.pad(output, (0, pad_len), value=value)


def load_dataset(data, max_headline_len, max_para_len, max_num_para):
    samples = []
    headline_tensors = []
    headline_lengths = []
    body_tensors = []
    para_length_tensors = []
    labels = []
    count_no_body = 0
    for d in data:
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

        para_length_tensor = create_padded_tensor([len(p[:max_para_len]) for p in paragraphs], max_num_para, dtype=torch.int)
        para_length_tensors.append(para_length_tensor)

        # label
        labels.append(int(d[2]))
    print(count_no_body)

    return TensorDataset(torch.stack(headline_tensors), torch.tensor(headline_lengths, dtype=torch.int), torch.stack(body_tensors), torch.stack(para_length_tensors), torch.tensor(labels, dtype=torch.float))


def get_batch(data, batch_size, encoder_size, context_size, encoderR_size, is_test, start_index=0, target_index=1, pad_index=0):
    """Get batch of data from processed dataset.
    inputs: 
            data: 
            batch_size : 
            encoder_size : max encoder time step
            context_size : max context encoding time step
            encoderR_size : max decoder time step
            
            is_test : batch data generation in test case
            start_index : 
            target_index : 0, 1

        return:
            encoder_inputs : [batch x context_size, time_step]
            encoderR_inputs : [batch, time_step]
            encoder_seq :
            context_seq  :
            encoderR_seq :
            target_labels : label
    """
    encoder_inputs, encoderR_inputs, encoder_seq, context_seq, encoderR_seq, target_labels = [], [], [], [], [], []
    index = start_index
    
    # Get a random batch of encoder and encoderR inputs from data,
    # pad them if needed

    for _ in range(batch_size):

        if is_test is False:
            encoderR_input, list_encoder_input, target_label = random.choice(data)
        else:
            # overflow case
            if index > len(data)-1:
                list_encoder_input = data[-1][1]
                encoderR_input = data[-1][0]
                target_label = data[-1][2]
            else:
                list_encoder_input = data[index][1]
                encoderR_input = data[index][0]
                target_label = data[index][2]

            index = index +1

        list_len = len( list_encoder_input )
        tmp_encoder_inputs = []
        tmp_encoder_seq = []
        
        for en_input in list_encoder_input:
            encoder_pad = [pad_index] * (encoder_size - len( en_input ))
            tmp_encoder_inputs.append( (en_input + encoder_pad)[:encoder_size] )        
            tmp_encoder_seq.append( min( len( en_input ), encoder_size ) )    
        
        # add pad
        for i in range( context_size - list_len ):
            encoder_pad = [pad_index] * (encoder_size)
            tmp_encoder_inputs.append( encoder_pad )
            tmp_encoder_seq.append( 0 ) 

        encoder_inputs.extend( tmp_encoder_inputs[:context_size] )
        encoder_seq.extend( tmp_encoder_seq[:context_size] )
        
        context_seq.append( min(  len(list_encoder_input), context_size  ) )
        
        encoderR_length = np.where( encoderR_input==0 )[-1]
        if ( len(encoderR_length)==0 ) : encoderR_length = encoderR_size
        else : encoderR_length = encoderR_length[0]
        
        
        # encoderR inputs are padded
        encoderR_pad = [pad_index] * (encoderR_size - encoderR_length)
        encoderR_inputs.append( (encoderR_input.tolist() + encoderR_pad)[:encoderR_size])

        encoderR_seq.append( min(encoderR_length, encoderR_size) )

        # Target Label for batch
        target_labels.append( int(target_label) )
                            
                
    return encoder_inputs, encoderR_inputs, encoder_seq, context_seq, encoderR_seq, np.reshape(target_labels, (batch_size, 1))
