# -*- coding: utf-8 -*-
import numpy as np
import random


def get_batch(self, data, batch_size, encoder_size, context_size, encoderR_size, is_test, start_index=0, target_index=1):
    """ inputs: 
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
            encoder_pad = [self.pad_index] * (encoder_size - len( en_input ))
            tmp_encoder_inputs.append( (en_input + encoder_pad)[:encoder_size] )        
            tmp_encoder_seq.append( min( len( en_input ), encoder_size ) )    
        
        # add pad
        for i in range( context_size - list_len ):
            encoder_pad = [self.pad_index] * (encoder_size)
            tmp_encoder_inputs.append( encoder_pad )
            tmp_encoder_seq.append( 0 ) 

        encoder_inputs.extend( tmp_encoder_inputs[:context_size] )
        encoder_seq.extend( tmp_encoder_seq[:context_size] )
        
        context_seq.append( min(  len(list_encoder_input), context_size  ) )
        
        encoderR_length = np.where( encoderR_input==0 )[-1]
        if ( len(encoderR_length)==0 ) : encoderR_length = encoderR_size
        else : encoderR_length = encoderR_length[0]
        
        
        # encoderR inputs are padded
        encoderR_pad = [self.pad_index] * (encoderR_size - encoderR_length)
        encoderR_inputs.append( (encoderR_input.tolist() + encoderR_pad)[:encoderR_size])

        encoderR_seq.append( min(encoderR_length, encoderR_size) )

        # Target Label for batch
        target_labels.append( int(target_label) )
                            
                
    return encoder_inputs, encoderR_inputs, encoder_seq, context_seq, encoderR_seq, np.reshape(target_labels, (batch_size, 1))
