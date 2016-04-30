import pickle
import random
import numpy as np
import sys
import os
import pdb

import theano
import theano.tensor as T
import lasagne

from collections import Counter
from lasagne.utils import floatX 

import ConfigParser

import climate
logging = climate.get_logger(__name__)
climate.enable_default_logging()

def load_fn(data_fn):
    num_ins = 0
    fea_len = 0
    with open(data_fn,'r') as fid:
        for aline in fid:
            num_ins += 1
            parts = aline.strip().split()
            if fea_len == 0:
                fea_len = len(parts)
            else:
                assert(fea_len == len(parts))
    fea = np.zeros((num_ins, fea_len), dtype= 'int32')
    with open(data_fn,'r') as fid:
        for row, aline in enumerate(fid):
            parts = aline.strip().split()
            for col, num in enumerate(parts):
                fea[row, col] = int(num)
                
    return fea

if __name__== '__main__':

    cf = ConfigParser.ConfigParser()
    if len(sys.argv) < 2:
        print 'Usage: {0} <conf>'.format(sys.argv[0])
        sys.exit()

    cf.read(sys.argv[1])
    seq_len = cf.getint('INPUT', 'seq_len')
    batch_size = cf.getint('INPUT', 'batch_size')
    emb_size = cf.getint('INPUT', 'emb_size')
    hid_size = cf.getint('INPUT', 'hid_size')
    vocab_size = cf.getint('INPUT', 'vocab_size')
    epochs = cf.getint('INPUT', 'epochs')
    print_freq = cf.getint('INPUT', 'print_freq')
    val_freq= cf.getint('INPUT', 'val_freq')
    save_freq = cf.getint('INPUT', 'save_freq')

    train_fn = cf.get('INPUT', 'train_fn')
    val_fn = cf.get('INPUT', 'val_fn')


    save_dir = cf.get('OUTPUT', 'save_dir')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logging.info('starting building model')

    l_input = lasagne.layers.InputLayer((batch_size, seq_len-1))

    l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input,
                                                     input_size=vocab_size,
                                                     output_size=emb_size,
                                                    )

    l_dropout_input = lasagne.layers.DropoutLayer(l_sentence_embedding, p=0.5)
    l_lstm = lasagne.layers.LSTMLayer(l_dropout_input,
                                  num_units=hid_size,
                                  unroll_scan=True,
                                  grad_clipping=5.)

    l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)

    # the RNN output is reshaped to combine the batch and time dimensions
    # dim (BATCH_SIZE * SEQUENCE_LENGTH, EMBEDDING_SIZE)
    l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, hid_size))

    # decoder is a fully connected layer with one output unit for each word in the vocabulary
    l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=vocab_size, nonlinearity=lasagne.nonlinearities.softmax)

    # finally, the separation between batch and time dimension is restored
    l_out = lasagne.layers.ReshapeLayer(l_decoder, (batch_size, seq_len-1, vocab_size))


    x_sym = T.imatrix()

    # ground truth for the RNN output
    y_sym = T.imatrix()

    output = lasagne.layers.get_output(l_out, {
                l_input: x_sym
                })
    
    output_tst = lasagne.layers.get_output(l_out, {l_input: x_sym}, deterministic=True)
    def calc_cross_ent(net_output, targets):
        # Helper function to calculate the cross entropy error
        preds = T.reshape(net_output, (-1, vocab_size))
        targets = T.flatten(targets)
        cost = T.nnet.categorical_crossentropy(preds, targets)
        return cost

    loss = T.mean(calc_cross_ent(output, y_sym))
    loss_val = T.mean(calc_cross_ent(output_tst, y_sym))


    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    all_grads = T.grad(loss, all_params)
    all_grads = [T.clip(g, -5, 5) for g in all_grads]

    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=0.001)

    logging.info('compiling functions.')

    f_train = theano.function([x_sym, y_sym],
                          loss,
                          updates=updates
                         )

    f_val = theano.function([x_sym, y_sym], loss_val)
    
    logging.info('loading data...')

    fea_train = load_fn(train_fn)
    fea_val = load_fn(val_fn)

    for ep in xrange(epochs):
        logging.info('epoch %d/%d', ep+1, epochs)
        
        train_seqs = range(fea_train.shape[0])
        random.shuffle(train_seqs)
        
        batch_cnt = 0
        for batch_start in range(0, fea_train.shape[0], batch_size)[0:-1]:
            batch_cnt += 1
            # Just simply discard the last batch, which may not be the number of batch_size.
            batch_idx = train_seqs[batch_start:batch_start + batch_size]
            batch_data = fea_train[batch_idx,:]
            x_train = batch_data[:,0:-1]
            y_train = batch_data[:,1:]
            loss_train = f_train(x_train, y_train)

            if batch_cnt % print_freq == 0:
                logging.info('epoch %d/%d, loss = %f', ep+1, epochs, loss_train)

            if batch_cnt % val_freq == 0:
                val_seqs = range(fea_val.shape[0])
                random.shuffle(val_seqs)
                # Just randomly pick one batch and evaluate it.
                batch_idx = val_seqs[0:batch_size]
                batch_data = fea_val[batch_idx,:]
                x_val = batch_data[:,0:-1]
                y_val = batch_data[:,1:]
                loss_val= f_val(x_val, y_val)
                logging.info('epoch %d/%d, val loss = %f', ep+1, epochs, loss_val)
        if ep % save_freq == 0:
            param_values = lasagne.layers.get_all_param_values(l_out)
            emb_values = lasagne.layers.get_all_param_values(l_sentence_embedding)
            d = {'param_values': param_values, 'emb_values': emb_values}
            pickle.dump(d, open(os.path.join(save_dir, 'model_{}.pkl'.format(ep)),'w'))
    param_values = lasagne.layers.get_all_param_values(l_out)
    emb_values = lasagne.layers.get_all_param_values(l_sentence_embedding)
    d = {'param_values': param_values, 'emb_values': emb_values}
    pickle.dump(d, open(os.path.join(save_dir, 'model.pkl'),'w'))
