import argparse, torch, Tags, numpy as np
from transformer import Transformer as t
import TranslationDataset as td
import torch.nn.functional as F
import math, time
from tqdm import tqdm
from transformer.Optimizer import ScheduledOptim
import torch.optim as optim
from transformer.utils import *


def train_epoch(model, training_data, optimizer, device, idx2word, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)


        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Tags.PAD_ID)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device, idx2word):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]
            batch_size, seq_length = gold.size()

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

            loss, n_correct = cal_performance(pred, gold, smoothing=False)
            prb = F.softmax(pred, dim=1)
            pred_sentences = greedy_decode(prb, idx2word, batch_size, seq_length)


            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Tags.PAD_ID)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer, device, opt, idx2word):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, idx2word, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, idx2word)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = "model/" + opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-d_model', type=int, default=300)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-save_param', type=str, default="model/transformer_params")

    options = parser.parse_args()

    # Set device type
    device = torch.device("cuda" if options.cuda else "cpu")

    # Load preprocessed data
    data = torch.load(options.data)
    training_data, dev_data = prepare_dataloaders(data, options)


    transformer = t.Transformer(
        src_embedding = data["glove"]["src"],
        tgt_embedding = data["glove"]["tgt"],
        len_max_seq = data["options"].max_len,
        d_k=options.d_k,
        d_v=options.d_v,
        d_model=options.d_model,
        d_word_vec=options.d_word_vec,
        d_inner=options.d_inner_hid,
        n_layers=options.n_layers,
        n_head=options.n_head,
        dropout=options.dropout).to(device)

    transformer_params = {
        "src_embedding" : data["glove"]["src"],
        "tgt_embedding" : data["glove"]["tgt"],
        "len_max_seq" : data["options"].max_len,
        "d_k" : options.d_k,
        "d_v" : options.d_v,
        "d_model" : options.d_model,
        "d_word_vec" : options.d_word_vec,
        "d_inner" : options.d_inner_hid,
        "n_layers" :options.n_layers,
        "n_head" : options.n_head,
        "dropout" : options.dropout,
        "tgt_idx2word": data["idx2word"]["tgt"],
        "batch_size" : options.batch_size}

    torch.save(transformer_params, options.save_param)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        options.d_model, options.n_warmup_steps)

    train(transformer, training_data, dev_data, optimizer, device, options, data["idx2word"]["tgt"])


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        td.TranslationDataset(
            src_sentences=data['train']['src'],
            tgt_sentences=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        td.TranslationDataset(
            src_sentences=data['dev']['src'],
            tgt_sentences=data['dev']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader

if __name__ == "__main__":
    main()