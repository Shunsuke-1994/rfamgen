# coding: UTF-8

import sys
sys.path.append("./src")
import os
import csv
import torch
import torch.nn as nn
import numpy as np 

def modifiedCELoss(pred, soft_targets, gamma = 0, summarize = True):
    """
    shape: (BATCH, N_COL, N_RULE)
    gamma for focal loss
    calc sum of a column and normalize a col by the sum, then calc CEloss.
    Sum CELoss for each column and normalize it by the number of columns.
    summarize: column-wise sum.
    """
    scale = soft_targets.nansum(dim = -1).unsqueeze(dim = -1)
    logsoftmax = nn.LogSoftmax(dim = -1)
    sotfmax = nn.Softmax(dim = -1)
    ce = - soft_targets/scale * ((1-sotfmax(pred)).pow(gamma)) * logsoftmax(pred)
    ce_colwise = torch.sum(ce, dim = -1)
    if summarize:
        return torch.sum(ce)
    else:
        return ce.sum(dim=-1).sum(dim=-1)

def save_model(model, dir_name, pt_file):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    torch.save(model.state_dict(), os.path.join(dir_name , pt_file))

def write_csv(d, dir_name, fname):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    path = os.path.join(dir_name , fname)
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(d.keys())
        writer.writerows(zip(*d.values()))
    print(f"Saved the log csv at {path}")

if __name__ == "__main__":
    import sys
    sys.path.append("./src")
    import random
    import yaml
    import argparse
    from pprint import pprint
    from models.CMVAE import CovarianceModelVAE, MyDataset
    from util import Timer, AnnealKL
    from torch.utils.data import DataLoader


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, required=True, help = "directory containing X_train etc.")
    parser.add_argument('--X_train', type = str, required=True, help = "file name of training data")
    parser.add_argument('--w_train', default = "", required=True, type = str, help = "file name of weight for training data")
    parser.add_argument('--X_valid', default = "", required=True, type = str, help = "file name of validation data")
    parser.add_argument('--w_valid', default = "", required=True, type = str, help = "file name of weight for validation data")
    parser.add_argument('--suffix', default = "", type = str, help = "suffix of output file name")
    
    parser.add_argument("--hidden", default = 128, type = int, help = "dimension of hidden space flanking latent space (default: 128)")
    parser.add_argument("--z_dim", default = 16, type = int, help = "dimension of latent space (default: 16)")

    parser.add_argument("--stride", default = 1, type = int, help = "stride parameter of all convs (default: 1)")
    parser.add_argument("--ker1", default = 5, type = int, help = "kernel size parameter of 1st conv (default: 5)")
    parser.add_argument("--ch1", default = 5, type = int, help = "channel num parameter of 1st conv (default: 5)")
    parser.add_argument("--ker2", default = 5, type = int, help = "kernel parameter of 2nd conv (default: 5)")
    parser.add_argument("--ch2", default = 5, type = int, help = "channel num parameter of 2nd conv (default: 5)")
    parser.add_argument("--ker3", default = 7, type = int, help = "kernel parameter of 3rd conv (default: 7)")
    parser.add_argument("--ch3", default = 8, type = int, help = "channel num parameter of 3rd conv (default: 8)")

    parser.add_argument("--beta", type = float, help = "maximum weight of KL divergence")
    parser.add_argument("--use_anneal", action = "store_true", help = "flag for cyclic annealing for KL divergence")
    parser.add_argument("--anneal_saturate_rate", default = 0.4, type = float, help = "maximum rate for cyclic annealing (default: 0.4)")
    parser.add_argument("--anneal_rate", default = 1, type = float, help = "number of iteration for one increment of annealing (default: 1)")
    parser.add_argument("--batch_size", default = 8, type = int, help = "batch size (defailt: 8)")
    # parser.add_argument("--dropout_rate", default = 0.0, type = float)
    parser.add_argument("--epoch", default = 200, type = int, help = "maximum epochs (default: 200)")
    parser.add_argument("-lr", "--learning_rate", default = 1e-3, type = float, help = "learning rate (default: 1e-3)")
    parser.add_argument("--clip", default = 20, type = float, help = "clipping value (default: 20)")

    parser.add_argument("--only_training", action = "store_true", help = "only training")

    parser.add_argument("--use_early_stopping", action = "store_true", help = "flag for early stopping")
    parser.add_argument("--tolerance", default = 3, type = int, help = "tolerance for early stopping (default: 3)")
    # parser.add_argument("--use_shuffle", action = "store_true")

    parser.add_argument('--save_ckpt', action = "store_true", help = "flag for checkpoint saving")
    parser.add_argument('--ckpt_iter', default = 3, type = int, help = "save checkpoint every this iteration (default: 3)")

    parser.add_argument("--random_seed", default = 42, type = int)
    parser.add_argument('--log', action = "store_true", help = "flag for log")
    parser.add_argument('--log_dir', type = str, help = "directory for log output")
    parser.add_argument("--print_every", default = 20, type = int, help = "iteration num to print log of learning (default: 20)")
    args = parser.parse_args()

    # training
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    timer = Timer()

    train_dataset = MyDataset(
        path = os.path.join(args.data_dir, args.X_train),
        weight_path = os.path.join(args.data_dir, args.w_train) if args.w_train != "" else ""
        )
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4, drop_last = True)

    if not args.only_training:
        valid_dataset = MyDataset(
            path = os.path.join(args.data_dir, args.X_valid),
            weight_path = os.path.join(args.data_dir, args.w_valid) if args.w_valid != "" else ""
            )
        valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4, drop_last = True)

    TR_LEN = train_dataset.data["tr"].shape[-2]
    S_LEN = train_dataset.data["s"].shape[-2]
    P_LEN = train_dataset.data["p"].shape[-2]
    DATA_SIZE = train_dataset.data["tr"].shape[0]
    
    conv_params = {
        "ker1":args.ker1, "ch1":args.ch1,
        "ker2":args.ker2, "ch2":args.ch2,
        "ker3":args.ker3, "ch3":args.ch3
    }

    model = CovarianceModelVAE(
        hidden_encoder_size = args.hidden, 
        z_dim = args.z_dim,
        hidden_decoder_size = args.hidden, 
        tr_len = TR_LEN,
        s_len = S_LEN, 
        p_len = P_LEN,
        stride = args.stride,
        # dropout_rate = args.dropout_rate,
        conv_params = conv_params,
    )
    anneal = AnnealKL(step = (DATA_SIZE / args.batch_size * args.anneal_saturate_rate)**-1, rate = args.anneal_rate)
    beta_sum_batch = args.beta * args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def train_model(log):
        for step, tr_s_p_w in enumerate(train_dataloader, 0):
            tr, s, p, w  = tr_s_p_w
            tr = tr.to(model.device)
            s = s.to(model.device)
            p = p.to(model.device)
            w = w.to(model.device)

            mu, logvar = model.encoder((tr, s, p))
            z = model.sample(mu, logvar)
            tr_, s_, p_ = model.decoder(z) 
            tr_loss = modifiedCELoss(tr_.transpose(-1, -2), tr.transpose(-1, -2), summarize = False)
            s_loss = modifiedCELoss(s_.transpose(-1, -2), s.transpose(-1, -2), summarize = False)
            p_loss = modifiedCELoss(p_.transpose(-1, -2), p.transpose(-1, -2), summarize = False)
            loss = (tr_loss + s_loss + p_loss)
            w = w.reshape(args.batch_size)
            loss = (w*(tr_loss + s_loss + p_loss)).sum() #/BATCH_SIZE
            kl = model.kl(mu, logvar)
            
            alpha = beta_sum_batch * anneal.alpha(step) if args.use_anneal else beta_sum_batch
            elbo = loss + alpha*kl

            # update parameters
            optimizer.zero_grad()
            elbo.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = args.clip)
            optimizer.step()
            
            # Logging info
            log['loss'].append(loss.to(torch.device('cpu')).clone().data.numpy())
            log['kl'].append(kl.to(torch.device('cpu')).clone().data.numpy())
            log['elbo'].append(elbo.to(torch.device('cpu')).clone().data.numpy())
            log['alpha'].append(alpha)
            if step % args.print_every == 0:
                print(
                    '| step {}/{} \t| loss {:.4f}\t| kl {:.4f}\t|'
                    ' elbo {:.4f}\t| alpha {:.6f}\t| {:.0f} sents/sec\t|'.format(
                        step, DATA_SIZE // args.batch_size,
                        np.mean(log['loss'][-args.print_every:]),
                        np.mean(log['kl'][-args.print_every:]),
                        np.mean(log['elbo'][-args.print_every:]),
                        np.mean(log['alpha'][-1]),
                        args.batch_size * args.print_every / timer.elapsed()
                        )
                    )
        return log

    def valid_model(log_valid):
        tmp_loss = []
        tmp_kl = []
        tmp_elbo = []

        for step, tr_s_p_w in enumerate(valid_dataloader, 0):
            tr, s, p, w  = tr_s_p_w
            tr = tr.to(model.device)
            s = s.to(model.device)
            p = p.to(model.device)
            w = w.to(model.device)

            mu, logvar = model.encoder((tr, s, p))
            z = model.sample(mu, logvar)
            tr_, s_, p_ = model.decoder(z)
            tr_loss = modifiedCELoss(tr_.transpose(-1, -2), tr.transpose(-1, -2), summarize = False)
            s_loss = modifiedCELoss(s_.transpose(-1, -2), s.transpose(-1, -2), summarize = False)
            p_loss = modifiedCELoss(p_.transpose(-1, -2), p.transpose(-1, -2), summarize = False)
            w = w.reshape(args.batch_size)
            loss = (w*(tr_loss + s_loss + p_loss)).sum() #/BATCH_SIZE
            kl = model.kl(mu, logvar)
            elbo = loss + beta_sum_batch*kl

            tmp_loss.append(loss.item())
            tmp_kl.append(kl.item())
            tmp_elbo.append(elbo.item())

        log_valid["loss_valid"].append(np.mean(tmp_loss))
        log_valid["kl_valid"].append(np.mean(tmp_kl))
        log_valid["elbo_valid"].append(np.mean(tmp_elbo))

        print(
            '| valid {}/{}\t| loss {:.4f}\t| kl {:.4f}\t|'
            ' elbo {:.4f}\t|'.format(
                epoch, args.epoch,
                log_valid["loss_valid"][-1],
                log_valid["kl_valid"][-1],
                log_valid["elbo_valid"][-1]
                )
        )
        print('=' * 69)
        return log_valid

    log = {'loss': [], 'kl': [], 'elbo': [], 'alpha':[]}
    log_valid = {"loss_valid": [], "kl_valid": [], "elbo_valid" : []}

    try:
        # save config
        config_dict = {
            "DATA_DIR" : args.data_dir,
            "X_TRAIN" : args.X_train,
            "W_TRAIN" : args.w_train,
            "X_VALID" : args.X_valid,
            "W_VALID" : args.w_valid,
            "EPOCH" : args.epoch,
            "HIDDEN" : args.hidden,
            "TR_WIDE" : TR_LEN,
            "S_WIDE" : S_LEN,
            "P_WIDE" : P_LEN,
            "BATCH_SIZE" : args.batch_size,
            #"DROPOUT_RATE" : args.dropout_rate,
            "ONLY_TRAINING" : args.only_training,
            "EARLY_STOPPING" : args.use_early_stopping,
            "EARLY_STOPPING_THRESHOLD" : args.tolerance,
            "LEARNING_RATE" : args.learning_rate,
            "PRINT_EVERY" : args.print_every,
            "USE_ANNEAL" : args.use_anneal,
            "ANNEAL_SATURATE_RATE" :args.anneal_saturate_rate,
            "ANNEAL_RATE" : args.anneal_rate,
            # "USE_SHUFFLE" : args.use_shuffle,
            "CLIP" :args.clip,
            "Z_DIM" : args.z_dim,

            "STRIDE" : args.stride,
            "KER1" : args.ker1,
            "CH1" : args.ch1,
            "KER2" : args.ker2,
            "CH2" : args.ch2,
            "KER3" : args.ker3,
            "CH3" : args.ch3,

            "BETA" :args.beta,
            "BETS_SUM" :beta_sum_batch,
            "CKPT_ITER" : args.ckpt_iter,
            "SUFFIX": args.suffix,
            "LOG": args.log, 
            "LOG_DIR": args.log_dir, 
            "PRINT_EVERY": args.print_every
            }
        pprint(config_dict)
        with open(os.path.join(args.log_dir, f"config{args.suffix}.yaml"), "w") as f:
            yaml.dump(config_dict, f)

        for epoch in range(1, args.epoch + 1):
            print('-' * 90)
            print('Epoch {}/{}'.format(epoch, args.epoch))
            print('-' * 90)
            model.train()
            log = train_model(log)
             
            if not args.only_training:
                model.eval()
                with torch.no_grad():
                    log_valid = valid_model(log_valid)

            # save model
            if args.save_ckpt:
                if epoch % args.ckpt_iter == 0:
                    save_model(model, args.log_dir, f'model_epoch{epoch}{args.suffix}.pt')

            # ealy stopping
            if args.use_early_stopping and (epoch > args.tolerance) and (not args.only_training):
                elbo_diff = np.diff(np.array(log_valid['elbo_valid'][-args.tolerance-1:]))
                if all(elbo_diff>0):
                    print(f'Early stopping at epoch {epoch}')
                    break

        # save final ver.
        if args.log: 
            save_model(model, args.log_dir, f'model_epoch{epoch}{args.suffix}.pt')
            write_csv(log, args.log_dir, f'log{args.suffix}.csv')
            write_csv(log_valid, args.log_dir, f'log_valid{args.suffix}.csv')

    except KeyboardInterrupt:
        print('-' * 90)
        print('Exiting training early')
        print('-' * 90)

