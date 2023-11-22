# activity prediction by CMVAE. 

import os 
import sys 
sys.path.append("/Users/sumishunsuke/desktop/RNA/genzyme/src")
import util
import torch 
import torch.nn as nn 
import numpy as np 
import random 
import h5py 
from models.CMVAE import CovarianceModelVAE


def load_data_cm(path):
    data = h5py.File(path, "r")
    tr = torch.from_numpy(data["tr"][:]).nan_to_num(0).transpose(-2, -1).float()
    s = torch.from_numpy(data["s"][:]).transpose(-2, -1).float()
    p = torch.from_numpy(data["p"][:]).transpose(-2, -1).float()
    return tr, s, p


def modifiedCELoss(pred, soft_targets, gamma = 0, summarize = True):
    """
    shape: (BATCH, N_COL, N_RULE)
    gamma for focal loss
    targetの列ごとの値scaleを求め, predとtarget/scaleのCELossをとる
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


def eval_ELBO(model, tr_s_p, beta, n_samples = 25):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tr,s,p = tr_s_p
    with torch.no_grad():
        mu, logvar = model.encoder((tr, s, p))
        eve = []
    for i in range(n_samples):
        z = model.sample(mu, logvar)
        tr_, s_, p_ = model.decoder(z) 

        tr_loss = modifiedCELoss(tr_.transpose(-1, -2), tr.transpose(-1, -2), summarize = False)
        s_loss = modifiedCELoss(s_.transpose(-1, -2), s.transpose(-1, -2), summarize = False)
        p_loss = modifiedCELoss(p_.transpose(-1, -2), p.transpose(-1, -2), summarize = False)
        loss = (tr_loss + s_loss + p_loss) #/BATCH_SIZE
        kl = model.kl(mu, logvar)
        elbo = loss + beta * kl
        eve.append(elbo)
    return torch.stack(eve).mean()


if __name__ == "__main__":
    import argparse
    import os 
    import sys 
    sys.path.append("src")
    import util

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_trsp', '-i')
    parser.add_argument('--model_config')
    parser.add_argument('--ckpt')
    parser.add_argument('--n_samples', default = 25, type = int)
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    model = CovarianceModelVAE.build_from_config(args.model_config)
    config = util.load_config(args.model_config)
    beta = config["BETA"]

    model.load_model_from_ckpt(args.ckpt)
    model.eval();

    tr, s, p = load_data_cm(args.in_trsp)
    with torch.no_grad():
        eve_scores = np.stack(
            [
                eval_ELBO(model,(tr[i:i+1], s[i:i+1], p[i:i+1]), beta, n_samples = args.n_samples).detach().numpy() for i in range(tr.shape[0])
            ]
        )
    
    prefix, ext = os.path.splitext(args.ckpt)

    if args.output: 
        oufile = args.output
    else:
        oufile = prefix + "_evescore.txt"

    with open(oufile, "w") as f:
        f.write(f"# Input       : {args.in_trsp}\n")
        f.write(f"# model config: {args.model_config}\n")
        f.write(f"# model ckpt  : {args.ckpt}\n")
        f.write(f"# n_samples: {args.n_samples}\n")
        for score in eve_scores:
            f.write(str(score)+"\n")
            
    print(f"# Input       : {args.in_trsp}")
    print(f"# model config: {args.model_config}")
    print(f"# model ckpt  : {args.ckpt}")
    print(f"Wrote {oufile}.")

