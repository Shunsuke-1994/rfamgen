import sys
sys.path.append("./src")
import torch
import torch.nn as nn
from torch.distributions import Normal
import argparse
import preprocess
import grammar
from infernal_tools import CovarianceModel, make_deriv_dict_from_trsp
from multiprocessing import Pool

print("torch.cuda.is_available: ", torch.cuda.is_available())

def helper_sampling_CMVAE(params):
    """
    params: (cm_deriv_dict, tr, s, p)
    """
    softmax = nn.Softmax(dim = -2)
    cm_deriv_dict, tr, s, p = params
    out_dict = make_deriv_dict_from_trsp(cm_deriv_dict, (softmax(tr), softmax(s), softmax(p)))
    cm = CovarianceModel(out_dict)
    seq, _ = cm.cmemit(sample = False)[0]
    return seq

def sampling_CMVAE(model, cm_deriv_dict, Z_DIM, SAMPLE_SIZE_Z):

    dist = Normal(torch.tensor([0.0]*Z_DIM), torch.tensor([1.0]*Z_DIM))
    z_sampled = [dist.sample() for i in range(SAMPLE_SIZE_Z)]
    seq_sampled = []
    trsp_sampled = []
    for i in range(SAMPLE_SIZE_Z):
        # if i%10 == 0: 
        print(i)
        z = z_sampled[i].to(model.device)
        tr, s, p = model.decoder(z.unsqueeze(dim = 0))
        trsp_sampled.append([cm_deriv_dict, tr.detach().cpu(), s.detach().cpu(), p.detach().cpu()])
    
    seq_sampled = [helper_sampling_CMVAE(param) for param in trsp_sampled]
    return seq_sampled


if __name__ == "__main__":
    import argparse
    import sys 
    sys.path.append("src")
    from util import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str)
    parser.add_argument('--config', type = str)
    parser.add_argument('--ckpt', type = str)
    parser.add_argument('--cmfile', default = "", type = str)
    parser.add_argument('--outfasta', help='output fasta', type = str)
    parser.add_argument('--n_samples', help='sampling size', default = 1000, type = int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # print(cfg)

    from models.CMVAE import CovarianceModelVAE
    from infernal_tools import CMReader

    cmreader = CMReader(args.cmfile)
    print("Start loading cm dict. This process may take much time for long sequences.")
    cm_deriv_dict = cmreader.load_derivation_dict_from_cmfile()
    model = CovarianceModelVAE.build_from_config(args.config)
    model.load_model_from_ckpt(args.ckpt)
    model.to(model.device)
    # model.eval();

    seq_sampled = sampling_CMVAE(model, cm_deriv_dict, cfg["Z_DIM"], args.n_samples)


    with open(args.outfasta, "w") as f:
        for i, seq in enumerate(set(seq_sampled)):
            f.write(f">seq{str(i)}\n")
            f.write(f"{str(seq)}\n")
        