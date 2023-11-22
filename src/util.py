import time
import os
import h5py
import gzip
from Bio import AlignIO
import grammar
import yaml
from nltk import CFG
from torch import nn 
import torch
import requests, json

class Timer:
	"""A simple timer to use during training"""
	def __init__(self):
		self.time0 = time.time()

	def elapsed(self):
		time1 = time.time()
		elapsed = time1 - self.time0
		self.time0 = time1
		return elapsed

class AnnealKL:
	"""Anneal the KL for VAE based training"""
	def __init__(self, step=1e-3, rate=500):
		self.rate = rate
		self.step = step

	def alpha(self, update):
		n, _ = divmod(update, self.rate)
		return min(1., n*self.step)

def load_data(data_path):
	"""Returns the h5 dataset as numpy array"""
	f = h5py.File(data_path, 'r')
	return f['data'][:]

### prepare fasta file from stockholm file 
def load_stk(stk_file):
    """
    open gzipped stockholm file and load sequences.
    """
    _, ext = os.path.splitext(stk_file)
    if ext == ".gz":
        stk = gzip.open(stk_file, "rt")
    else:
        stk = open(stk_file, "r")

    align      = AlignIO.read(stk, "stockholm")
    stk_format = align.format("stockholm")
    id_list    = []
    seq_list   = []
    for record in align:
        id_list.append(record.id)
        seq_list.append(record.seq)
    stk.close()

    for line in stk_format.split("\n"):
        if line.startswith("#=GC RF"):
            RF      = line.split(" ")[2]
        elif line.startswith("#=GC SS_cons"):
            SS_cons = line.split(" ")[2]

    return RF, SS_cons, seq_list, id_list

def recover_seqss_from_stk(seq_aligned, SS_cons):
    """
    recover sequence and structure from WUSS anotation in stockholm file.
    for WUSS annotatin, see infernal documentation.
    """
    seq = ""
    ss  = ""
    for n, s in zip(seq_aligned, SS_cons):
        if n != "-":
            seq += n
            if s in "<{[":
                s = "("
            elif  s in ">}]":
                s = ")"
            elif s in "~-":
                s = "."
            ss += s
    return seq, ss

def clean_SScons(SS_cons):
    ss = ""
    for s in SS_cons:
        if    s in "<{[": s = "("
        elif  s in ">}]": s = ")"
        else: s = "."
        ss += s
    return ss

def write_fasta_from_stockholm(stk_gz_file, fasta):
    """
    write fasta file from stockholm file. fasta contains id, sequence and the concensus structure
    """
    _, SS_cons, seq_list, id_list = load_stk(stk_gz_file)

    with open(fasta,"w") as f:
        for seq_aligned, seq_id in zip(seq_list, id_list):
            seq, ss = recover_seqss_from_stk(seq_aligned, SS_cons)
            f.write(">"+seq_id + "\n")
            f.write(seq + "\n")
            f.write(ss + "\n")

    return print("Wrote ", fasta)

# def visualize_RNA(seq, ss):
#     bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{ss}\n{seq}')[0]
#     fvm.plot_rna(bg, lighten=0.7, backbone_kwargs={"linewidth":3})
#     return 

def gname2gstr(name):
    if name == "g3":
        grammar_str = grammar.grammar_g3
    elif name == "g4":
        grammar_str = grammar.grammar_g4
    elif name == "g5":
        grammar_str = grammar.grammar_g5
    elif name == "g6":
        grammar_str = grammar.grammar_g6
    elif name == "g6s":
        grammar_str = grammar.grammar_g6s
    elif name == "g7":
        grammar_str = grammar.grammar_g7
    elif name == "g8":
        grammar_str = grammar.grammar_g8
    elif name == "eq":
        grammar_str = grammar.grammar_eq
    elif name == "cm":
        grammar_str = grammar.grammar_CM
    else:
        raise Exception("No such a grammar.")

    return grammar_str

def gstr2gname(grammar_str):
    if grammar_str == grammar.grammar_g3:
        name = "g3" 
    elif grammar_str == grammar.grammar_g4:
        name = "g4"
    elif grammar_str == grammar.grammar_g5:
        name = "g5"
    elif grammar_str == grammar.grammar_g6:
        name = "g6"
    elif grammar_str == grammar.grammar_g6s:
        name = "g6s" 
    elif grammar_str == grammar.grammar_g7:
        name = "g7"
    elif grammar_str == grammar.grammar_g8:
        name = "g8"
    elif grammar_str == grammar.grammar_eq:
        name = "eq"
    elif grammar_str == grammar.grammar_CM:
        name = "cm"
    else:
        raise Exception("No such a grammar.")

    return name

def len_of_rule(grammar_str):
    cfg_obj = CFG.fromstring(grammar_str)
    rules   = cfg_obj.productions()
    return len(rules)

def get_derivation_shape(h5py_file):
    with h5py.File(h5py_file, "r") as f:
        data = f["data"][0]
        max_len = data.shape
    return max_len

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def continuous_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def logsumexp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

if __name__ == "__main__":
    print(len_of_rule(grammar.grammar_eq))
