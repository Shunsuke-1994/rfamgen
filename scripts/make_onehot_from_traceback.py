import os
import sys 
sys.path.append("src")
from infernal_tools import TracebackFileReader, make_trsp_from_deriv_dict
import preprocess
import gzip

def wrapper_of_make_trsp_from_deriv_dict(doublet):
    path_to_cmfile, deriv_dict_cminit = doublet
    tr, ss, bp = make_trsp_from_deriv_dict(path_to_cmfile, deriv_dict_cminit)
    return tr, ss, bp


# memory saving code. slow...
def make_onehot_of_cm_from_traceback(path_to_traceback, path_to_cmfile):

    id_all = []
    with gzip.open(path_to_traceback, "rb") as tb:
        for line in tb:
            if line.startswith(b'>'):
                id_all.append(line.replace(b">", b"").decode('utf-8'))
    
    n_size         = len(id_all)

    print(f"Start reading {path_to_traceback}...")
    tbreader       = TracebackFileReader(path_to_cmfile, path_to_traceback)
    tbtext_all     = [list(tbreader.traceback_iter())[0] for i in range(n_size)]
    print(f"Start making tbtext")
    tbdf_all       = [tbreader._make_tbdf_from_tbtext(tbtext) for tbtext in tbtext_all]
    print(f"Start making tbdict")
    deriv_dict_all = [tbreader.make_aligned_tbdict_from_tbdf_ELinitCM(tbdf) for tbdf in tbdf_all]

    # infer data shape
    tr0, ss0, bp0 = wrapper_of_make_trsp_from_deriv_dict((path_to_cmfile, deriv_dict_all[0]))

    # writing datafile
    output_h5 = path_to_traceback.replace(".txt.gz", "") + f"_onehot_cm.h5"
    print(f"Start writing {output_h5}...")
    with h5py.File(output_h5, 'w') as datafile:
        datafile.create_dataset('id', (n_size, ), dtype=h5py.special_dtype(vlen=str))
        datafile.create_dataset('tr', (n_size, *tr0.shape))
        datafile.create_dataset('s', (n_size, *ss0.shape)) # 最後に転置
        datafile.create_dataset('p', (n_size, *bp0.shape)) # 最後に転置

        for i, deriv_dict_cminit in enumerate(deriv_dict_all):
            tr, ss, bp = wrapper_of_make_trsp_from_deriv_dict((path_to_cmfile, deriv_dict_cminit))
            datafile["tr"][i] = tr
            datafile["s"][i]  = ss
            datafile["p"][i]  = bp
            if i%100 == 0:
                print(f"seq {str(i)}")

    return output_h5


if __name__ == '__main__':
    import argparse
    import h5py
    import numpy as np
    import sys 
    sys.path.append("src")
    import preprocess

    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', default = "", help='path to fasta file. Fasta file is automatically aligned to cmfile and its traceback will be converted to onehot.')
    parser.add_argument('--traceback', default = "", help='path to gzipped tracebackfile')
    parser.add_argument('--cmfile', default = "", required = True, help='path to cm file')
    parser.add_argument('--cpu', default=4, type = int, help = "CPU cores for cmalign program. (default: 4)")
    args = parser.parse_args()

    if args.fasta != "":
        preprocess.cmalign(
            cmfile = args.cmfile,
            seqfile = args.fasta,
            log = True,
            trunc = False, 
            suffix = "_notrunc",
            cpu = args.cpu) # --notruncation for dataset preprocessing.
        basename, _  = os.path.splitext(args.fasta)
        path_to_traceback = basename + "_notrunc_traceback.txt.gz"
    else:
        path_to_traceback = args.traceback

    print(f"Loading {path_to_traceback}.")
    output_h5 = make_onehot_of_cm_from_traceback(path_to_traceback, args.cmfile)
    print(f"wrote\t\t: {output_h5}")
