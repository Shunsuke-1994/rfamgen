from Bio import AlignIO

def make_a2mfasta_from_stk(stkfile, target_id, outname, include_gap = False):
    """
    for detailed info about a2m format, see https://github.com/debbiemarkslab/EVmutation
    """
    alignments = AlignIO.read(stkfile, "stockholm")
    target_alignment = [aln for aln in alignments if aln.id == target_id][0]
    with open(outname, "w") as a2m:
        a2m.write(f">{target_alignment.id} {target_alignment.description}\n")
        target_seq = str(target_alignment.seq)
        target_seq = target_seq.replace("-", "")

        a2m.write(f"X{target_seq}\n")

        for aln in alignments:
            if not (aln is target_alignment):
                aln_seq = str(aln.seq)
                tar_seq = str(target_alignment.seq)
                # if not include_gap: seq_ = "".join([n for n, tar_n in zip(aln_seq, tar_seq) if tar_n != "-"])
                seq_ = "".join([n for n, tar_n in zip(aln_seq, tar_seq) if tar_n != "-"]).replace("-", "X")
                
                a2m.write(f">{aln.id} {aln.description}\n")
                a2m.write(f"X{seq_}\n")

    return outname

import pandas as pd 
import numpy as np 

def read_coupling_score(scorefile):
    with open(scorefile, "r") as c:
        df_c = pd.read_csv(c, sep = " ", header=None)[[0, 2, 5]]
    L = df_c[2].max()
    np_c = np.zeros([L, L])
    np_c[:,:] = np.nan
    for x,y,v in np.array(df_c):
        np_c[int(x)-1, int(y)-1] = v

    return np_c


if __name__ == "__main__":
    import os 
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type = str)
    parser.add_argument("-o", "--output", default = "", type = str)
    parser.add_argument("--include_gap", action = "store_true")
    parser.add_argument("--target_id", type = str)
    args = parser.parse_args()

    if args.output == "":
        outpath = os.path.splitext(args.input)[0] + ".a2m"
    else:
        outpath = args.output
    make_a2mfasta_from_stk(args.input, args.target_id, outpath, include_gap = args.include_gap)
    print(f"Wrote {outpath}")
