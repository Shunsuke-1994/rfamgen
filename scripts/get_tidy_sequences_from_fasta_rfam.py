# for rfam preprocessing

import os
from Bio import AlignIO, SeqIO
from Bio.Alphabet import generic_rna
from Bio.Seq import Seq
import sys
sys.path.append("./src")
import preprocess

def fetch_seed_sequence(path_to_Rfamseed, rfam_acc, output_dir):
    # fetch seed sequences from Rfam.seed
    # if not f"{rfam_acc}_seed_gapped.fa" in os.listdir(output_dir):
    order_of_rfam_of_interest = 0
    with open(path_to_Rfamseed, "r", encoding = "latin-1") as f:
        line = f.readline()
        while line:
            if line.startswith("#=GF AC"):
                id_ = line.split(" ")[-1].replace("\n", "")
                if id_ == rfam_acc:
                    break
                order_of_rfam_of_interest +=1
            line = f.readline()

    counter=0
    with open(path_to_Rfamseed, "r", encoding = "latin-1") as handle:
        for align in AlignIO.parse(handle, "stockholm", alphabet = generic_rna):
            if counter == order_of_rfam_of_interest:
                path_gapped = os.path.join(output_dir, f"{rfam_acc}_seed_gapped.fa")
                AlignIO.write(align, path_gapped, format = "fasta")
                # print(f"wrote {path_gapped}")
            counter+=1
    
    # remove gap from seed fasta
    with open(f"{path_gapped}", "r") as file:
        ungapped = []
        for record in SeqIO.parse(file, "fasta", alphabet = generic_rna):
            record.seq = Seq(str(record.seq).replace("-", ""))
            ungapped.append(record)
        path_ungapped = os.path.join(output_dir, f"{rfam_acc}_seed.fa")
        SeqIO.write(ungapped, path_ungapped, format = "fasta")

    return path_ungapped

def remove_seed_from_full(fasta_seed, fasta_full):
    """
    input full path.
    """

    seq_seed = []
    with open(fasta_seed, "r") as file:
        for record in SeqIO.parse(file, "fasta", alphabet=generic_rna):
            seq = str(record.seq).upper().replace("-", "").replace("T", "U")
            seq_seed.append(seq)
    seq_seed = set(seq_seed)

    seq_all = []
    records_seed_removed = []
    with open(fasta_full, "r") as full:
        for record in SeqIO.parse(full, "fasta", alphabet=generic_rna):
            seq = str(record.seq).upper().replace("-", "").replace("T", "U")
            seq_all.append(seq)
            if (not seq in seq_seed) and (set(seq) <= {'A', 'C', 'G', 'U', 'T'}):
                record.seq = Seq(seq)
                records_seed_removed.append(record)
    seq_all = set(seq_all)

    fastaname_seed_removed = os.path.splitext(fasta_full)[0] + "_seed_removed.fa"
    SeqIO.write(records_seed_removed, fastaname_seed_removed, "fasta")

    print("seed_seq\t\t:", len(seq_seed))
    print("all_seq \t\t:", len(seq_all))
    print("full_seq - seed_seq\t:", len(records_seed_removed))
    return fastaname_seed_removed

def main(rfam_acc, output_dir, path_to_Rfamseed, cpu = 2):
    #1. uniquenize 
    #2. remove seed sequences from full sequenes

    #1.get
    fastaname_uniqued = preprocess.uniquenize(os.path.join(output_dir, f"{rfam_acc}.fa"))

    #2. remove seed sequences from full sequenes
    ungapped_seed = fetch_seed_sequence(path_to_Rfamseed, rfam_acc, output_dir)
    fastaname_uniques_seed_removed = remove_seed_from_full(ungapped_seed, fastaname_uniqued)

    return fastaname_uniques_seed_removed


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_file', default='./datasets/Rfam.seed')
    parser.add_argument('--rfam', help='Rfam accession', default="RF01317")
    parser.add_argument('--output_dir', default = "datasets/ForFigure2")
    parser.add_argument('--cpu', default = 2, type = int)

    args = parser.parse_args()

    main(args.rfam, args.output_dir, args.seed_file, args.cpu)

