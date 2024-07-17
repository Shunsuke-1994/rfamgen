from Bio import SeqIO, AlignIO
import subprocess
import re
import os
import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split

"""
preprocessing from Rfam to onehot exp.
1. cmalign
2. refold
3. make onehot by a grammar
4. save onehot 
"""

def uniquenize(fasta):
    d = dict()
    count = 0
    with open(fasta, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            count += 1
            d[str(record.seq)] = str(record.description)
    print("Total number\t\t: ", count)
    print("After uniquenize\t: ", len(d))

    basename, ext    = os.path.splitext(fasta)
    file_uniquenized = basename+"_unique"+ext
    with open(file_uniquenized, "w") as f:
        for k,v in d.items():
            f.write(">"+v+"\n")
            f.write(k+"\n")
    print(f"created {file_uniquenized}")
    return file_uniquenized

def cmalign(cmfile, seqfile, log = False, CYK = False, nonbanded = False, small = False, trunc = True, cpu = 2, suffix = ""): 
    """
    cmalign: cm & fasta -> stockholm
    AlignIO: stockholm -> stockholm(reformat)
    see: https://biopython.org/wiki/AlignIO
    changed CMALIGN_MAX_NSEQ parameter to 200000 from priginal number.
    """
    outname, _     = os.path.splitext(seqfile)
    outname        = outname+suffix
    outstk         = outname + ".sto"
    outstk_tmp     = outname + "_tmp.sto"
    score_file     = f"--sfile {outname}_score.txt" if log else ""
    traceback_file = f"--tfile {outname}_traceback.txt" if log else "" 
    insertion_file = f"--ifile {outname}_insertion.txt" if log else ""
    ELstate_file   = f"--elfile {outname}_ELstate.txt" if log else ""
    cyk            = "--cyk" if CYK else ""
    nonbanded      = "--nonbanded" if nonbanded else ""
    trunc          = "--notrunc" if not trunc else ""
    small          = "--small --notrunc --noprob" if small else ""
    cmd            = f"cmalign --cpu {cpu} {cyk} {nonbanded} {trunc} {small} {score_file} {traceback_file} {insertion_file} {ELstate_file} {cmfile} {seqfile} > {outstk_tmp}"
    print(cmd)
    proc = subprocess.run(cmd, shell=True, stdout = subprocess.PIPE, stderr = subprocess.PIPE) 

    # gzip compress and remove traceback file. 
    if log:
        try:
            cmd_gzip = f"gzip {outname}_traceback.txt"
            # cmd_rm_traceback = f"rm {outname}_traceback.txt"
            print(cmd_gzip)
            # print(cmd_rm_traceback)
            subprocess.run(cmd_gzip, shell=True)
            # subprocess.run(cmd_rm_traceback, shell=True)
        except:
            pass

    # reformat stockholm file by esl-reformat(1) and AlignIO(2)
    # 1. cmalign automatically reformat stockholm formet to pfam when using big data. Get back rfam format by esl-reformat in HMMER.
    cmd_esl_reformat = f"esl-reformat --informat stockholm -o {outstk} stockholm {outstk_tmp}"
    print(cmd_esl_reformat)
    subprocess.run(cmd_esl_reformat, shell = True)
    subprocess.run(f"rm {outstk_tmp}", shell = True)
    
    # 2. output stockholm from cmalign contains dots. This is not recognized by refold. remove by AlignIO.
    alignment = AlignIO.read(outstk, "stockholm")
    # Capitalize the characters. RNAfold doesn't consider lower characters!!!
    for record in alignment:
        record.seq = str(record.seq).upper()
    AlignIO.write(alignment, outstk, "stockholm")
    print("Reformated the stockhoml file by Bio.AlignIO.")

    return print(proc.stdout.decode("utf8"))

def refold(aln_file, n_turn = 2):
    """
    refold sequences from aligned sequences.
    1. run RNAalifold -f S --SS_cons
    2. run refold.pl --turn 2
    3. RNAfold -C --enforceConstraint
    ref: https://www.tbi.univie.ac.at/RNA/refold.1.html
    """

    # assert ".fa" in outfasta or ".fasta" in outfasta, "outfile should be fasta file"
    basename, _          = os.path.splitext(aln_file)
    alifold_file         = basename + ".alifold"
    constraint_file      = basename + "_constraint.fa"
    outfasta             = basename + "_refold.fa"
    
    cmd1                 = f"RNAalifold -f S --noPS --SS_cons {aln_file} > {alifold_file}"
    cmd2                 = f"$HOME/miniconda3/envs/genzyme/share/ViennaRNA/bin/refold.pl −−turn {n_turn} {aln_file} {alifold_file} > {constraint_file}"
    cmd3                 = f"RNAfold --noPS -C --enforceConstraint {constraint_file} >{outfasta}"
    
    subprocess.run(cmd1, shell=True)
    subprocess.run(cmd2, shell=True)
    subprocess.run(cmd3, shell=True)

    print(cmd1)
    print(cmd2)
    print(cmd3)
    return [cmd1, cmd2, cmd3]

# another way to generate 2nd strcuture from an alignment file
def normalize_SS_cons(ss):
    norm_ss = ""
    for ss_n in ss:
        if ss_n == "<":
            norm_ss += "("
        elif ss_n == ">":
            norm_ss += ")"
        elif ss_n in ["(", ")"]:
            norm_ss += ss_n
        else:
            norm_ss += "."
    return norm_ss

def fold_by_SS_cons(alnseq, normalized_SS_cons):
    alnseq = alnseq.upper()
    deque_5bra = deque()
    seq = []
    ss   = []
    for n, (alnseq_n, alnss_n) in enumerate(zip(alnseq, normalized_SS_cons)):
            
        if alnss_n == "(":
            deque_5bra.append([n, alnseq_n])
            seq_n, ss_n = alnseq_n, "("

        elif alnss_n == ")":
            counter_n, counter_alnseq_n = deque_5bra.pop()
            # both nuc is not deleted.
            if not "-" in [alnseq_n, counter_alnseq_n]:
                # paired
                if {alnseq_n, counter_alnseq_n} in [{"A", "U"}, {"U", "G"}, {"G","C"}]:
                    seq_n, ss_n = alnseq_n, ")"
                # unpaired
                else:
                    seq_n, ss_n = alnseq_n, "."
                    ss[counter_n] = "."
                    
            # this nuc is  deleted
            elif alnseq_n == "-" and counter_alnseq_n != "-":
                seq_n, ss_n = "-", " "
                ss[counter_n] = "."
                
            # the counter nuc is deleted
            elif alnseq_n != "-" and counter_alnseq_n == "-":
                seq_n, ss_n = alnseq_n, "."
                ss[counter_n] = " "
            
            # both is deleted
            else:
                seq_n, ss_n = "-", " "
                ss[counter_n] = " "

        # not deleted and single strand
        elif alnseq_n != "-" and alnss_n == "." :
            seq_n, ss_n = alnseq_n, "."
        
        # deletion and single strand
        else:
            seq_n, ss_n = "-", " "
            
        seq+= seq_n
        ss  += ss_n
    return "".join(seq).replace("-", ""), "".join(ss).replace(" ", "")

def split_fasta_train_test_valid(fasta, train_size = 0.7):
    """
    split fasta file to train_ratio : (1-train_ratio)/2 : (1-train_ratio)/2.
    and save at save_dir.
    """

    dict_seq = dict()
    with open(fasta, "r") as f:
        for seq_record in SeqIO.parse(f, "fasta"):
            if not str(seq_record.id) in dict_seq:
                dict_seq[str(seq_record.id)] = str(seq_record.seq)

    ID_train, ID_test_valid = train_test_split(list(dict_seq.keys()), train_size=train_size)
    ID_test, ID_valid       = train_test_split(ID_test_valid, train_size=0.5)
    dir_name, filename      = os.path.split(fasta)
    train_file              = os.path.join(dir_name, "train", "train_"+filename)
    test_file               = os.path.join(dir_name, "test", "test_"+filename)
    valid_file              = os.path.join(dir_name, "valid", "valid_"+filename)

    with open(fasta, "r") as f:
        with open(train_file, "w") as train, open(test_file, "w") as test, open(valid_file, "w") as valid:
            for record in SeqIO.parse(f, "fasta"):
                if str(record.id) in ID_train:
                    train.write(">" + str(record.description) + "\n")
                    train.write(str(record.seq)[:int(len(record.seq)/2)]+"\n")
                elif str(record.id) in ID_test:
                    test.write(">" + str(record.description) + "\n")
                    test.write(str(record.seq)[:int(len(record.seq)/2)]+"\n")
                else:
                    valid.write(">" + str(record.description) + "\n")
                    valid.write(str(record.seq)[:int(len(record.seq)/2)]+"\n")

    return print(f"split {fasta} to {train_file} and {test_file} and {valid_file}")


def load_seq_and_ss_from_alnfasta(fasta):
    dict_seq = dict()
    with open(fasta, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            line            = re.sub(r"\(.*\d+\.+\d+\)", "", str(record.seq)) #remove energy information ( dd.dd)
            aln_seq, aln_ss = line[:len(line)//2], line[len(line)//2:]
            seq, ss         = "", ""
            for seq_n, ss_n in zip(aln_seq, aln_ss):
                if seq_n != ".":
                    seq += seq_n
                    ss  += ss_n
            dict_seq[record.id] = (seq, ss) #

    return dict_seq


### preproc for CVAE

dna_vocab = {"A":0, "T":1, "G":2, "C":3}
rna_vocab = {"A":0, "U":1, "G":2, "C":3}
dna_gap_vocab = {"A":0, "T":1, "G":2, "C":3, "-":4}
rna_gap_vocab = {"A":0, "U":1, "G":2, "C":3, "-":4}


def onehot_encode(sequence, vocab_type="dna", max_len = 50):
    sequence = sequence + "*"*(max_len - len(sequence))
    encoded = []

    if vocab_type == "dna":
        vocab = dna_vocab
    elif vocab_type == "rna":
        vocab = rna_vocab
    elif vocab_type == "dna_gap":
        vocab = dna_gap_vocab
    elif vocab_type == "rna_gap":
        vocab = rna_gap_vocab
    else:
        print("No vocab")

    for nucleotide in sequence:
        onehot = [0]*len(vocab)
        onehot[vocab[nucleotide]] = 1
        encoded.append(onehot)
    return encoded

def onehot_decode(encode, vocab_type = "dna", sample = False):
    """decode sequence from logits"""
    sequence = ""

    if vocab_type == "dna":
        rev_vocab = {v:k for k,v in dna_vocab.items()}
    elif vocab_type == "rna":
        rev_vocab = {v:k for k,v in rna_vocab.items()}
    elif vocab_type == "dna_gap":
        rev_vocab = {v:k for k,v in dna_gap_vocab.items()}
    elif vocab_type == "rna_gap":
        rev_vocab = {v:k for k,v in rna_gap_vocab.items()}
    else:
        print("No vocab")

    for onehot in encode:
        if sample:
            prob = np.exp(onehot)/np.sum(np.exp(onehot))
            nuc_index = np.random.choice(range(len(rev_vocab)), p=prob)
        else:
            nuc_index = np.argmax(onehot)
        sequence += rev_vocab[nuc_index]
    return sequence


if __name__ == "__main__":
    onehot = [[-1, 1, 0.25, 1, 1] for i in range(10)]
    print("Sampled :", onehot_decode(onehot, sample = True, vocab_type="dna_gap"))
    print("Argmax  :", onehot_decode(onehot, sample = False, vocab_type="dna_gap"))
