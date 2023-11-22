from Bio import SeqIO
import re 

def get_unique_valid_novel(fasta_train, fasta_gen, score_file, GA_THRESHOLD):
    """
    unique > valid > novel
    """
    # get unique and valid
    seqid_to_bitscore = score_parser(score_file)
    valid = {k:v for k,v in seqid_to_bitscore.items() if float(v)>GA_THRESHOLD}    
    
    # get novel
    seq_train = set()
    with open(fasta_train, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            seq = str(record.seq).replace("T", "U")
            if not seq in seq_train:
                seq_train.update(set([seq]))

    
    seq_gen = dict()
    with open(fasta_gen, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            seq = str(record.seq).replace("T", "U")
            id_ = str(record.id)
            if not seq in seq_gen:
                seq_gen[id_] = seq

    valid_seq = set([v for k,v in seq_gen.items() if k in valid])
    novel = valid_seq - seq_train
    return seqid_to_bitscore, valid, novel

def score_parser(score_file):
    """
    score file (cmalign output) -> {seqid:bitscore}
    """
    seqid_to_bitscore = dict()
    with open(score_file, "r") as f:
        seqid_to_bitscore = dict()
        for line in f.readlines():
            if not line.startswith("#"):
                seqid     = re.sub(r"\s+", ",", line).split(",")[2]
                bit_score = re.sub(r"\s+", ",", line).split(",")[7]
                seqid_to_bitscore[seqid] = float(bit_score)

    return seqid_to_bitscore