import sys
sys.path.append("src")
import grammar
from tqdm import tqdm
from Bio import SeqIO
from multiprocessing import Pool

def parse_seqss_by_grammar(triplet):
    """
    INPUT
        triplet: (record, max_derivlen, grammar_str)
    OUTPU
        parsable record
    """
    record, max_derivlen, grammar_str = triplet
    seq_ss = str(record.seq)
    seq = seq_ss[:int(len(seq_ss)/2)]
    ss  = seq_ss[int(len(seq_ss)/2):]

    try:
        derivation = grammar.make_derivation_from_seq_ss(seq, ss, grammar = grammar_str)
        if len(derivation[0]) <= max_derivlen:
            onehot = grammar.make_onehot_from_derivation(derivation, grammar = grammar_str, max_len= max_derivlen)
            return record
    except:
        pass

    # derivation = grammar.make_derivation_from_seq_ss(seq, ss, grammar = grammar_str)
    # if len(derivation[0]) <= max_derivlen:
    #     onehot = grammar.make_onehot_from_derivation(derivation, grammar = grammar_str, max_len= max_derivlen)
    #     return record


def extract_parsable_by_grammar(fasta_refolded, max_derivlen, grammar_str, cpu = 4):
    """
    INPUT
        fasta_refolded, max_derivlen, grammar_str)
    OUTPU
        fasta file containing G3 parsable (seq, ss)
    """

    # total arg in tqdm doesn't accept len(list(iterator))
    records = list(SeqIO.parse(open(fasta_refolded, "r"), "fasta"))
    with Pool(cpu) as p:
        tmp = p.imap(parse_seqss_by_grammar, [(record, max_derivlen, grammar_str) for record in records])
        records_parsable = list(tqdm(tmp, total= len(records)))
    records_parsable_woNone = [record for record in records_parsable if record is not None] # remove None object.
    print("Number of the parsable sequences\t: ", len(records_parsable_woNone))
    
    return records_parsable_woNone


if __name__ == "__main__":
    import argparse
    import os
    import util
    import re 

    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_seqss', default = '/users/sumishunsuke/desktop/RNA/genzyme/datasets/ForFigure2/RF00163/test/test_RF00163_unique_seed_removed_44ntTO48nt_refoldSScons_G3parsable.fa')
    # parser.add_argument('--max_derivlen', default = 96, type = int, help='Maximal deriv length, 2xmaxseqlen')
    parser.add_argument('--grammar', '-g', default = "g3", help='Parsing grammar')
    parser.add_argument('--cpu', default=4, type = int)
    args = parser.parse_args()

    pattern = r"\d+nt"
    min_derivlen, max_derivlen = [int(i.replace("nt", ""))*2 for i in re.findall(pattern, args.fasta_seqss)]

    records_parsable_woNone = extract_parsable_by_grammar(
        args.fasta_seqss,
        max_derivlen,
        util.gname2gstr(args.grammar),
        args.cpu)

    # SeqIO.write doesn't work for fasta recording seq and ss.
    fastaname_refolded_parsable = os.path.splitext(args.fasta_seqss)[0] + f"_{args.grammar}parsable.fa"
    with open(fastaname_refolded_parsable, "w") as f:
        for record in records_parsable_woNone:
            seq, ss = str(record.seq)[:len(record)//2], str(record.seq)[len(record)//2:]
            f.write(">"+str(record.id)+" "+str(record.description)+"\n")
            f.write(seq+"\n")
            f.write(ss+"\n")
        print("Wrote", fastaname_refolded_parsable)

    # write seqonly fasta file for CharVAE and CMVAE.
    fastaname_refolded_parsable_seqonly = os.path.splitext(args.fasta_seqss)[0] + f"_{args.grammar}parsable_seqonly.fa"
    with open(fastaname_refolded_parsable_seqonly, "w") as f:
        for record in records_parsable_woNone:
            seq, ss = str(record.seq)[:len(record)//2], str(record.seq)[len(record)//2:]
            f.write(">"+str(record.id)+" "+str(record.description)+"\n")
            f.write(seq+"\n")
        print("Wrote", fastaname_refolded_parsable_seqonly)
