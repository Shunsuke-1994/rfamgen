import sys 
sys.path.append("src")
import util
import random 
random.seed(42)
import RNA
from metric_helper.alignment_mi_metric import calc_MItable
from Bio import AlignIO, Align
from Bio.Alphabet import generic_rna


def calc_information_content(stockholm, cpu, n_samples = float("inf")):
    """
    get mutual information according to refseq/SS_cons
    """
    align   = AlignIO.read(stockholm, "stockholm", alphabet=generic_rna)
    if n_samples < len(align):
        print(f"sampling {n_samples} records from {stockholm}")
        align = Align.MultipleSeqAlignment(
            random.sample(list(align), int(n_samples)),
            column_annotations = align.column_annotations
        )

    # calculate seq entropy based on #GC RF 
    unmasked_col = [i if n in "AUCGT" else -1 for i,n in enumerate(align.column_annotations["reference_annotation"].upper())]
    bptable_1st = [len(unmasked_col)]
    bptable_1st += [i+1 for i in unmasked_col]

    # calculate 2/3 deg information based on #GC SS_cons
    _, ss_cons, _, _ = util.load_stk(stockholm)
    bptable_2nd = RNA.pt_pk_remove(RNA.ptable_from_string(ss_cons))
    bptable_all = RNA.ptable_from_string(ss_cons)

    mi_1st = calc_MItable(align, bptable_1st, cpu = cpu)
    info_1st = mi_1st.sum()
    mi_2nd = calc_MItable(align, bptable_2nd, cpu = cpu)
    info_2nd = mi_2nd.sum()
    mi_all = calc_MItable(align, bptable_all, cpu = cpu)
    info_all = mi_all.sum()
    info_3rd = info_all - info_2nd
    
    return ss_cons, mi_1st+mi_all, info_1st, info_2nd, info_3rd

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stockholm', '-i')
    parser.add_argument('--cpu', default = 4, type = int)
    parser.add_argument('--n_samples', default = "", type = str)

    args = parser.parse_args()
    n_samples = float("inf") if args.n_samples == "" else int(args.n_samples)
    
    _, _, info_1st, info_2nd, info_3rd = calc_information_content(args.stockholm, args.cpu, n_samples = n_samples)

    print(f"{args.stockholm},{info_1st:.3f},{info_2nd:.3f},{info_3rd:.3f}")

