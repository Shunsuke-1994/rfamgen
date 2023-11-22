# utility for MI between alignment columns.
import RNA
import numpy as np
from collections import Counter
from multiprocessing import Pool

def _pairwise_MI(align_i_j):
    """
    helper function for calc_MItable
    """
    align, i, j = align_i_j
    aln_size = len(align)
    if i != j:
        col_i = align[:,i]
        col_j = align[:,j]
        pairs_count = Counter([i+j for i,j in zip(col_i, col_j)])
        total_count = sum(pairs_count.values())
        mi_ij = [(c/total_count)*(np.log2(c*total_count/(col_i.count(p[0])*col_j.count(p[1])))) for p, c in pairs_count.items()]
        return i, j, sum(mi_ij)
    else:
        # entropy
        col_i = align[:, i]
        nuc_count = Counter(col_i)
        mi_diag = [(c/aln_size)*(np.log2(c) - np.log2(aln_size) + 2) for nuc, c in nuc_count.items()]

        return i, i, sum(mi_diag)


def calc_MItable(align, bptable, cpu = 4):
    """
    lower triangle matrix of mutual informaion.
    Extract match columns by using "#=GC RF" annotation
    """
    _, *bp_index, = *bptable,
    paircol_ij = [(align, i, j-1) for i, j in enumerate(bp_index) if j!=0]

    with Pool(cpu) as p:
        i_j_mi_ij = p.map(_pairwise_MI, paircol_ij)
        
    mitable = np.zeros([align.get_alignment_length(), align.get_alignment_length()])
    for i,j,mi in i_j_mi_ij:
        mitable[i, j] = mi
        
    return mitable

def get_bptable(ss):
    """
    see: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/group__struct__utils__pair__table.html#ga792503f8b2c6783248e5c8b3d56e9148
    """
    pt = RNA.ptable_from_string(ss)  # RNA.ptable_from_string can recognize wuss notation {Aa, {}, <>, =, ~, etc}
    bptable = np.zeros([pt[0], pt[0]])
    for i,j in enumerate(pt[1:]):
        if j != 0: bptable[i,j-1] = 1
    return bptable

if __name__ == "__main__":
    print(get_bptable("(.{)}"))
