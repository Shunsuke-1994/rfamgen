# -*- coding: utf-8 -*-

import torch
import nltk
import numpy as np
from nltk import CFG, Nonterminal, Tree
from stack import Stack
from collections import deque
from torch.distributions import Categorical

# Example grammar for equation.
grammar_eq = """S -> S '+' T
S -> S '*' T
S -> S '/' T
S -> T
T -> '(' S ')'
T -> 'sin(' S ')'
T -> 'exp(' S ')'
T -> 'x'
T -> '1'
T -> '2'
T -> '3'
Nothing -> None"""


# the following grammars are from https://pubmed.ncbi.nlm.nih.gov/15180907/

grammar_g3 = """
S -> 'A(' S 'U)' | 'U(' S 'A)' | 'G(' S 'C)' | 'C(' S 'G)' | 'G(' S 'U)' | 'U(' S 'G)' | n L | R n | L S
L -> 'A(' S 'U)' | 'U(' S 'A)' | 'G(' S 'C)' | 'C(' S 'G)' | 'G(' S 'U)' | 'U(' S 'G)' | n L
R -> R n | 
n -> 'A.' | 'U.' | 'G.' | 'C.'
Nothing -> None
"""

grammar_g4 = """
S -> n S | T | 
T -> T n | 'A(' S 'U)' | 'U(' S 'A)' | 'G(' S 'C)' | 'C(' S 'G)' | 'G(' S 'U)' | 'U(' S 'G)' | T 'A(' S 'U)' | T 'U(' S 'A)' | T 'G(' S 'C)' | T 'C(' S 'G)' | T 'G(' S 'U)' | T 'U(' S 'G)' 
n -> 'A.' | 'U.' | 'G.' | 'C.'
Nothing -> None
"""

grammar_g5 = """
S -> 'A.' S | 'U.' S | 'G.' S | 'C.' S | 'A(' S 'U)' S | 'U(' S 'A)' S | 'G(' S 'C)' S | 'C(' S 'G)' S | 'G(' S 'U)' S | 'U(' S 'G)' S |
Nothing -> None
"""

grammar_g6 ="""
S -> L S | L
L -> 'A(' F 'U)' | 'U(' F 'A)' | 'G(' F 'C)' | 'C(' F 'G)' | 'G(' F 'U)' | 'U(' F 'G)' | 'A.' | 'U.' | 'G.' | 'C.'
F -> 'A(' F 'U)' | 'U(' F 'A)' | 'G(' F 'C)' | 'C(' F 'G)' | 'G(' F 'U)' | 'U(' F 'G)' | L S 
Nothing -> None
"""

grammar_g6s ="""
S -> L S | L
L -> 'A(' F_au 'U)' | 'U(' F_ua 'A)' | 'G(' F_gc 'C)' | 'C(' F_cg 'G)' | 'G(' F_gu 'U)' | 'U(' F_ug 'G)' | 'A.' | 'U.' | 'G.' | 'C.'
F_au -> 'A(' F_au 'U)' | 'U(' F_ua 'A)' | 'G(' F_gc 'C)' | 'C(' F_cg 'G)' | 'G(' F_gu 'U)' | 'U(' F_ug 'G)' | L S
F_ua -> 'A(' F_au 'U)' | 'U(' F_ua 'A)' | 'G(' F_gc 'C)' | 'C(' F_cg 'G)' | 'G(' F_gu 'U)' | 'U(' F_ug 'G)' | L S 
F_gc -> 'A(' F_au 'U)' | 'U(' F_ua 'A)' | 'G(' F_gc 'C)' | 'C(' F_cg 'G)' | 'G(' F_gu 'U)' | 'U(' F_ug 'G)' | L S 
F_cg -> 'A(' F_au 'U)' | 'U(' F_ua 'A)' | 'G(' F_gc 'C)' | 'C(' F_cg 'G)' | 'G(' F_gu 'U)' | 'U(' F_ug 'G)' | L S 
F_gu -> 'A(' F_au 'U)' | 'U(' F_ua 'A)' | 'G(' F_gc 'C)' | 'C(' F_cg 'G)' | 'G(' F_gu 'U)' | 'U(' F_ug 'G)' | L S 
F_ug -> 'A(' F_au 'U)' | 'U(' F_ua 'A)' | 'G(' F_gc 'C)' | 'C(' F_cg 'G)' | 'G(' F_gu 'U)' | 'U(' F_ug 'G)' | L S 
Nothing -> None
"""

# G3 + stacking
# this doesnt allow lone pairs
grammar_g7 ="""
S -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | n L | R n | L S
L -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | n L
R -> R n | 
P_au -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_ua -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_gc -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_cg -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_gu -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_ug -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
M -> n L | R n | L S
n -> 'A.' | 'U.' | 'G.' | 'C.'
Nothing -> None
"""
# G4 + stacking
# this doesnt allow lone pairs
grammar_g8 ="""
S -> n S | T | 
T -> T n | 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | T 'A(' P_au 'U)' | T 'U(' P_ua 'A)' | T 'G(' P_gc 'C)' | T 'C(' P_cg 'G)' | T 'G(' P_gu 'U)' | T 'U(' P_ug 'G)' 
P_au -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_ua -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_gc -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_cg -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_gu -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
P_ug -> 'A(' P_au 'U)' | 'U(' P_ua 'A)' | 'G(' P_gc 'C)' | 'C(' P_cg 'G)' | 'G(' P_gu 'U)' | 'U(' P_ug 'G)' | 'A(' M 'U)' | 'U(' M 'A)' | 'G(' M 'C)' | 'C(' M 'G)' | 'G(' M 'U)' | 'U(' M 'G)'
M -> n S | T n | T 'A(' P_au 'U)' | T 'U(' P_ua 'A)' | T 'G(' P_gc 'C)' | T 'C(' P_cg 'G)' | T 'G(' P_gu 'U)' | T 'U(' P_ug 'G)'
n -> 'A.' | 'U.' | 'G.' | 'C.'
Nothing -> None
"""

grammar_g1CM ="""
S -> n '(' S n ')' | n '.' S |  S '.' n | S S | 
n -> 'A' | 'G' | 'U' | 'C' | '-' 
"""

grammar_g5CM ="""
S -> n '.' S | n '(' S n ')' S |
n -> 'A' | 'G' | 'U' | 'C' | '-' 
"""
grammar_CM = """
S -> IL | IR | ML | MP | MR | D | B
IL -> n '.' IL | n '.' IR | n '.' ML | n '.' MP | n '.' MR | n '.' D | n '.' E  | n '.' B
IR -> IR '.' n | ML '.' n | MP '.' n | MR '.' n | D '.' n | E '.' n | B '.' n
ML -> n '.' IL | n '.' IR | n '.' ML | n '.' MP | n '.' MR | n '.' D | n '.' E | n '.' B
MP -> nl '(' IL ')' nr | nl '(' IR ')' nr | nl '(' ML ')' nr | nl '(' MP ')' nr | nl '(' MR ')' nr | nl '(' D ')' nr | nl '(' E ')' nr | nl '(' B ')' nr
MR -> IL '.' n | IR '.' n | ML '.' n | MP '.' n | MR '.' n | D '.' n | E '.' n | B '.' n
D -> IL | IR | ML | MP | MR | D | E | B
B -> S S 
E -> 
n -> 'A' | 'U' | 'G' | 'C'
nl_nr -> 'AA' | 'AC' | 'AG' | 'AU' | 'CA' | 'CC' | 'CG' | 'CU' | 'GA' | 'GC' | 'GG' | 'GU' | 'UA' | 'UC' | 'UG' | 'UU'
"""

# RNAG = grammar_g3
# GCFG = CFG.fromstring(RNAG)
# parser = nltk.ChartParser(GCFG)

S = Nonterminal('S')
L = Nonterminal('L')
F = Nonterminal('F')

def get_mask(nonterminal, GCF_obj, as_variable=False):
    if isinstance(nonterminal, Nonterminal):
        mask = [rule.lhs() == nonterminal for rule in GCF_obj.productions()]
        mask = torch.FloatTensor(mask) if as_variable else mask
        return mask
    else:
        raise ValueError('Input must be instance of nltk.Nonterminal')

### generate sequence and structure from derivation(model output) 
def make_derivation_from_logits(logits, grammar, sample=False, max_len=100):
    """Generate a valid expression from logits. copy of generate method in model class."""
    GCFG = CFG.fromstring(grammar)
    stack = Stack(CFG_obj=GCFG, start_symbol=S)

    rules = []
    t = 0
    while stack.nonempty:
        alpha = stack.pop()
        mask = get_mask(alpha, GCFG, as_variable=True)
        probs = mask * logits[t].exp()
        probs = probs / probs.sum()
        # print(probs)
        if sample:
            m = Categorical(probs)
            i = m.sample()
        else:
            _, i = probs.max(-1) # argmax
        # convert PyTorch Variable to regular integer
        # print(i)
        i = i.item()
        # select rule i
        rule = GCFG.productions()[i]
        rules.append(rule)
        # add rhs nonterminals to stack in reversed order
        for symbol in reversed(rule.rhs()):
            if isinstance(symbol, Nonterminal):
                stack.push(symbol)
        t += 1
        if t == max_len:
            break

    return rules

### make onehot expression from sequence and structure. 
def tokenize(seq, ss, mode = "normal"):
    """
    tokenize RNA and its structure
    mode = "normal" or "cm". Token changes depending on CM or not.
    """
    if mode == "normal":
        token = [seq_n+ss_n for seq_n, ss_n in zip(seq, ss)]
    elif mode == "cm":
        token = []
        for i in range(len(seq)):
            token.append(seq[i])
            token.append(ss[i])
    return token

def make_derivation_from_seq_ss(seq, ss, grammar, mode = "normal"):
    """
    input: seq, ss
    output: derivation, onehot
    """
    token = tokenize(seq, ss, mode = mode)
    parser = nltk.ChartParser(CFG.fromstring(grammar))
    parse_trees = parser.parse(token)
    derivation = [tree.productions() for tree in parse_trees]
    assert len(derivation) == 1, "derivation is strange!"
    return derivation

def make_onehot_from_derivation(derivation, grammar, max_len = 100):
    """
    input: derivation, maxlen
    output: onehot
    """
    prod_map = {}
    GCFG = CFG.fromstring(grammar)
    for ix, prod in enumerate(GCFG.productions()):
        prod_map[prod] = ix
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in derivation]

    n_char = len(GCFG.productions()) 
    one_hot = np.zeros((len(indices), max_len, n_char), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1. # 1 ~ num_productions まではきちんと割り当て
        one_hot[i][np.arange(num_productions, max_len),-1] = 1. # 残りのnum_productions ~ MAX_LENまでは最後に割り当て
    one_hot = torch.from_numpy(one_hot)
    return one_hot

def make_nltk_tree_from_derivation(derivation, print_error = False):
    start_node = derivation[0].lhs()
    derivation = deque(derivation)
    rules = []
    def make_tree_from_node(node):
        # dfs implementation
        try:
            rule = derivation.popleft()
            rules.append(rule)
            while not rule.lhs() == node:
                rule = derivation.popleft()
            return Tree(node, [make_tree_from_node(next_node) if isinstance(next_node, Nonterminal) else next_node for next_node in rule.rhs()])
        except:
            # もし再構成できなかったらそれまでのrulesを返す
            if print_error:
                print(rules)
    return make_tree_from_node(start_node)

def make_RNA_and_SS_from_derivation(derivation):
    tree      = make_nltk_tree_from_derivation(derivation)
    RNAstruct = "".join([l for l in tree.leaves() if l != None]) # noneがあるとjoinできない.
    seq, ss   = RNAstruct[0::2], RNAstruct[1::2]
    return seq, ss


if __name__ == '__main__':
    ex1_seq = "G--GU-GG"
    ex1_ss  = "((..)())"
    derivation = make_derivation_from_seq_ss(ex1_seq, ex1_ss, grammar_g5CM, mode = "cm")
    print(tokenize(ex1_seq, ex1_ss))
    print(derivation[0]) # [S -> n '(' S n ')' S, n -> 'G', S -> n '(' S n ')' S, n -> '-', S -> n '.' S, n -> '-', S -> n '.' S, n -> 'G', S -> , n -> 'U', S -> n '(' S n ')' S, n -> '-', S -> , n -> 'G', S -> , n -> 'G', S -> ]

