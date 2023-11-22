import re 
from nltk import CFG, Nonterminal, Production
from pprint import pprint
from collections import OrderedDict, deque
import grammar
import gzip
import pandas as pd
import copy 
import torch
import numpy as np


class CovarianceModel:
    def __init__(self, deriv_dict):
        self.deriv_dict = deriv_dict
        # remove node info
        self._trans_prob_dict = {}
        for k,v in self.deriv_dict.items():
            self._trans_prob_dict.update(v)

    def cmemit(self, n=1, sample = False):
        outputs = []
        for _ in range(n):
            seq_ss        = "S_0"
            current_state = "S_0"
            # When "E-> ",
            # 1: next node == END  ---> next_state = None and break
            # 2: next node == BEGL ---> pop next_state from bif_stack
            self._bif_stack = deque()
            ins_loop_counter = 0
            break_ins = False
            while current_state != None:
                rhs, next_state, probs_trans = self._fetch_rule_and_next(current_state, sample, break_ins = break_ins)
                break_ins = False
                if current_state == next_state:
                    ins_loop_counter += 1
                    if ins_loop_counter == int(1/(1-probs_trans)):
                        break_ins = True                        
                        ins_loop_counter = 0
                seq_ss          = seq_ss.replace(current_state, rhs, 1)
                current_state   = next_state
                
            seq_ss = re.sub(r"(\s|-)", "", seq_ss)
            outputs.append((seq_ss[0::2], seq_ss[1::2]))
        return outputs
    
    def _fetch_rule_and_next(self, current_state, sample, break_ins = False):
        """
        current_state : Lefthand state.
        sample        : How to determine a next state. True means random. 
        break_ins     : Flag of escapeing from endless looping of an insertion state. If True, break from while. 
        """
        trans_emit_from_current_node = self._trans_prob_dict[current_state]
        child_states, probs_trans    = list(trans_emit_from_current_node["trans"].keys()), list(trans_emit_from_current_node["trans"].values())
        
        # 1 or 2, no child state
        if child_states == []:
            next_state = self._bif_stack.pop() if len(self._bif_stack) != 0 else None
            return "--", next_state, 0
        
        # child states exist. determine the right hand side(rhs).
        else:
            prob_trans_sum = sum(probs_trans)
            # determine next state
            if sample:
                next_state = np.random.choice(child_states, 1, p = np.array(probs_trans)/prob_trans_sum)[0]
            
            else: # fetch in a deterministic way
                if break_ins: # escape from ins loop
                    prob_2nd = sorted(probs_trans, reverse=True)[1] #2nd max
                    next_state = child_states[probs_trans.index(prob_2nd)] #

                    if current_state == next_state: #exception in case that several top probs
                        next_state = child_states[probs_trans.index(prob_2nd)+1]
                else:
                    next_state = child_states[np.array(probs_trans).argmax()]
            
            # processing next state,emission and the following processing
            if re.match(r"(D|S)_.*", current_state):  # deletion | start
                rhs             = " ".join([next_state])
            elif re.match(r"B_.*", current_state):  # bif
                child_states = list(self._trans_prob_dict[current_state]["trans"].keys())
                for bif in child_states[::-1]:
                    self._bif_stack.append(bif)
                rhs        = " ".join([child_state for child_state in child_states])
                next_state = self._bif_stack.pop()
            else: 
                emit_nucs, probs_emit = list(trans_emit_from_current_node["emit"].keys()), list(trans_emit_from_current_node["emit"].values())
                if sample:
                    emit = np.random.choice(emit_nucs, 1, p = np.array(probs_emit)/sum(probs_emit))[0]
                else:
                    emit = emit_nucs[np.array(probs_emit).argmax()]
                    
                if re.match(r".*R_", current_state):   rhs = " ".join([next_state, emit, "."])
                elif re.match(r".*L_", current_state): rhs = " ".join([emit, ".", next_state])
                elif re.match(r".*P_", current_state): rhs = " ".join([emit[0], "(", next_state, emit[1], ")"])
                else: Exception(f"Unrecognized state: {current_state}")

            return rhs, next_state, max(probs_trans)

    def cmeval(self, seq):
        """
        inside algorithm of covariance model.
        should i name "forward"?
        see: "Biological Sequence Analysis", p286
        TODO: use logsumexp and scaling
        """
        L = len(seq)
        M = len(self._trans_prob_dict)
        a = torch.zeros(M, L+2, L+2)

        state2index = self._get_state2index()
        DeltaL = {"P":1, "L":1, "D":0, "S":0, "B":0, "E":0, "R":0,}
        DeltaR = {"P":1, "R":1, "D":0, "S":0, "B":0, "E":0, "L":0,}

        # initialization
        for j in range(0, L+1):
            for state in list(self._trans_prob_dict.keys())[::-1]:
                if self._get_statetype(state) == "E":
                    a[state2index[state]][j+1, j] = 1

                elif self._get_statetype(state) in "SD":
                    for child_state, trans_prob in self._trans_prob_dict[state]["trans"].items():
                        a[state2index[state]][j+1, j] += trans_prob * a[state2index[child_state]][j+1, j]

                elif self._get_statetype(state) == "B":
                    child_bif1, child_bif2 = self._trans_prob_dict[state]["trans"].keys()
                    a[state2index[state]][j+1, j] = a[state2index[child_bif1]][j+1, j] * a[state2index[child_bif2]][j+1, j]
                
                elif self._get_statetype(state) in "PRL":
                    pass

        # recursion
        for j in range(1, L+1):
            for i in range(j, 0, -1):
                for state in list(self._trans_prob_dict.keys())[::-1]:
                    if   self._get_statetype(state) == "E": pass

                    elif self._get_statetype(state) == "P" and i == j: pass

                    elif self._get_statetype(state) == "B":
                        child_bif1, child_bif2 = self._trans_prob_dict[state]["trans"].keys()
                        for k in range(i-1, j+1):
                            a[state2index[state]][i][j] +=\
                                a[state2index[child_bif1]][i, k] * a[state2index[child_bif2]][k+1, j]

                    else:
                        if   self._get_statetype(state) == "L": emit_prob = self._trans_prob_dict[state]["emit"][seq[i-1]]
                        elif self._get_statetype(state) == "R": emit_prob = self._trans_prob_dict[state]["emit"][seq[j-1]]
                        elif self._get_statetype(state) == "P": emit_prob = self._trans_prob_dict[state]["emit"][seq[i-1] + seq[j-1]]
                        else:                                   emit_prob = 1.
                        
                        for child_state, trans_prob in self._trans_prob_dict[state]["trans"].items():
                            a[state2index[state]][i][j] += \
                                emit_prob *\
                                trans_prob *\
                                a[state2index[child_state]]\
                                    [i + DeltaL[self._get_statetype(state)],\
                                     j - DeltaR[self._get_statetype(state)]]

        return a[0][1,L]

    def _get_state2index(self):
        d = dict(enumerate(self._trans_prob_dict.keys()))
        return {v:k for k,v in d.items()}

    def _get_statetype(self, state_name):
        return state_name.split("_")[0][-1]


class CMReader:
    """
    parser for cm file.
    """
    def __init__(self, path):
        self.path = path
        with open(path, "r") as f:
            self.content = f.readlines()

        # loading threshold info.
        for line in self.content:
            if re.match(r"GA\s+\d+(\.)+\d+", line) != None:
                self.GA_THRESHOLD = float(re.match(r"GA\s+\d+(\.)+\d+", line).group().split(" ")[-1])
            elif re.match(r"TC\s+\d+(\.)+\d+", line) != None:
                self.TC_THRESHOLD = float(re.match(r"TC\s+\d+(\.)+\d+", line).group().split(" ")[-1])
            elif re.match(r"NC\s+\d+(\.)+\d+", line) != None:
                self.NC_THRESHOLD = float(re.match(r"NC\s+\d+(\.)+\d+", line).group().split(" ")[-1])
                break

        self.grammar = grammar.grammar_CM
        self.cfg = CFG.fromstring(self.grammar)

    def _num_to_state(self):
        num_to_state= dict()
        flag = False
        for line in self.content:
            if re.match("CM", line):
                flag = True
            elif re.match("//", line):
                flag = False
                break
                
            # main part
            if flag and not re.match("CM", line):
                # state lines
                if re.match(r"\s{45}", line) == None:
                    state_line = re.split(r"\s+", line)
                    num_to_state[int(state_line[2])] = "_".join(state_line[1:3])
        return num_to_state

    def _read_node_line(self, line):
        nodeline = re.split(r"\s+", line)[2:4]
        node_name = "_".join(nodeline)
        
        return node_name

    def _read_state_line(self, line):
        """
        auxiliary function for `load_derivation_dict`.
        Read lines describing information about a state in a node.
        """
        num_to_state = self._num_to_state()
        state_line   = re.split(r"\s+", line)
        state_name   = "_".join(state_line[1:3])
                          
        # state line has two information of transtion and emission
        # 2-1. transision
        prob_trans       = OrderedDict()
        lowest_child_idx = int(state_line[5])
        num_child        = int(state_line[6]) if int(state_line[6]) <=6 else 2 
        for i in range(num_child):
            # BIF でない場合
            if re.match(r"B_\d+", state_name) == None: 
                log_odds = state_line[11+i]
                if log_odds != '*':
                    prob = 2**float(log_odds)
                else: # proc "*" in D or E state
                    prob = float(0)
                prob_trans.update({num_to_state[lowest_child_idx + i]:prob})

            # if BIF, left col of lowest_child_idx = BIF_R
            # prob o both splited state is １
            else:
                prob_trans.update({num_to_state[lowest_child_idx]:float(1), num_to_state[int(state_line[6])]:float(1)})
     
        # 2-2. emission
        if re.match(r"(IL|IR|ML|MR)_\d+", state_name) != None:
            prob_emit  = {k:(2**float(v)/4) for k,v in zip(['A', 'C', 'G', 'U'], state_line[-5:-1])}
        elif 'MP' in state_name:
            double_nuc = [n+m for n in ['A', 'C', 'G', 'U'] for m in ['A', 'C', 'G', 'U'] ]
            prob_emit  = {k:(2**float(v)/16) for k, v in zip(double_nuc, state_line[-17:-1])}
        else: # S or E or D
            prob_emit  = {}

        assert re.match(r"[A-Z+]+_\d+", state_name) != None, "State name is strange."
        assert len(prob_trans) <= 6, "Length of transition prob is strange."
        assert len(prob_emit) in {0, 4, 16}, "Length of emission prob is strange."

        return state_name, prob_trans, prob_emit
    
    def load_derivation_dict_from_cmfile(self):
        """
        load trans/emit prob as json format from the cm file.
        To calculate trans/emit prob, we have to convert log-odds to prob.
        For transmission except for bifurcation, log-odds = \log_2{prob_transition}. 
        For emission except, log-odds = \log_2{\frac{prob_transition}{1/4}}. 
        '*' indicates, infinity i.e. impossible transition/emission.
        """
        derivation_dict = OrderedDict()
        flag            = False
        for line in self.content:
            #start reading
            if re.match("CM", line):
                flag = True
            # finish reading
            elif re.match("//", line):
                flag = False
                
            # main part
            if flag and not re.match("CM", line):
                # 1: node line
                if re.match(r"\s{45}", line):
                    state_dict = OrderedDict() # extract state infos of a node
                    node_name  = self._read_node_line(line)
                # 2: state line. called many times in a node
                else:
                    state_name, prob_trans, prob_emit = self._read_state_line(line)
                    state_dict.update({state_name:{"trans":prob_trans, "emit":prob_emit}})
                derivation_dict.update({node_name:state_dict})

        return derivation_dict
    
    def _make_derivation_state_column_from_rule2value(self, rule_to_value):
        """
        auxiliary function for `make_derivation`
        """
        derivation_column = []
        for rule in self.cfg.productions():
            if rule in rule_to_value:
                derivation_column.append(rule_to_value[rule])
            else:
                derivation_column.append(float("nan"))
        return derivation_column
    
    def _make_rule2val_from_trans_emit_dict(self, current_state_name, trans_emit):
        current_state_type = current_state_name.split("_")[0]
        rule_to_value = dict()
        n     = Nonterminal('n')
        nl    = Nonterminal('nl')
        nr    = Nonterminal('nr')
        nl_nr = Nonterminal('nl_nr')
        S     = Nonterminal('S')
        B     = Nonterminal('B')
        MP    = Nonterminal('MP')
        
        # 1: emission
        for nuc, val in trans_emit['emit'].items():
            if nuc in {'A', 'C', 'G', 'U'}:
                rule_to_value.update({Production(n, [nuc]):val})
            else:# double emissin
                rule_to_value.update({Production(nl_nr, [nuc]):val})

        # 2: transition
        for child_state, val in trans_emit["trans"].items():
            child_state_type = child_state.split("_")[0]
            if current_state_type in {'IL', 'ML'}:
                rule = Production(Nonterminal(current_state_type), [n, '.', Nonterminal(child_state_type)])
            elif current_state_type in {'IR', 'MR'}:
                rule = Production(Nonterminal(current_state_type), [Nonterminal(child_state_type), '.', n])
            elif current_state_type == 'MP':
                rule = Production(MP, [nl, '(', Nonterminal(child_state_type), ')', nr])
            elif current_state_type in {'S', 'D'}:
                rule = Production(Nonterminal(current_state_type), [Nonterminal(child_state_type)])
            elif current_state_type == 'B':
                rule = Production(B, [S, S])
            else:
                raise Exception(f"Unidentified current_state_type: {current_state_type}")
            
            rule_to_value.update({rule:val if val != '*' else 0})
        
        if current_state_type == "E":
            rule = Production(Nonterminal("E"), [])
            rule_to_value.update({rule:1})
        return rule_to_value

    def make_derivation_array_from_derivation_dict(self, derivation_dict):
        derivation_array = []
        for states in derivation_dict.values():
            for current_state_name, trans_emit in states.items():
                rule_to_value           = self._make_rule2val_from_trans_emit_dict(current_state_name, trans_emit)
                derivation_state_column = self._make_derivation_state_column_from_rule2value(rule_to_value)
                derivation_array.append(derivation_state_column)

        return derivation_array

    # def make_derivation_dict_from_derivation_array(self, derivation_array):
    #     return

    # def dump_cmfile_from_derivatioin_dict(self, node_dict, new_file_path):
    #     """
    #     write new cm file from node_dict
    #     node_dict: json format. check "read_trans_emit_info" method
    #     new_file_path: path for the new cm file.
    #     For more details, check: https://github.com/EddyRivasLab/infernal/blob/master/src/cm_file.c
    #     """
    #     cmd = f"cp {self.path} {new_file_path}"
    #     return print(f"Created {new_file_path} based off of {self.path}!")


class TracebackFileReader:
    """
    Class for traceback -> aligned_tbdict
    Load cm file from cm reader and utilize the key to fill the val in traceback file.
    """
    def __init__(self, cm_file, traceback_file):
        self.traceback_file = traceback_file
        self.traceback      = gzip.open(traceback_file, "rb")
        self.cm_file        = cm_file
        cmio = CMReader(self.cm_file)
        self.cm_deriv_dict  = cmio.load_derivation_dict_from_cmfile()

    def traceback_iter(self):
        """
        returns generator of traceback text of a single strand.
        A traceback file may be too heavy.
        """
        Read      = True
        IsContent = False
        tbtext    = []
        line      = self.traceback.readline()
        while Read:
            if line.startswith(b'----- ------ ------ ------- ----- ----- ----- ----- ----- -----\n'):
                IsContent = not IsContent
                if not IsContent:
                    Read = False
                    break
            elif IsContent:
                tbtext.append(line)
            elif line == b'':
                break
            line = self.traceback.readline()
        yield tbtext
        
    def _make_tbdf_from_tbtext(self, traceback):
        backtrack_log = []
        for line in traceback:
            _, _, *token, _ = re.split(r"\s+", line.decode())
            backtrack_log.append(token)
        header = "emitl  emitr   state  mode  nxtl  nxtr  prv   tsc   esc"
        return pd.DataFrame(backtrack_log, columns = re.split(r"\s+", header))

    def _extract_trans_emit(self,row):
        current_state = re.search(r"[A-Z]+", row["state"]).group() + "_" + re.search(r"\d+", row["state"]).group()
        emitl         = row["emitl"][-1] if not row["emitl"].isnumeric() else ""
        emitr         = row["emitr"][-1] if not row["emitr"].isnumeric() else ""
        return current_state, emitl+emitr

    def _make_tbdict_from_tbdf(self,tbdf):
        traceback_dict = OrderedDict()
        parent_state   = ""
        last_emit      = ""
        bif_stack      = deque() # prepare bif for bifircation
        
        #  iteration for child
        for i, row in tbdf.iterrows():
            trans_emit_from_parent   = OrderedDict({"trans":OrderedDict(), "emit":OrderedDict()})
            child_state, emitted_nuc = self._extract_trans_emit(row)
            if "B" in child_state:
                bif_stack.append(child_state)
                
            # Pass for the first state(S)
            if i != 0:
                # proc S of BIF_R. The right before state must be "E" state. 
                # trans: B -> S
                if "S" in child_state and "E" in parent_state:
                    # 直前のBまで戻ってB→Sを追加
                    parent_bif = bif_stack.pop()
                    traceback_dict[parent_bif]["trans"].update({child_state:1})
                    
                else: # transitions other than BIF.
                    # For insertion, this may be called for multiple times.
                    if child_state in trans_emit_from_parent["trans"]:
                        trans_emit_from_parent["trans"][child_state] += 1
                    else:
                        trans_emit_from_parent["trans"].update({child_state:1})
                        
                if not last_emit == "":
                    if last_emit in trans_emit_from_parent["emit"]:
                        trans_emit_from_parent.update["emit"][last_emit] += 1
                    else:
                        trans_emit_from_parent["emit"].update({last_emit:1})
            
                traceback_dict.update({parent_state:trans_emit_from_parent})
            parent_state = child_state
            last_emit    = emitted_nuc
        return traceback_dict

    def make_aligned_tbdict_from_tbdf(self,tbdf):
        """
        Fill zeros in missing val in tbdict.
        Use cm_dict as a template.
        """
        aligned_tbdict = copy.deepcopy(self.cm_deriv_dict)
        tbdict         = self._make_tbdict_from_tbdf(tbdf)
        # Is there all keys in node_dict?
        # if there, assign count
        # else padding with zero.
        for node, states_in_nodes in aligned_tbdict.items():
            for parent_state, trans_emit in states_in_nodes.items():
                # no parent state, assign zero
                if not parent_state in tbdict: 
                    for child_state, prob in trans_emit["trans"].items():
                        aligned_tbdict[node][parent_state]["trans"][child_state] = 0
                    for nuc, prob in trans_emit["emit"].items():
                        aligned_tbdict[node][parent_state]["emit"][nuc] = 0
                else:
                    # any panret state, search deeper
                    for child_state, prob in trans_emit["trans"].items():
                        # devide by num of transitions.
                        # BIF is a special case
                        sum_count_from_parent = 1 if "B" in parent_state else sum(tbdict[parent_state]["trans"].values()) 
                        if child_state in tbdict[parent_state]["trans"]:
                            val = tbdict[parent_state]["trans"][child_state]/sum_count_from_parent
                        else:
                            val = 0
                        aligned_tbdict[node][parent_state]["trans"][child_state] = val

                    for nuc, prob in trans_emit["emit"].items():
                        sum_count_from_parent = sum(tbdict[parent_state]["emit"].values())
                        if nuc in tbdict[parent_state]["emit"]:
                            count = tbdict[parent_state]["emit"][nuc]/sum_count_from_parent
                        else:
                            count = 0
                        aligned_tbdict[node][parent_state]["emit"][nuc] = count
                        
        return aligned_tbdict

    def make_aligned_tbdict_from_tbdf_ELinitCM(self,tbdf):
        """
        Complements 0 to missing values in tbdict. 
        Edit cm_dict as a template.
        When EL state appears, all the values of cm_deriv_dict are taken.  
        """
        aligned_tbdict = copy.deepcopy(self.cm_deriv_dict)
        tbdict         = self._make_tbdict_from_tbdf(tbdf)
        # Is there all keys in node_dict?
        # if there, assign count
        # else padding with zero.
        modeEL = False
        for node, states_in_nodes in aligned_tbdict.items():
            for parent_state, trans_emit in states_in_nodes.items():
                # ELmode ends when S state appears during ELmode
                if modeEL and ("S" in parent_state):
                    modeEL = False

                if parent_state in tbdict.keys(): 
                    for nuc, prob in trans_emit["emit"].items():
                        sum_count_from_parent = sum(tbdict[parent_state]["emit"].values())
                        if nuc in tbdict[parent_state]["emit"]:
                            count = tbdict[parent_state]["emit"][nuc]/sum_count_from_parent
                        elif not modeEL:
                            count = 0
                        else:
                            count = aligned_tbdict[node][parent_state]["emit"][nuc]
                        aligned_tbdict[node][parent_state]["emit"][nuc] = count

                    # If tbdict's child has EL state, ELmode.
                    for tbchild in tbdict[parent_state]["trans"]:
                        if "EL_" in tbchild:
                            modeEL = True
                            break

                    for child_state, prob in trans_emit["trans"].items():
                        sum_count_from_parent = 1 if "B" in parent_state else sum(tbdict[parent_state]["trans"].values()) 
                        if child_state in tbdict[parent_state]["trans"]:
                            val = tbdict[parent_state]["trans"][child_state]/sum_count_from_parent
                        elif not modeEL:
                            val = 0
                        else:
                            val = aligned_tbdict[node][parent_state]["trans"][child_state]
                        aligned_tbdict[node][parent_state]["trans"][child_state] = val
                    
        return aligned_tbdict

# reconstruction of derivdict from tr/s/p
# for sampliing sequences from decoder output.
def make_deriv_dict_from_trsp(cm_deriv_dict, trsp):
    """
    Getting cm_deriv_dict from cmreader is a time-consuming process.
    so, this function takes cm_deriv_dict as an input.
    """
    tr,s,p = list(map(lambda x: x.squeeze().detach().numpy(), trsp))
    dirty_deriv_dict = copy.deepcopy(cm_deriv_dict)
    all_rules = CFG.fromstring(grammar.grammar_CM).productions()
    
    n     = Nonterminal('n')
    nl    = Nonterminal('nl')
    nr    = Nonterminal('nr')
    nl_nr = Nonterminal('nl_nr')
    S     = Nonterminal('S')
    B     = Nonterminal('B')
    MP    = Nonterminal('MP')

    s_i = 0
    p_i = 0
    for node_i, (node, states_in_nodes) in enumerate(dirty_deriv_dict.items()):
        for parent_state, trans_emit in states_in_nodes.items():
            parent_state_type = parent_state.split("_")[0]
            # 1: emission
            for nuc, val in trans_emit['emit'].items():
                if nuc in {'A', 'C', 'G', 'U'}:
                    rule = Production(n, [nuc])
                    rule_i = all_rules.index(rule) -56
                    rule_val = s[rule_i, s_i]
                else:# double emissin
                    rule = Production(nl_nr, [nuc])
                    rule_i = all_rules.index(rule) -60
                    rule_val = p[rule_i, p_i]
                dirty_deriv_dict[node][parent_state]["emit"][nuc] = rule_val
            
            if parent_state_type in {'IL', 'ML', 'IR', 'MR'}:
                s_i+=1
            elif parent_state_type in {'MP'}:
                p_i+=1
                    
            # fill tr values
            for child_state, val in trans_emit["trans"].items():
                child_state_type = child_state.split("_")[0]
                if parent_state_type in {'IL', 'ML'}:
                    rule = Production(Nonterminal(parent_state_type), [n, '.', Nonterminal(child_state_type)])
                elif parent_state_type in {'IR', 'MR'}:
                    rule = Production(Nonterminal(parent_state_type), [Nonterminal(child_state_type), '.', n])
                elif parent_state_type == 'MP':
                    rule = Production(MP, [nl, '(', Nonterminal(child_state_type), ')', nr])
                elif parent_state_type in {'S', 'D'}:
                    rule = Production(Nonterminal(parent_state_type), [Nonterminal(child_state_type)])
                elif parent_state_type == 'B':
                    rule = Production(B, [S, S])
                else:
                    raise Exception(f"Unidentified parent_state_type: {parent_state_type}")
                
                rule_i = all_rules.index(rule)
                rule_val = tr[rule_i, node_i]
                dirty_deriv_dict[node][parent_state]["trans"][child_state] = rule_val
                
    def cleanup_deriv_dict(dirty_deriv_dict):
        """
        values of transition are not normalized.
        So, normalize so that sum of transitions =1
        """
        for node, states_in_nodes in dirty_deriv_dict.items():
            for parent_state, trans_emit in states_in_nodes.items():
                scale = sum(trans_emit["trans"].values())
                for child_state, val in trans_emit["trans"].items():
                    dirty_deriv_dict[node][parent_state]["trans"][child_state] = val/scale
        return dirty_deriv_dict
                
    return cleanup_deriv_dict(dirty_deriv_dict)

# conversion of derivdict to tr/s/p
def make_trsp_from_deriv_dict(path_to_cmfile, deriv_dict):
    """
    function to make trsp(onehot) from dictionary of CM.
    """
    cmreader = CMReader(path_to_cmfile)
    trans_map, single_map, pair_map = [], [], []
    
    for node, states in deriv_dict.items():
        trans_col    = [np.nan]*56
        for current_state_name, trans_emit in states.items():

            # emissionは何もない可能性があるので+1しておく
            if re.match(r".+P_", current_state_name) != None:
                pair_col   = [np.nan]*16
            if re.match(r".+(L|R)_", current_state_name) != None:
                single_col = [np.nan]*4

            rule_to_value = cmreader._make_rule2val_from_trans_emit_dict(current_state_name, trans_emit)
            for rule, val in rule_to_value.items():
                idx = cmreader.cfg.productions().index(rule)
                if idx <= 55:
                    trans_col[idx] = val
                elif 56 <= idx <= 59:
                    single_col[idx-56] = val
                elif 60 <= idx <= 76:
                    pair_col[idx-60] = val

            if re.match(r".+P_", current_state_name) != None:
                pair_map.append(pair_col)
            if re.match(r".+(L|R)_", current_state_name) != None:
                single_map.append(single_col)
        trans_map.append(trans_col)

    return torch.from_numpy(np.vstack(trans_map)),\
            torch.from_numpy(np.vstack(single_map)),\
            torch.from_numpy(np.vstack(pair_map))

# test
if __name__ == '__main__':
    # reader  = TracebackFileReader(
    #     "/Users/sumishunsuke/Desktop/RNA/genzyme/datasets/RF00163/RF00163.cm",
    #     "./outputs/EXP03/GVAE_chemparam_g4/GVAEchemparam_g4_random_sampled_traceback.txt.gz"
    #     )
    # content      = list(reader.traceback_iter())[0]
    # tbdf         = reader._make_tbdf_from_tbtext(content)
    # align_tbdict = reader.make_aligned_tbdict_from_tbdf(tbdf)
    # pprint(align_tbdict)
    # pprint(deriv)

    # toy CM. (see Fig.7 in [Janssen, BMC Bioinfo., 2015])
    root0 = OrderedDict([('S_0',
                                {'trans': OrderedDict([
                                    ('IL_1', 0.),
                                    ('IR_2', 0.),
                                    ('MP_3', 1.),
                                    ('ML_4', 0.),
                                    ('MR_5', 0.),
                                    ('D_6', 0.)]),
                                'emit': {}}),
                        ('IL_1',
                            {'trans': OrderedDict([('IL_1', 1/6),
                                        ('IR_2', 1/6),
                                        ('MP_3', 1/6),
                                        ('ML_4', 1/6),
                                        ('MR_5', 1/6),
                                        ('D_6',  1/6)]),
                            'emit': {'A': 0.25,
                            'C': 0.25,
                            'G': 0.25,
                            'U': 0.25}}),
                        ('IR_2',
                            {'trans': OrderedDict([('IR_2', 1/6),
                                        ('MP_3', 1/6),
                                        ('ML_4', 1/6),
                                        ('MR_5', 1/6),
                                        ('D_6', 1/6)]),
                            'emit': {'A': 0.25,
                            'C': 0.25,
                            'G': 0.25,
                            'U': 0.25}})])

    matp1 =OrderedDict([('MP_3',
                                {'trans': OrderedDict([('IL_7', 1/3),
                                            ('IR_8', 0.),
                                            ('ML_9', 2/3),
                                            ('D_10', 0.)]),
                                'emit': {'AA': 0.,
                                'AC': 0.,
                                'AG': 0.,
                                'AU': 0.,
                                'CA': 0.,
                                'CC': 0.,
                                'CG': 0.,
                                'CU': 0.,
                                'GA': 0.,
                                'GC': 0.,
                                'GG': 0.,
                                'GU': 0.,
                                'UA': 1.,
                                'UC': 0.,
                                'UG': 0.,
                                'UU': 0.}}),
                            ('ML_4',
                                {'trans': OrderedDict([('IL_7',
                                            1/4),
                                            ('IR_8', 1/4),
                                            ('ML_9', 1/4),
                                            ('D_10', 1/4)]),
                                'emit': {'A': 0.25,
                                'C': 0.25,
                                'G': 0.25,
                                'U': 0.25}}),
                            ('MR_5',
                                {'trans': OrderedDict([('IL_7',
                                            1/4),
                                            ('IR_8', 1/4),
                                            ('ML_9', 1/4),
                                            ('D_10', 1/4)]),
                                'emit': {'A': 0.25,
                                'C': 0.25,
                                'G': 0.25,
                                'U': 0.25}}),
                            ('D_6',
                                {'trans': OrderedDict([('IL_7',
                                            1/4),
                                            ('IR_8', 1/4),
                                            ('ML_9', 1/4),
                                            ('D_10', 1/4)]),
                                'emit': {}}),
                            ('IL_7',
                                {'trans': OrderedDict([('IL_7',
                                            0.),
                                            ('IR_8', 0.),
                                            ('ML_9', 1.),
                                            ('D_10', 0.)]),
                                'emit': {'A': 1.,
                                'C': 0.,
                                'G': 0.,
                                'U': 0.}}),
                            ('IR_8',
                                {'trans': OrderedDict([('IR_8',
                                            1/3),
                                            ('ML_9', 1/3),
                                            ('D_10', 1/3)]),
                                'emit': {'A': 0.25,
                                'C': 0.25,
                                'G': 0.25,
                                'U': 0.25}})])

    matl2 = OrderedDict([
                            ('ML_9',
                                {'trans': OrderedDict([('IL_11', 0.), ('E_12', 1.)]),
                                'emit': {'A': 1.,
                                'C': 0.,
                                'G': 0.,
                                'U': 0.}}),
                            ('D_10',
                                {'trans': OrderedDict([
                                    ('IL_11', 0.5),
                                    ('E_12', 0.5)]),
                                'emit': {}}),
                            ('IL_11',
                                {'trans': OrderedDict([('IL_11', 0.), ('E_12', 1.)]),
                                'emit': {'A': 0.25,
                                'C': 0.25,
                                'G': 0.25,
                                'U': 0.25}})])

    end3 = OrderedDict([('E_12', {'trans': OrderedDict(), 'emit': {}})])

    test_dict = OrderedDict()
    test_dict["ROOT_0"] = root0
    test_dict["MATP_1"] = matp1
    test_dict["MATL_2"] = matl2
    test_dict["END_3"] = end3

    cm = CovarianceModel(test_dict)
    print(cm.cmeval("UAA"))

