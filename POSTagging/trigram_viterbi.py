#!/usr/bin/python

"""

Implement the trigram Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.

Usage:  python trigram_viterbi.py hmm-file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""

import sys
import math

def trigram_viterbi_ip():
    # constants init
    init_state = "init"
    final_state = "final"
    OOV_symbol = "OOV"

    verbose = False

    """
    vocab => All words vocab
    TP_TRI => Transition probabilities from trigram
    EP => Emission probablities 
    States => All states
    TP_BI => Transition probabilities from bigram
    BackTrace => Dictonary to store previous best states
    """

    vocab = {} 
    TP_TRI = {}  
    EP = {}  
    States = {}
    TP_BI = {}

    ourHmmFile = sys.argv[1]

    # read in the HMM and store the probabilities as log probabilities
    with open(ourHmmFile, 'r') as HMM:
        for line in HMM:
            if line.startswith("transBI"):
                _, qq, q, p = line.split()
                TP_BI[qq] = TP_BI.get(qq, {})
                TP_BI[qq][q] = math.log(float(p))
                States[qq] = 1
                States[q] = 1
            elif line.startswith("emit"):
                _, q, w, p = line.split()
                EP[q] = EP.get(q, {})
                EP[q][w] = math.log(float(p))
                States[q] = 1
                vocab[w] = 1
            elif line.startswith("trans"):
                _, qqq, qq, q, p = line.split()
                TP_TRI[(qqq, qq)] = TP_TRI.get((qqq, qq), {})
                TP_TRI[(qqq, qq)][q] = math.log(float(p))
                States[qqq] = 1
                States[qq] = 1
                States[q] = 1

    count=1
    lambdaa=1 
    # xfile = sys.argv[2]
    # with open(xfile, 'r') as xi:
    for sen in sys.stdin:
        # for sen in xi:
        # read in one sentence at a time
        sen = sen.strip()
        w = sen.split()
        n = len(w)
        V = {}
        V_BI = {}
        Backtrace = {}
        Backtrace_BI = {}
        V_BI[0] = {init_state: 0.0} # base case of the recurisve equations for bigram!
        V[0] = {(init_state, init_state): 0.0} # base case of the recurisve equations for trigram!

        w = [""]+w #just to make code llr with perl code
        for x in range(n): # work left to right ...
            i = x+1 #just to make code llr with perl code
            # if a word isn't in the vocabulary, rename it with the OOV symbol
            if w[i] not in vocab:
                if verbose:
                    print(f"OOV: {w[i]}", file=sys.stderr)
                w[i] = OOV_symbol
            
            V[i] = {}
            V_BI[i] = {}
            Backtrace[i] = {}
            Backtrace_BI[i] = {}
            for q in States: # consider each possible current state
                for qq in States: # each possible previous state
                    for qqq in States: # each possible previous previous state
                        if (qqq, qq) in TP_TRI and q in TP_TRI[(qqq, qq)] and q in EP and w[i] in EP[q] and (qqq, qq) in V[i - 1]: # only consider "non-zeros"
                            v = V[i - 1][(qqq,qq)] + TP_TRI[(qqq, qq)][q] + EP[q][w[i]]
                            # v = lambdaa*v_tri + (1-lambdaa)*v_bi
                            if (qq, q) not in V[i] or v > V[i][(qq, q)]:
                                # if we found a better previous state for trigram, take note!
                                V[i][(qq, q)] = v  # Trigram Viterbi probability
                                Backtrace[i][(qq, q)] = qqq # best previous previous state
                    if verbose:
                        print(f"V[{i}, {(qq, q)}] = {V[i].get((qq, q), '')} ({Backtrace[i].get((qq, q), '')})", file=sys.stderr)    
                
                    if qq in TP_BI and q in TP_BI[qq] and q in EP and w[i] in EP[q] and qq in V_BI[i - 1]: # only consider "non-zeros"
                        v_bi = V_BI[i - 1][qq] + TP_BI[qq][q] + EP[q][w[i]]
                        if q not in V_BI[i] or v_bi > V_BI[i][q]:
                            # if we found a better previous state for bigram, take note!
                            V_BI[i][q] = v_bi  # Bigram Viterbi probability
                            Backtrace_BI[i][q] = qq # best previous state
                if verbose:
                    print(f"V[{i}, {q}] = {V_BI[i].get(q, '')} ({Backtrace_BI[i].get(q, '')})", file=sys.stderr)    


        # this handles the last of the Viterbi equations, the one that brings
        # in the final state. for both bigram and trigram
        foundgoal = False
        foundgoal_BI = False
        goal = sys.float_info.min
        goal_BI = sys.float_info.min
        for qqq in States:
            for qq in States:
                if (qqq, qq) in TP_TRI and final_state in TP_TRI[(qqq, qq)] and (qqq, qq) in V[n]:
                    v = V[n][(qqq, qq)] + TP_TRI[(qqq, qq)][final_state]
                    if not foundgoal or v > goal:
                        # we found a better path from trigram; remember it
                        goal = v
                        foundgoal = True
                        q = (qqq, qq)
            if qqq in TP_BI and final_state in TP_BI[qqq] and qqq in V_BI[n]:
                v_bi = V_BI[n][qqq] + TP_BI[qqq][final_state]
                if not foundgoal_BI or v_bi > goal_BI:
                    # we found a better path from bigram; remember it
                    goal_BI = v_bi
                    foundgoal_BI = True
                    q_bi = qqq

        # this is the backtracking step for trigram.
        if foundgoal:
            t = []
            for i in range(n, 0, -1):
                t = [q[1]] + t
                q = (Backtrace[i][q], q[0])
        # this is the backtracking step for bigram.
        if foundgoal_BI:
            t_bi = []
            for i in range(n, 0, -1):
                t_bi = [q_bi]+t_bi
                q_bi = Backtrace_BI[i][q_bi] 

        if verbose:
            print(math.exp(goal), file=sys.stderr)

        # if we found a possible way from trigram we will use that 
        # if not then we will use bigram predicted tags 
        if foundgoal:
            print(" ".join(t))
        elif foundgoal_BI:
            print(" ".join(t_bi))
        else:
            print("")
        # print(f"Sen DONE {count}", file=sys.stderr)
        # count +=1
    sys.stdout.close()

if __name__ == "__main__":
    trigram_viterbi_ip()