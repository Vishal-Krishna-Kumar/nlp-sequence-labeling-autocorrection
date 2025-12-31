#!/usr/bin/python

"""

Implement the trigram Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.

Usage:  python trigram_viterbi_without_ip.py hmm-file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""

import sys
import math

def trigram_viterbi():
    # constants init
    init_state = "init"
    final_state = "final"
    OOV_symbol = "OOV"

    verbose = False

    """
    vocab => All words vocab
    TP => Transition probabilities
    EP => Emission probablities 
    States => All states
    BackTrace => Dictonary to store previous best states
    """

    vocab = {} 
    TP = {}  
    EP = {}  
    States = {}

    ourHmmFile = sys.argv[1]

    # read in the HMM and store the probabilities as log probabilities
    with open(ourHmmFile, 'r') as HMM:
        for line in HMM:
            if line.startswith("trans"):
                _, qqq, qq, q, p = line.split()
                TP[(qqq, qq)] = TP.get((qqq, qq), {})
                TP[(qqq, qq)][q] = math.log(float(p))
                States[qqq] = 1
                States[qq] = 1
                States[q] = 1
            elif line.startswith("emit"):
                _, q, w, p = line.split()
                EP[q] = EP.get(q, {})
                EP[q][w] = math.log(float(p))
                States[q] = 1
                vocab[w] = 1

    # xfile = sys.argv[2]
    # with open(xfile, 'r') as xi:
    count=0
    for sen in sys.stdin:
        # read in one sentence at a time
        sen = sen.strip()
        w = sen.split()
        n = len(w)
        V = {}
        Backtrace = {}
        V[0] = {(init_state, init_state): 0.0} # base case of the recurisve equations!

        w = [""]+w #just to make code llr with perl code
        for x in range(n): # work left to right ...
            i = x+1 #just to make code llr with perl code
            # if a word isn't in the vocabulary, rename it with the OOV symbol
            if w[i] not in vocab:
                if verbose:
                    print(f"OOV: {w[i]}", file=sys.stderr)
                w[i] = OOV_symbol
            
            V[i] = {}
            Backtrace[i] = {}
            for q in States: # consider each possible current state
                for qq in States: # each possible previous state
                    for qqq in States: # each possible previous previous state
                        if (qqq, qq) in TP and q in TP[(qqq, qq)] and q in EP and w[i] in EP[q] and (qqq, qq) in V[i - 1]: # only consider "non-zeros"
                            v = V[i - 1][(qqq,qq)] + TP[(qqq, qq)][q] + EP[q][w[i]]
                            if (qq, q) not in V[i] or v > V[i][(qq, q)]:
                                # if we found a better previous state, take note!
                                V[i][(qq, q)] = v  # Viterbi probability
                                Backtrace[i][(qq, q)] = qqq # best previous state
                    if verbose:
                        print(f"V[{i}, {(qq, q)}] = {V[i].get((qq, q), '')} ({Backtrace[i].get((qq, q), '')})", file=sys.stderr)    

        # this handles the last of the Viterbi equations, the one that brings
        # in the final state.
        foundgoal = False
        goal = sys.float_info.min
        for qqq in States:
            for qq in States:
                if (qqq, qq) in TP and final_state in TP[(qqq, qq)] and (qqq, qq) in V[n]:
                    v = V[n][(qqq, qq)] + TP[(qqq, qq)][final_state]
                    if not foundgoal or v > goal:
                        # we found a better path; remember it
                        goal = v
                        foundgoal = True
                        q = (qqq, qq)

        # this is the backtracking step.
        if foundgoal:
            t = []
            for i in range(n, 1, -1):
                t = [q[1]] + t
                q = (Backtrace[i][q], q[0])
        t = [q[1]] + t
        if verbose:
            print(math.exp(goal), file=sys.stderr)
        if foundgoal:
            print(" ".join(t))
        else:
            print("")
        # print(f"Sen No {count}", file=sys.stderr)
        # count +=1
    sys.stdout.close()

if __name__ == "__main__":
    trigram_viterbi()