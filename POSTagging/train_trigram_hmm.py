#!/usr/bin/python

"""

Implement a trigrm HMM here. 
You model should output the HMM similar to `train_hmm.py`.

Usage:  python train_trigram_hmm.py tags text > hmm-file

Implemented a interpolated trigram here with bigram 
lamdaa controls interpolation => lamba*trigram+ (1-lamda)*bigram

"""

import sys, re
from collections import defaultdict

def train_trigram_hmm_ip():
    if len(sys.argv) != 3:
        print("Error : sys exit")
        sys.exit(1)
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]

    vocab = {}
    States = {}
    OOV_WORD = "OOV"
    INIT_STATE = "init"
    FINAL_STATE = "final"
    lambdaa = 0.7 # Control interpolation between trigram and bigram 

    emissions_tri = {}
    transitions_tri = {}
    transitions_triTotal = defaultdict(int)
    emissions_triTotal = defaultdict(int)

    transitions_bi = {}
    transitions_inter = {}
    transitions_biTotal = defaultdict(int)

    with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
        for tagString, tokenString in zip(tagFile, tokenFile):
            tags = re.split("\s+", tagString.rstrip())
            tokens = re.split("\s+", tokenString.rstrip())
            pairs = list(zip(tags, tokens))

            prevtag1 = INIT_STATE
            prevtag2 = INIT_STATE

            for (tag, token) in pairs:

                # this block is a little trick to help with out-of-vocabulary (OOV)
                # words.  the first time we see *any* word token, we pretend it
                # is an OOV.  this lets our model decide the rate at which new
                # words of each POS-type should be expected (e.g., high for nouns,
                # low for determiners).

                if token not in vocab:
                    vocab[token] = 1
                    token = OOV_WORD

                if tag not in States:
                    States[tag] = 1

                # increment the emission/transition observation for bigrams and trigrams

                ######## trigram transitions 
                if tag not in emissions_tri:
                    emissions_tri[tag] = defaultdict(int)
                if (prevtag2, prevtag1) not in transitions_tri:
                    transitions_tri[(prevtag2, prevtag1)] = defaultdict(int)

                emissions_tri[tag][token] += 1
                emissions_triTotal[tag] += 1

                transitions_tri[(prevtag2, prevtag1)][tag] += 1
                transitions_triTotal[(prevtag2, prevtag1)] += 1

                ######## Bigram transitions
                if prevtag1 not in transitions_bi:
                    transitions_bi[prevtag1] = defaultdict(int)

                transitions_bi[prevtag1][tag] += 1
                transitions_biTotal[prevtag1] += 1

                prevtag2 = prevtag1
                prevtag1 = tag

            ##trigram
            # don't forget the stop probability for each sentence
            if (prevtag2, prevtag1) not in transitions_tri:
                transitions_tri[(prevtag2, prevtag1)] = defaultdict(int)
            transitions_tri[(prevtag2, prevtag1)][FINAL_STATE] += 1
            transitions_triTotal[(prevtag2, prevtag1)] += 1

            ##bigram
            # don't forget the stop probability for each sentence
            if prevtag1 not in transitions_bi:
                transitions_bi[prevtag1] = defaultdict(int)

            transitions_bi[prevtag1][FINAL_STATE] += 1
            transitions_biTotal[prevtag1] += 1

    for (prevtag2, prevtag1) in transitions_tri:
        for tag in transitions_tri[(prevtag2, prevtag1)]:
            transitions_tri[(prevtag2, prevtag1)][tag] = float(transitions_tri[(prevtag2, prevtag1)][tag]) / transitions_triTotal[(prevtag2, prevtag1)]
    
    for prevtag in transitions_bi:
        for tag in transitions_bi[prevtag]:
            transitions_bi[prevtag][tag] = float(transitions_bi[prevtag][tag]) / transitions_biTotal[prevtag]

    N = len(States)

    ## interpolate trigram and bigram
    for q in States:

        # add (START,START,tag) 
        x = 0
        if (INIT_STATE,INIT_STATE) in transitions_tri and q in transitions_tri[(INIT_STATE,INIT_STATE)]:
            x += float(lambdaa) * transitions_tri[(INIT_STATE,INIT_STATE)][q]

        if INIT_STATE in transitions_bi and q in transitions_bi[INIT_STATE]:
            x += float(1.0-lambdaa) * transitions_bi[INIT_STATE][q]

        if x != 0:
            if (INIT_STATE,INIT_STATE) not in transitions_inter:
                transitions_inter[(INIT_STATE, INIT_STATE)] = defaultdict(int)
            transitions_inter[(INIT_STATE, INIT_STATE)][q] = x

        # add (START, tag, FINAL)
        x = 0
        if ((INIT_STATE,q) in transitions_tri and FINAL_STATE in transitions_tri[(INIT_STATE,q)]) :
            x += float(lambdaa) * transitions_tri[(INIT_STATE,q)][FINAL_STATE]
        if (q in transitions_bi and FINAL_STATE in transitions_bi[q]):
            x += float(1.0-lambdaa) * transitions_bi[q][FINAL_STATE]
            
        if x != 0:
            if (INIT_STATE,q) not in transitions_inter:
                transitions_inter[(INIT_STATE, q)] = defaultdict(int)
            transitions_inter[(INIT_STATE, q)][FINAL_STATE] = x

    ## interpolate trigram and bigram
    for q in States:
        for qq in States:

            # add (tag, tag,FINAL) # qq q f
            x = 0
            if ((INIT_STATE,qq) in transitions_tri and q in transitions_tri[(INIT_STATE,qq)]):
                x += float(lambdaa) * transitions_tri[(INIT_STATE,qq)][q] 
            if (qq in transitions_bi and q in transitions_bi[qq]):
                x += float(1.0-lambdaa) * transitions_bi[qq][q]

            if x != 0:
                if (INIT_STATE,qq) not in transitions_inter:
                    transitions_inter[(INIT_STATE, qq)] = defaultdict(int)
                transitions_inter[(INIT_STATE, qq)][q] = x
            
            # add (START, tag, tag) #s, qq, q
            x = 0
            if ((qq, q) in transitions_tri and FINAL_STATE in transitions_tri[(qq,q)]):
                x += float(lambdaa) *  transitions_tri[(qq,q)][FINAL_STATE]
            if (q in transitions_bi and FINAL_STATE in transitions_bi[q]):
                x += float(1.0-lambdaa) * transitions_bi[q][FINAL_STATE]

            if x != 0:    
                if (qq, q) not in transitions_inter:
                    transitions_inter[(qq, q)] = defaultdict(int)
                transitions_inter[(qq, q)][FINAL_STATE] = x

    ## interpolate rest of tags
    for qqq in States:
        for qq in States:
            for q in States:
                x = 0
                if ((qqq, qq) in transitions_tri and q in transitions_tri[(qqq, qq)]):
                    x += float(lambdaa) * transitions_tri[(qqq,qq)][q]
                if (qq in transitions_bi and q in transitions_bi[qq]):
                    x += float(1.0-lambdaa) * transitions_bi[qq][q]

                if x != 0:
                    if (qqq,qq) not in transitions_inter:
                        transitions_inter[(qqq, qq)] = defaultdict(int)
                    transitions_inter[(qqq, qq)][q] = x

    # save interpolated transitions
    for (prevtag2, prevtag1) in transitions_inter:
        for tag in transitions_inter[(prevtag2, prevtag1)]:
            print(("trans %s %s %s %s" % (prevtag2, prevtag1, tag, transitions_inter[(prevtag2, prevtag1)][tag])))

    # save emissions
    for tag in emissions_tri:
        for token in emissions_tri[tag]:
            print(("emit %s %s %s " % (tag, token, float(emissions_tri[tag][token]) / emissions_triTotal[tag])))

    # also save bigram transitions for backoff
    for prevtag in transitions_bi:
        for tag in transitions_bi[prevtag]:
            print(("transBI %s %s %s" % (prevtag, tag, float(transitions_bi[prevtag][tag]) / transitions_biTotal[prevtag])))

if __name__ == "__main__":
    train_trigram_hmm_ip()
