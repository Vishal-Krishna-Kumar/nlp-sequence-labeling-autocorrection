#!/usr/bin/python

"""

Implement a trigrm HMM here. 
You model should output the HMM similar to `train_hmm.py`.

Usage:  python train_trigram_hmm_without_ip.py tags text > hmm-file

"""

import sys, re
from collections import defaultdict

def train_trigram_hmm():
    if len(sys.argv) != 3:
        print("Error : sys exit")
        sys.exit(1)
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]

    vocab = {}
    OOV_WORD = "OOV"
    INIT_STATE = "init"
    FINAL_STATE = "final"

    emissions = {}
    transitions = {}
    transitionsTotal = defaultdict(int)
    emissionsTotal = defaultdict(int)

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

                if tag not in emissions:
                    emissions[tag] = defaultdict(int)
                if (prevtag2, prevtag1) not in transitions:
                    transitions[(prevtag2, prevtag1)] = defaultdict(int)

                # increment the emission/transition observation
                emissions[tag][token] += 1
                emissionsTotal[tag] += 1

                transitions[(prevtag2, prevtag1)][tag] += 1
                transitionsTotal[(prevtag2, prevtag1)] += 1

                prevtag2 = prevtag1
                prevtag1 = tag

            # don't forget the stop probability for each sentence
            if (prevtag2, prevtag1) not in transitions:
                transitions[(prevtag2, prevtag1)] = defaultdict(int)
            transitions[(prevtag2, prevtag1)][FINAL_STATE] += 1
            transitionsTotal[(prevtag2, prevtag1)] += 1
    
    for (prevtag2, prevtag1) in transitions:
        for tag in transitions[(prevtag2, prevtag1)]:
            print(("trans %s %s %s %s" % (prevtag2, prevtag1, tag, float(transitions[(prevtag2, prevtag1)][tag]) / transitionsTotal[(prevtag2, prevtag1)])))

    for tag in emissions:
        for token in emissions[tag]:
            print(("emit %s %s %s " % (tag, token, float(emissions[tag][token]) / emissionsTotal[tag])))

if __name__ == "__main__":
    train_trigram_hmm()
