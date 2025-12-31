#!/usr/bin/python

"""
Anusha Lavanuru
11/23/05

This script is to plot learning curve plots to
evaluate performance on a fixed test set (y-axis) 
against the training dataset size (x-axis). 


Usage:  Uncomment respective plot code and 
run the program without any args
python learning_curve.py

"""


import sys
import math
import subprocess
import matplotlib.pyplot as plt

HMM_SCRIPT = "train_hmm.py"
EVAL_SCRIPT = "tag_acc.py"
TRAIN_DATA = "data/ptb.2-21.txt"
TRAIN_TAGS = "data/ptb.2-21.tgs"
DD_DATA = "data/ptb.22.txt"
DD_TAGS = "data/ptb.22.tgs"

def extract_error_word(line):
    words = line.split()
    error_rate_by_word = float(words[4])
    total_errors_by_word = int(words[5][1:])
    return error_rate_by_word, total_errors_by_word

def extract_error_sen(line):
    words = line.split()
    total_errors_by_sentence = int(words[5][1:])
    error_rate_by_sentence = float(words[4])
    return error_rate_by_sentence, total_errors_by_sentence

training_sizes = range(1000, 40001, 1000)
results = {}
for training_size in training_sizes:
    subprocess.run(f"head -n {training_size} {TRAIN_DATA} > curveplot/sub_corpus.txt", shell=True)
    subprocess.run(f"head -n {training_size} {TRAIN_TAGS} > curveplot/sub_corpus_tags.tgs", shell=True)
    subprocess.run(f"python {HMM_SCRIPT} curveplot/sub_corpus_tags.tgs curveplot/sub_corpus.txt > curveplot/sub_hmm_model.hmm", shell=True)
    subprocess.run(f"python viterbi.py curveplot/sub_hmm_model.hmm < {DD_DATA} > curveplot/sub_dd_tags.out", shell=True)
    subprocess.run(f"python tag_acc.py {DD_TAGS} curveplot/sub_dd_tags.out > curveplot/sub_evaluation.txt", shell=True)
    with open("curveplot/sub_evaluation.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            # print(line)
            if 'error rate by word' in line:
                error_rate_by_word, total_errors_by_word = extract_error_word(line)
            if 'error rate by sentence' in line:
                error_rate_by_sentence, total_errors_by_sentence = extract_error_sen(line)
    i=0
    results[training_size] = {}
    results[training_size][i] = error_rate_by_word
    results[training_size][i+1] = total_errors_by_word
    results[training_size][i+2] = error_rate_by_sentence
    results[training_size][i+3] = total_errors_by_sentence
    print('completed eval with traning size :', training_size)

# print(results)

y1 = [value[0] for value in results.values()]
y2 = [value[1] for value in results.values()]
y3 = [value[2] for value in results.values()]
y4 = [value[3] for value in results.values()]


#### plot 1
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# x = training_sizes
# axs[0, 0].plot(x, y1, label='error rate by word')
# axs[0, 0].set_title('error rate by word plot')
# axs[0, 0].set_xlabel('Traning dataset size')
# axs[0, 0].set_ylabel('ERS with fixed test set')
# axs[0, 0].legend()
# axs[0, 1].plot(x, y2, label='total errors by word')
# axs[0, 1].set_title('total errors by word plot')
# axs[0, 1].set_xlabel('Traning dataset size')
# axs[0, 1].set_ylabel('TEW with fixed test set')
# axs[0, 1].legend()
# axs[1, 0].plot(x, y3, label='error rate by sentence')
# axs[1, 0].set_title('error rate by sentence plot')
# axs[1, 0].set_xlabel('Traning dataset size')
# axs[1, 0].set_ylabel('ERS with fixed test set')
# axs[1, 0].legend()
# axs[1, 1].plot(x, y4, label='total errors by sentence')
# axs[1, 1].set_title('total errors by sentence plot')
# axs[1, 1].set_xlabel('Traning dataset size')
# axs[1, 1].set_ylabel('TES with fixed test set')
# axs[1, 1].legend()
# fig.suptitle('POS-tagged data analysis Plots')
# plt.tight_layout()
# plt.show()

#### plot 2
# fig, axs = plt.subplots(1, 2)
# x = training_sizes
# axs[0].plot(x, y1, label='error rate by word')
# axs[0].set_title('Error rate by word plot')
# axs[0].set_xlabel('Traning dataset size')
# axs[0].set_ylabel('ERW with fixed test set')
# axs[0].legend()
# axs[1].plot(x, y3, label='error rate by sentence')
# axs[1].set_title('Error rate by sentence plot')
# axs[1].set_xlabel('Traning dataset size')
# axs[1].set_ylabel('ERS with fixed test set')
# axs[1].legend()
# fig.suptitle('POS-tagged data analysis Plots')
# plt.tight_layout()
# plt.show()

#### plot 3
# x = training_sizes
# plt.plot(x, y1, color='r', label='error rate by word')
# plt.plot(x, y3, color='g', label='error rate by sentence')
# plt.xlabel("Traning dataset size") 
# plt.ylabel("Varying rates with fixed test set") 
# plt.title("POS-tagged data analysis Plots") 
# plt.legend()
# plt.show()

#### plot 4
y1_min = min(y1)
y1_max = max(y1)
y1_norm = [(val - y1_min) / (y1_max - y1_min) for val in y1]
y3_min = min(y3)
y3_max = max(y3)
y3_norm = [(val - y3_min) / (y3_max - y3_min) for val in y3]
x = training_sizes
plt.plot(x, y1_norm, color='r', label='normalized error rate by word')
plt.plot(x, y3_norm, color='g', label='normalized error rate by sentence')
plt.xlabel("Traning dataset size") 
plt.ylabel("Normalized varying rates with fixed test set") 
plt.title("Normalized POS-tagged data analysis Plots") 
plt.legend()
plt.show()