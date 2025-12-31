# NLP Projects: POS Tagging and Autocorrection

This repository contains two distinct NLP-focused projects: **Part-of-Speech (POS) Tagging** and **Autocorrection**. Each project implements, tests, and evaluates various models and methodologies specific to its domain.

## Table of Contents
1. [POS Tagging](#pos-tagging)
   - [Introduction](#introduction)
   - [Steps to Run and Evaluate](#steps-to-run-and-evaluate)
   - [Results and Metrics](#results-and-metrics)
2. [Autocorrection](#autocorrection)
   - [Introduction](#introduction-1)
   - [Execution and Evaluation](#execution-and-evaluation)

---

## POS Tagging

### Introduction

The **POS Tagging** module investigates part-of-speech tagging using both traditional probabilistic methods and modern neural network approaches. Probabilistic models such as Hidden Markov Models (HMM) with bigram and trigram structures are implemented alongside neural architectures, including Vanilla RNNs, LSTMs, and Bidirectional LSTMs. These models are tested across multiple languages, such as English, Japanese, and Bulgarian, to assess their performance.

### Steps to Run and Evaluate

#### Using HMM with Viterbi Algorithm:
1. **Train the HMM model**:
    ```bash
    python3 train_hmm.py data/ptb.2-21.tgs data/ptb.2-21.txt > my.hmm
    ```

2. **Apply Viterbi for tagging**:
    ```bash
    python3 viterbi.py my.hmm < data/ptb.22.txt > my.out
    ```

3. **Evaluate tagging results**:
    ```bash
    python3 tag_acc.py data/ptb.22.tgs my.out
    ```

#### For Neural Models (Vanilla RNN, LSTM, Bidirectional LSTM):
1. **Train and evaluate**:
    ```bash
    python3 vrnn_lstm_bidlstm.py data/ptb.2-21.tgs data/ptb.2-21.txt data/ptb.22.tgs data/ptb.22.txt 22_1.out
    ```

### Results and Metrics

#### Evaluation Metrics

The performance is measured using the following metrics:
- **ERW (Error Rate by Word)**: Proportion of incorrectly tagged words.
- **ERS (Error Rate by Sentence)**: Proportion of sentences with at least one error.

#### Dataset Details
- **English (ENG)**: ~40,000 tokens
- **Japanese (JP)**: ~17,000 tokens
- **Bulgarian (BG)**: ~13,000 tokens

#### Results for Hidden Markov Models (HMM)

| Model   | ENG ERW | ENG ERS | JP ERW | JP ERS | BG ERW | BG ERS |
|---------|---------|---------|--------|--------|--------|--------|
| Bigram  | 0.054   | 0.655   | 0.062  | 0.136  | 0.115  | 0.751  |
| Trigram | 0.049   | 0.613   | 0.063  | 0.133  | 0.110  | 0.721  |

#### Results for Neural Models (RNNs)

| Model         | ENG ERW | ENG ERS | JP ERW | JP ERS | BG ERW | BG ERS |
|---------------|---------|---------|--------|--------|--------|--------|
| Vanilla RNN   | 0.295   | 0.758   | 0.105  | 0.155  | 0.572  | 0.809  |
| LSTM          | 0.291   | 0.748   | 0.117  | 0.191  | 0.575  | 0.821  |
| Bidirectional LSTM | 0.283 | 0.714 | 0.103  | 0.135  | 0.558  | 0.804  |

#### Learning Curve for HMM Bigram Model

The learning curve for the Bigram HMM model illustrates the change in error rates (ERW and ERS) as training progresses on increasing data.

![Learning Curve](/POSTagging/Figure_4_normalizederserwplot.png)

---

## Autocorrection

### Introduction

The **Autocorrection** project implements and evaluates different spell-correction methods. These include unigram, bigram, and trigram language models, with options for smoothing and backoff strategies. The system also incorporates an edit model to predict and correct errors. Performance is evaluated based on accuracy and runtime efficiency.

### Execution and Evaluation

#### Run the Edit Model:
1. **Sanity check**:
    ```bash
    python3 EditModel.py
    ```

#### Evaluate Autocorrection Model:
1. **Run the main autocorrection model**:
    ```bash
    python3 SpellCorrect.py
    ```

This process evaluates various language models, including their ability to handle noisy input and provide accurate corrections.

---
