# NLP-Postagging-and-Autocorrection

This repository contains two sub-projects focused on essential aspects of Natural Language Processing (NLP): **Part-of-Speech (POS) Tagging** and **Autocorrection**. Each sub-project focuses on implementing and evaluating models and algorithms related to their respective areas.

## Table of Contents
1. [POS Tagging](#pos-tagging)
    - [Overview](#overview)
    - [How to Run and Evaluate](#how-to-run-and-evaluate)
    - [Results and Evaluation](#results-and-evaluation)
2. [Autocorrection](#autocorrection)
    - [Overview](#overview-1)
    - [How to Run and Performance Check](#how-to-run-and-performance-check)
3. [Contributing](#contributing)

---

## POS Tagging

### Overview
The **POS Tagging** project explores part-of-speech tagging using several methods, including Hidden Markov Models (HMM) with bigram and trigram implementations. Additionally, the project compares the performance of neural models such as Vanilla RNN, LSTM, and Bidirectional LSTM for POS tagging across different languages, including English, Japanese, and Bulgarian.

### How to Run and Evaluate

#### For HMM with Viterbi Algorithm:
1. **Training the HMM model**:
    ```bash
    python3 train_hmm.py data/ptb.2-21.tgs data/ptb.2-21.txt > my.hmm
    ```

2. **Running Viterbi for POS tagging**:
    ```bash
    python3 viterbi.py my.hmm < data/ptb.22.txt > my.out
    ```

3. **Evaluating the results**:
    ```bash
    python3 tag_acc.py data/ptb.22.tgs my.out
    ```

#### For VRNN, LSTM, Bidirectional LSTM:
1. **Training and evaluation**:
    ```bash
    python3 vrnn_lstm_bidlstm.py data/ptb.2-21.tgs data/ptb.2-21.txt data/ptb.22.tgs data/ptb.22.txt 22_1.out
    ```

### Results and Evaluation

#### Evaluations
The following tables provide the evaluation of part-of-speech tagging performance using both Hidden Markov Models (HMM) and Recurrent Neural Networks (RNNs). The metrics used for evaluation are:

- **ERW** - Error Rate by Word
- **ERS** - Error Rate by Sentence

The token counts for the datasets used are:
- **English (ENG)**: ~40,000 tokens
- **Japanese (JP)**: ~17,000 tokens
- **Bulgarian (BG)**: ~13,000 tokens

#### Hidden Markov Models (HMM)

##### Emission and Transition Probabilities

| Model   | ENG ERW | ENG ERS | JP ERW | JP ERS | BG ERW | BG ERS |
|---------|---------|---------|--------|--------|--------|--------|
| Bigram  | 0.054   | 0.655   | 0.062  | 0.136  | 0.115  | 0.751  |
| Trigram | 0.049   | 0.613   | 0.063  | 0.133  | 0.110  | 0.721  |

#### Recurrent Neural Networks (RNNs)

##### Emission and Transition Probabilities

| Model      | ENG ERW | ENG ERS | JP ERW | JP ERS | BG ERW | BG ERS |
|------------|---------|---------|--------|--------|--------|--------|
| Vanilla RNN    | 0.295   | 0.758   | 0.105  | 0.155  | 0.572  | 0.809  |
| LSTM           | 0.291   | 0.748   | 0.117  | 0.191  | 0.575  | 0.821  |
| BIDLSTM        | 0.283   | 0.714   | 0.103  | 0.135  | 0.558  | 0.804  |

#### POS-Tagged Data Analysis Plot - Learning Curve for Bigram HMM

The following plot visualizes the learning curve for the Bigram HMM model, showing the error rate by word and sentence as the model trains on increasing amounts of data.

![Analysis Plot](/POSTagging/Figure_4_normalizederserwplot.png)

---

## Autocorrection

### Overview
The **Autocorrection** project evaluates different approaches to spell correction. It utilizes various language models, including unigram, bigram, trigram models, and their smoothed and backoff versions. The project employs an edit model for error correction, with evaluations based on metrics such as accuracy and runtime.

### How to Run and Performance Check

1. **Sanity check for the Edit Model**:
    ```bash
    python3 EditModel.py
    ```

2. **Evaluating the performance of the autocorrection model**:
    ```bash
    python3 SpellCorrect.py
    ```

Each of these commands runs the model with different language modeling techniques to correct spelling errors and evaluate performance based on accuracy metrics.

---

This repository provides a hands-on exploration of POS tagging and autocorrection techniques in NLP. Both traditional and neural approaches are evaluated to provide insights into their strengths and weaknesses.