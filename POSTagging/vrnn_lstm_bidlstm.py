#!/usr/bin/python

"""

This script is to train three RNN models 
    1. VanillaRNN
    2. LSTM
    3. BI Directional LSTM
for POS tagging and check their perfomences

Usage: python3 vrnn_lstm_bidlstm.py data/ptb.2-21.tgs data/ptb.2-21.txt data/ptb.22.tgs data/ptb.22.txt 22_1.out
You need to give train tags then train texts
then test tags and then test texts as args

"""

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Dense, Input, TimeDistributed, LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import sys, re
from collections import defaultdict

class BonusQuestionModels:
    def __init__(self):
        self.X = []
        self.Y = []
        self.X_train = []
        self.Y_train = []
        self.X_val = []
        self.Y_val = []
        self.X_test = []
        self.Y_test = []
        self.emissions = {}
        self.transitions = {}
        self.transitionsTotal = defaultdict(int)
        self.emissionsTotal = defaultdict(int)
        self.word_tokenizer = Tokenizer()
        self.tag_tokenizer = Tokenizer(lower=False)
        self.MAX_SEQ_LENGTH = 120
        self.EMBEDDING_SIZE = 0
        self.VOCABULARY_SIZE = 0
        self.NUM_CLASSES = 0
        self.models = {
            'VRNN': None, 
            'LSTM': None, 
            'BIDLSTM': None
        }

    def load_data(self, TAG_FILE, TOKEN_FILE, TEST_TAG_FILE, TEST_TOKEN_FILE):
        with open(TAG_FILE) as tagFile, open(TOKEN_FILE) as tokenFile:
            for tagString, tokenString in zip(tagFile, tokenFile):
                tags = re.split("\s+", tagString.rstrip())
                tokens = re.split("\s+", tokenString.rstrip())
                X_sentence = []
                Y_sentence = []
                pairs = list(zip(tags, tokens))
                for (tag, token) in pairs:
                    X_sentence.append(token)
                    Y_sentence.append(tag)
                self.X.append(X_sentence)
                self.Y.append(Y_sentence)
        print(len(self.X), len(self.Y))
        # num_words = len(set([word.lower() for sentence in self.X for word in sentence]))
        # num_tags = len(set([word.lower() for sentence in self.Y for word in sentence]))
        num_words = len(set([word for sentence in self.X for word in sentence]))
        num_tags = len(set([word for sentence in self.Y for word in sentence]))
        print("Total number of tagged sentences: {}".format(len(self.X)))
        print("Vocabulary size: {}".format(num_words))
        print("Total number of tags: {}".format(num_tags))

        with open(TEST_TAG_FILE) as tagFile, open(TEST_TOKEN_FILE) as tokenFile:
            for tagString, tokenString in zip(tagFile, tokenFile):
                tags = re.split("\s+", tagString.rstrip())
                tokens = re.split("\s+", tokenString.rstrip())
                X_sentence = []
                Y_sentence = []
                pairs = list(zip(tags, tokens))
                for (tag, token) in pairs:
                    X_sentence.append(token)
                    Y_sentence.append(tag)
                self.X_test.append(X_sentence)
                self.Y_test.append(Y_sentence)

    def preprocess_data(self):
        self.word_tokenizer.fit_on_texts(self.X)
        X_encoded = self.word_tokenizer.texts_to_sequences(self.X)
        # X_encoded_back = self.word_tokenizer.sequences_to_texts(X_encoded)
        X_test_encoded = self.word_tokenizer.texts_to_sequences(self.X_test)

        self.tag_tokenizer.fit_on_texts(self.Y)
        Y_encoded = self.tag_tokenizer.texts_to_sequences(self.Y)
        # Y_encoded_back = self.tag_tokenizer.sequences_to_texts(Y_encoded)
        Y_test_encoded = self.tag_tokenizer.texts_to_sequences(self.Y_test)

        lengths = [len(seq) for seq in X_encoded]
        print("Length of longest sentence: {}".format(max(lengths)))

        different_length = [1 if len(input) != len(output) else 0 for input, output in zip(X_encoded, Y_encoded)]
        print("{} sentences have disparate input-output lengths.".format(sum(different_length)))

        X_padded = pad_sequences(X_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="pre", truncating="post")
        Y_padded = pad_sequences(Y_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="pre", truncating="post")
        X_test_padded = pad_sequences(X_test_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="pre", truncating="post")
        Y_test_padded = pad_sequences(Y_test_encoded, maxlen=self.MAX_SEQ_LENGTH, padding="pre", truncating="post")

        self.X, self.Y = X_padded, Y_padded
        self.Y = to_categorical(self.Y)
        VALID_SIZE = 0.15
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X, self.Y, test_size=VALID_SIZE, random_state=4)

        self.X_test, self.Y_test = X_test_padded, Y_test_padded
        self.Y_test = to_categorical(self.Y_test)

        desired_shape = self.Y_train.shape

        if self.Y_test.shape[2] < desired_shape[2]:
            padding = desired_shape[2] - self.Y_test.shape[2]
            self.Y_test = np.pad(self.Y_test, ((0, 0), (0, 0), (0, padding)), 'constant')
            print("Added padding to missing tags in test data")
        else:
            print("One hot encoding size is same for both train and test")
            
        self.NUM_CLASSES = self.Y.shape[2]
        self.EMBEDDING_SIZE = 400
        self.VOCABULARY_SIZE = len(self.word_tokenizer.word_index) + 1

    def build_rnn_model(self):

        rnn_model = Sequential()
        rnn_model.add(Embedding(input_dim=self.VOCABULARY_SIZE,
                               output_dim=self.EMBEDDING_SIZE,
                               input_length=self.MAX_SEQ_LENGTH,
                               trainable=True))

        rnn_model.add(SimpleRNN(64, return_sequences=True))
        rnn_model.add(TimeDistributed(Dense(self.NUM_CLASSES, activation='softmax')))

        rnn_model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['acc'])

        rnn_model.summary()
        self.models["VRNN"] = rnn_model

    def build_lstm_model(self):

        lstm_model = Sequential()
        lstm_model.add(Embedding(input_dim=self.VOCABULARY_SIZE,
                                output_dim=self.EMBEDDING_SIZE,
                                input_length=self.MAX_SEQ_LENGTH,
                                trainable=True))

        lstm_model.add(LSTM(64, return_sequences=True))
        lstm_model.add(TimeDistributed(Dense(self.NUM_CLASSES, activation='softmax')))

        lstm_model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['acc'])

        lstm_model.summary()
        self.models["LSTM"] = lstm_model

    def build_bidirectional_model(self):

        bidirect_model = Sequential()
        bidirect_model.add(Embedding(input_dim=self.VOCABULARY_SIZE,
                                    output_dim=self.EMBEDDING_SIZE,
                                    input_length=self.MAX_SEQ_LENGTH,
                                    trainable=True))

        bidirect_model.add(Bidirectional(LSTM(64, return_sequences=True)))
        bidirect_model.add(TimeDistributed(Dense(self.NUM_CLASSES, activation='softmax')))

        bidirect_model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['acc'])

        bidirect_model.summary()
        self.models["BIDLSTM"] = bidirect_model

    def train_models(self):
        for model_type in self.models:
            if self.models[model_type] is not None:
                print(f"Training {model_type} model...")
                self.models[model_type].fit(self.X_train, self.Y_train, batch_size=125, epochs=10, validation_data=(self.X_val, self.Y_val))
                print(f"{model_type} model trained successfully.")

    def evaluate_model(self, model_type):
        if model_type in self.models:
            model = self.models[model_type]
            if model is not None:
                loss, accuracy = model.evaluate(self.X_test, self.Y_test, verbose=1)
                print(f"{model_type} Model - Loss: {loss}, Accuracy: {accuracy}")
            else:
                print(f"{model_type} model has not been trained yet.")
        else:
            print(f"Invalid model type: {model_type}")

    def evaluate_model_and_write_to_file(self, model_type, file_path):
        if model_type in self.models:
            model = self.models[model_type]
            if model is not None:
                print(f"predicting tags with model type: {model_type}")
                predicted_one_hot_labels = model.predict(self.X_test)
                predicted_labels = np.argmax(predicted_one_hot_labels, axis=2)
                test_tags = self.tag_tokenizer.sequences_to_texts(predicted_labels)
                # actual_tags = [tag.upper() for tag in test_tags]
                print(f"writing predicted tags with model type: {model_type} ...")
                file_path = 'my_'+model_type+'_'+file_path
                with open(file_path, 'w') as file:
                    for tag in test_tags:
                        file.write(tag + ' \n')
                print(f"writing done for predictions with model type: {model_type}")

    def visualize_training(self, model_type):
        if model_type in self.models:
            model = self.models[model_type]
            if model is not None:
                plt.plot(model.history['acc'])
                plt.plot(model.history['val_acc'])
                plt.title(f'{model_type} model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc="lower right")
                plt.show()
            else:
                print(f"{model_type} model has not been trained yet.")
        else:
            print(f"Invalid model type: {model_type}")

if __name__ == "__main__":
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]
    TEST_TAG_FILE = sys.argv[3]
    TEST_TOKEN_FILE = sys.argv[4]
    PREDICT_TAG_FILE = sys.argv[5]
    bq = BonusQuestionModels()
    bq.load_data(TAG_FILE, TOKEN_FILE, TEST_TAG_FILE, TEST_TOKEN_FILE)
    bq.preprocess_data()
    bq.build_rnn_model()
    bq.build_lstm_model()
    bq.build_bidirectional_model()
    bq.train_models()
    bq.evaluate_model('VRNN')
    bq.evaluate_model_and_write_to_file('VRNN', PREDICT_TAG_FILE)
    bq.evaluate_model('LSTM')
    bq.evaluate_model_and_write_to_file('LSTM', PREDICT_TAG_FILE)
    bq.evaluate_model('BIDLSTM')
    bq.evaluate_model_and_write_to_file('BIDLSTM', PREDICT_TAG_FILE)
