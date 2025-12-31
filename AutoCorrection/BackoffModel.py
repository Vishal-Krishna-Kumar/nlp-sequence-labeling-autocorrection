import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """ 
    for sentence in corpus.corpus:
      previousToken = None
      for datum in sentence.data:  
        token = datum.word
        self.unigramCounts[token] = self.unigramCounts[token] + 1
        self.total += 1
        #build bi gram here
        if previousToken is not None:
          self.bigramCounts[(previousToken, token)] += 1
        previousToken = token 

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    previousToken = None
    totalUniSize = len(self.unigramCounts)
    for token in sentence:
      uni_count = self.unigramCounts[previousToken]
      bi_count = self.bigramCounts[(previousToken, token)]

      if bi_count > 0 and uni_count > 0:
        # get bi gram prob first
        getBiProb = bi_count/uni_count
      else:
        #unigram laplace smooth 
        getBiProb = (self.unigramCounts[token]+1)/(self.total + totalUniSize)
      previousToken = token
      score += math.log(getBiProb)
    return score
