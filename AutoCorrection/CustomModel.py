import math, collections
class CustomModel:

  """custom model = smoothed trigram with back off bigram and backoff smoothed unigram"""

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.trigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.""" 
    for sentence in corpus.corpus:
      previousToken_1 = None
      previousToken_2 = None
      for datum in sentence.data:  
        token = datum.word
        self.unigramCounts[token] = self.unigramCounts[token] + 1
        self.total += 1
        #build bi gram here
        if previousToken_1 is not None:
          self.bigramCounts[(previousToken_1, token)] += 1
          if previousToken_2 is not None:
            self.trigramCounts[(previousToken_2, previousToken_1, token)] +=1
        previousToken_2 = previousToken_1
        previousToken_1 = token

  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    score = 0.0
    previousToken_1 = None
    previousToken_2 = None
    totalUniSize = len(self.unigramCounts)
    for token in sentence:
      bi_tri_count = self.bigramCounts[(previousToken_2, previousToken_1)]
      tri_count = self.trigramCounts[(previousToken_2, previousToken_1, token)]
      getTriProb = (tri_count + 1)/(bi_tri_count + totalUniSize)

      
      bi_count = self.bigramCounts[(previousToken_1, token)]
      uni_bi_count = self.unigramCounts[previousToken_1]
      getBiProb = (bi_count + 1)/(uni_bi_count + totalUniSize)

      uni_count = self.unigramCounts[token]
      getUniProb = (uni_count+1)/(self.total+ totalUniSize)

      # interpolation
      # x = 0.08*getTriProb + 0.52*getBiProb + 0.4*getUniProb

      #backoff
      if tri_count > 0 and bi_tri_count > 0:
        x = tri_count/bi_tri_count
      elif bi_count > 0 and uni_bi_count > 0:
        x = bi_count/uni_bi_count
      else:
        if uni_count > 0:
          x = uni_count/totalUniSize
        else:
          x = (uni_count+1)/(self.total + totalUniSize)

      previousToken_2 = previousToken_1
      previousToken_1 = token
      score += math.log(x)
    return score
