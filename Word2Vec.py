import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from datetime import datetime
from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances
from glob import glob
import os
import sys
import string

sys.path.append(os.path.abspath('..'))
from brownCorpus import get_sentences_with_word2idx_limit_vocab as get_brown


def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3

# Get the data i.e sentences and word2idx mapings from brown courpus in nltk
def train_model(savedir):
  sentences, word2idx = get_brown()
  vocab_size = len(word2idx)

  # Hyperparameter initialization 
  window_size = 5
  learning_rate = 0.025
  final_learning_rate = 0.0001
  num_negatives = 5 
  epochs = 20
  D = 50 


  # Claculating learning rate decay
  learning_rate_delta = (learning_rate - final_learning_rate) / epochs
  W = np.random.randn(vocab_size, D) 
  V = np.random.randn(D, vocab_size) 


  # Probability distribution for drawing negative samples.
  p_neg = get_negative_sampling_distribution(sentences, vocab_size)
  costs = []
  total_words = sum(len(sentence) for sentence in sentences)
  print("Total number of words in corpus:", total_words)

  # Subsampling each sentence
  threshold = 1e-5
  p_drop = 1 - np.sqrt(threshold / p_neg)


  #Training the model
  for epoch in range(epochs):
    np.random.shuffle(sentences)
    cost = 0
    counter = 0
    t0 = datetime.now()
    for sentence in sentences:
      sentence = [w for w in sentence \
        if np.random.random() < (1 - p_drop[w])
      ]
      if len(sentence) < 2:
        continue

      randomly_ordered_positions = np.random.choice(
        len(sentence),
        size=len(sentence),
        replace=False,
      )

      
      for pos in randomly_ordered_positions:
        word = sentence[pos]
        context_words = get_context(pos, sentence, window_size)
        neg_word = np.random.choice(vocab_size, p=p_neg)
        targets = np.array(context_words)

        # Performing Gradient Descent
        c = sgd(word, targets, 1, learning_rate, W, V)
        cost += c
        c = sgd(neg_word, targets, 0, learning_rate, W, V)
        cost += c

      counter += 1
      if counter % 100 == 0:
        sys.stdout.write("processed %s / %s\r" % (counter, len(sentences)))
        sys.stdout.flush()
 
    dt = datetime.now() - t0
    print("epoch complete:", epoch, "cost:", cost, "dt:", dt)
    costs.append(cost)
    learning_rate -= learning_rate_delta # Updating the learning rate


  # Visualization
  plt.plot(costs)
  plt.show()


  # save the model
  if not os.path.exists(savedir):
    os.mkdir(savedir)

  with open('%s/word2idx.json' % savedir, 'w') as f:
    json.dump(word2idx, f)

  np.savez('%s/weights.npz' % savedir, W, V)
  return word2idx, W, V


def get_negative_sampling_distribution(sentences, vocab_size):
  word_freq = np.zeros(vocab_size)
  word_count = sum(len(sentence) for sentence in sentences)
  for sentence in sentences:
      for word in sentence:
          word_freq[word] += 1

  # Smoothening
  p_neg = word_freq**0.75

  #Normalization
  p_neg = p_neg / p_neg.sum()

 # assert(np.all(p_neg > 0))
  return p_neg


def get_context(pos, sentence, window_size):
  
  start = max(0, pos - window_size)
  end_  = min(len(sentence), pos + window_size)

  context = []
  for ctx_pos, ctx_word_idx in enumerate(sentence[start:end_], start=start):
    if ctx_pos != pos:
      context.append(ctx_word_idx)
  return context

# Gradient Descent function
def sgd(input_, targets, label, learning_rate, W, V):
  
  activation = W[input_].dot(V[:,targets])
  prob = sigmoid(activation)

  # Calculating the gradients and updating the parameters
  gV = np.outer(W[input_], prob - label) 
  gW = np.sum((prob - label)*V[:,targets], axis=1) 
  V[:,targets] -= learning_rate*gV
  W[input_] -= learning_rate*gW 
  cost = label * np.log(prob + 1e-10) + (1 - label) * np.log(1 - prob + 1e-10)
  return cost.sum()


def load_model(savedir):
  with open('%s/word2idx.json' % savedir) as f:
    word2idx = json.load(f)
  npz = np.load('%s/weights.npz' % savedir)
  W = npz['arr_0']
  V = npz['arr_1']
  return word2idx, W, V



def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
  V, D = W.shape
  print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
  for w in (pos1, neg1, pos2, neg2):
    if w not in word2idx:
      print("Sorry, %s not in word2idx" % w)
      return

  p1 = W[word2idx[pos1]]
  n1 = W[word2idx[neg1]]
  p2 = W[word2idx[pos2]]
  n2 = W[word2idx[neg2]]

  vec = p1 - n1 + n2

  distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
  idx = distances.argsort()[:10]
  best_idx = -1
  keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
  for i in idx:
    if i not in keep_out:
      best_idx = i
      break

  print("got: %s - %s = %s - %s" % (pos1, neg1, idx2word[best_idx], neg2))
  print("closest 10:")
  for i in idx:
    print(idx2word[i], distances[i])

  print("dist to %s:" % pos2, cos_dist(p2, vec))


def test_model(word2idx, W, V):
  
  idx2word = {i:w for w, i in word2idx.items()}

  for We in (W, (W + V.T) / 2):
    print("**********")

    analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)
    analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
    analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
    analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
    analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
    analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
    analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
    analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
    analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
    analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
    analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
    analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
    analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
    analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
    analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
    analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
    analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
    analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
    analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
    analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
    analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
    analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
    analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
    analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
    analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
    analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
    analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
   


if __name__ == '__main__':
  word2idx, W, V = train_model('w2v_model')
  test_model(word2idx, W, V)

