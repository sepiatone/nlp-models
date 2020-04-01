"""
utility functions used across models
"""


import os
import numpy as np
import torch


def get_dataset(dataset):
  try:
    dir_data = "../data"
    os.makedirs(dir_data, exist_ok = False)
  except OSError:
    print("data directory already present")
  
  if dataset == "en_vi_iwslt_15":
    os.system("wget -nv -O ../data/train.en https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en")
    os.system("wget -nv -O ../data/train.vi https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi")
    os.system("wget -nv -O ../data/tst2013.en https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en")
    os.system("wget -nv -O ../data/tst2013.vi https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi")
    os.system("wget -nv -O ../data/vocab.en https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.en")
    os.system("wget -nv -O ../data/vocab.vi https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.vi")

  else:
    print("incorrect dataset specified")
    

# read in a text file
# if the type is "sentence", return a list of sentences, if the type is "word" return a list of lists (words in sentences)
def read_file_txt(filename, encoding = "utf-8", type = "sentence"):
  bindings = -1
  l_sentence = []
  
  with open(filename, "r", bindings, encoding) as f:      
    for line in f:
      if type == "word":
        l_sentence.append(line.strip().split())
      elif type == "sentence":
        l_sentence.append(line.strip())
      else:
        print("unknown type")
        break
  
  return l_sentence


def read_vocab_file(filename):
  with open(filename, "r") as f:
    return [line.strip() for line in f]


"""
prepare an one hot encoding of the items in obj
obj - list of items whose one hot encoding is to be prepared
vocab2id - ids of all items in the vocabulary
vocabulary - the vocabulary

returns a one hot vector of the items in obj
"""
def one_hot_encoding(obj, vocab2id, vocabulary, unk = "UNK"):
    # Example input `sent` (a list of words):
    # ['2', 'start', 'restaurants', 'with', 'inside', 'dining']

    one_hot = torch.zeros(len(obj), len(vocab2id))
    
    # list_words = [word for word in sent]

    for idx in range(len(obj)):
        if obj[idx] not in vocabulary:
            obj[idx] = unk
            
        one_hot[idx][vocab2id[obj[idx]]] = 1        

    return one_hot
  

"""
convert the items in obj to indexes
obj - list of items which have to be converted to indexes
obj2idx - indexes of all items in the vocabulary

returns a list of indexes corresponding to the items in obj
"""
def encoding_idx(obj, tag2id):
    l_idx = torch.zeros(len(obj), dtype = torch.long)
       
    list_tags = [tag for tag in obj]


    # print(id_seq.shape, len(list_tags))
    
    for idx in range(len(obj)):
        l_idx[idx] = tag2id[list_tags[idx]]
        assert  list_tags[idx] == obj[idx]

    return l_idx


MAX_SENT_LENGTH = 48
MAX_SENT_LENGTH_PLUS_SOS_EOS = 50

# We only keep sentences that do not exceed 48 words, so that later when we add <s> and </s> to a sentence it still won't exceed 50 words.
def filter_data(src_sentences_list, trg_sentences_list, max_len = MAX_SENT_LENGTH):
  new_src_sentences_list, new_trg_sentences_list = [], []
  
  for src_sent, trg_sent in zip(src_sentences_list, trg_sentences_list):
    if (len(src_sent) <= max_len and len(trg_sent) <= max_len and len(src_sent) > 0 and len(trg_sent)) > 0:
      new_src_sentences_list.append(src_sent)
      new_trg_sentences_list.append(trg_sent)
  
  return new_src_sentences_list, new_trg_sentences_list
  
  
"""
used to load the dataset. the class MTDataset is built on top of the data loader api provided by pytorch
"""
from torch.utils import data

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# These IDs are reserved.
PAD_INDEX = 0
UNK_INDEX = 1
SOS_INDEX = 2
EOS_INDEX = 3


class MTDataset(data.Dataset):
  def __init__(self, src_sentences, src_vocabs, trg_sentences, trg_vocabs,
               sampling=1., max_seq_length = MAX_SENT_LENGTH_PLUS_SOS_EOS):
    self.src_sentences = src_sentences[:int(len(src_sentences) * sampling)]
    self.trg_sentences = trg_sentences[:int(len(src_sentences) * sampling)]

    self.max_src_seq_length = max_seq_length
    self.max_trg_seq_length = max_seq_length

    self.src_vocabs = src_vocabs
    self.trg_vocabs = trg_vocabs

    self.src_v2id = {v : i for i, v in enumerate(src_vocabs)}
    self.src_id2v = {val : key for key, val in self.src_v2id.items()}
    self.trg_v2id = {v : i for i, v in enumerate(trg_vocabs)}
    self.trg_id2v = {val : key for key, val in self.trg_v2id.items()}

  def __len__(self):
    return len(self.src_sentences)

  def __getitem__(self, index):
    src_sent = self.src_sentences[index]
    src_len = len(src_sent) + 2   # add <s> and </s> to each sentence
    src_id = []
    for w in src_sent:
      if w not in self.src_vocabs:
        w = '<unk>'
      src_id.append(self.src_v2id[w])
    src_id = ([SOS_INDEX] + src_id + [EOS_INDEX] + [PAD_INDEX] *
              (self.max_src_seq_length - src_len))

    trg_sent = self.trg_sentences[index]
    trg_len = len(trg_sent) + 2
    trg_id = []
    for w in trg_sent:
      if w not in self.trg_vocabs:
        w = '<unk>'
      trg_id.append(self.trg_v2id[w])
    trg_id = ([SOS_INDEX] + trg_id + [EOS_INDEX] + [PAD_INDEX] *
              (self.max_trg_seq_length - trg_len))

    return torch.tensor(src_id), src_len, torch.tensor(trg_id), trg_len

  
import torch.nn.functional as F

class SimpleLossCompute:
  """A simple loss compute and train function."""

  def __init__(self, generator, criterion, opt = None):
    self.generator = generator
    self.criterion = criterion
    self.opt = opt

  def __call__(self, x, y, norm):
    x = self.generator(x)
    loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
    loss = loss / norm

    if self.opt is not None:  # training mode
      loss.backward()          
      self.opt.step()
      self.opt.zero_grad()

    return loss.data.item() * norm


def plot_perplexity(perplexities):
  fig, ax = plt.subplots()
  ax.plot(perplexities)
  ax.set_xlabel("Epoch")
  ax.set_ylabel("Perplexity")
  ax.set_title("Perplexity per Epoch");
