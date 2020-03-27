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
    

def read_sentence_file(filename):
  sentences_list = []
  
  with open(filename, "r") as f:      
    for line in f:
      sentences_list.append(line.strip().split())
  
  return sentences_list


def read_vocab_file(filename):
  with open(filename, "r") as f:
    return [line.strip() for line in f]
  

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
