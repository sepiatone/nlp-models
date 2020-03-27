"""
utility functions used across models
"""


import os


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
