"""
utility functions used across models
"""


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
