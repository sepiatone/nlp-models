import pandas as pd
import re

# col_names = ["en", "fr", "attrib"]
# data_pd = pd.read_csv("fra-eng.txt", sep = "\t", names = col_names)
# 
# print(data_pd.head(10))
# print(data_pd.shape)
# print(data_pd.tail(10))
# 
# data_en_pd = data_pd[col_names[0]]
# data_fr_pd = data_pd[col_names[1]]
# 
# data_en_pd.to_csv("en.txt", header = False, index = False)
# data_fr_pd.to_csv("fr.txt", header = False, index = False)

# data_pd = pd.read_csv("en.txt", header = None, names = ["word"])
# 
# print(data_pd.head(5))
# print(data_pd.shape)
# 
# data_pd = pd.DataFrame(data_pd["word"].str.strip())
# 
# print(data_pd.head(5))
# print(data_pd.shape)
# 
# print(type(data_pd["word"].str))
# data_tmp = re.findall(r'\w+', data_pd["word"])
# print(type(data_pd))
# print(type(pd.DataFrame(data_tmp)))
# data_tmp = pd.DataFrame(data_tmp)
# print(data_tmp.head(5))



def create_vocab(filename_1, filename_2, encoding = "utf-8"):
  bindings = -1
  l_word = []
  
  with open(filename_1, "r", bindings, encoding) as f:      
    for line in f:
        line_tmp = line.strip().split()
        
        for idx in range(len(line_tmp)):
            l_word.append(line_tmp[idx])
      
  f.close()
  
  with open(filename_2, "w", encoding = "utf-8") as f:
    for idx in range(len(l_word)):
        # print(l_word[idx].encode("utf-8"))
        f.write(l_word[idx])
        f.write("\n")
  
  f.close()

filename_1 = "en.txt"
filename_2 = "en_vocab.txt"
  
create_vocab(filename_1, filename_2)

filename_1 = "fr.txt"
filename_2 = "fr_vocab.txt"

# create_vocab(filename_1, filename_2)