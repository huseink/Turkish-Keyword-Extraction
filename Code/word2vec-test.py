from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import pandas as pd
import nltk
import xml.etree.ElementTree as ET
import os
import gensim
path = 'Stemming/Makale Kokleri/Biyoloji.xml'
ozetceler = []
tree = ET.parse(path)
root = tree.getroot()
for ozetce in root:
    ozetce = ozetce.text
    ozetce = str(ozetce)
    ozetceler.append(ozetce)
# print(ozetceler[0])
kelimeler = []
for i in ozetceler:
    kelimeler.append(i.split())

model =Word2Vec(kelimeler, vector_size =100,window = 10, min_count= 1, sg = 0)
model.wv["Türkiye"]
print(model.wv.most_similar("Türkiye"))

#! test kodu 




#print(model)

