import xml.etree.ElementTree as ET
import os
import json

keywords = []
abstracts = []
articles = []

keywordPath = 'keywords/'
abstractPath = 'stopwords_filtered/'

keywordsTag = 'Anahtar'
abstractTag = 'Özetçe'

outputJsonFilePath = 'articles/articles.json'

def traverse(path,attribute):
    tree = ET.parse(path)
    root = tree.getroot()
    
    for event in root.findall(attribute):
        textToAppend = event.text.strip()
        textToAppend = textToAppend.replace(',',' ')
        if(attribute == keywordsTag):
            keywords.append(textToAppend)
        else:
            abstracts.append(textToAppend)

def constructArticles(fileName):
    traverse(keywordPath + fileName,keywordsTag)
    traverse(abstractPath + fileName, abstractTag)
    for i in range(len(abstracts)):
        articles.append({"Ozetce": abstracts[i], "Anahtar": keywords[i]})  

directory = r'stopwords_filtered'
for fileName in os.listdir(directory):
    constructArticles(fileName)

with open(outputJsonFilePath, 'w') as outfile:
    json.dump(articles, outfile)