import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
import PyPDF2
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import os
from langdetect import detect
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt') 
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import streamlit as st

def translate_to_english(path_to_pdf:str) -> str:
    '''
    Args:
            path_to_pdf(str): path to the pdf to be translated

    Output:
            translateText(str): pdf content translated to english
    '''

    Text=''
    for foldername,subfolders,files in os.walk(r"C:\\Users\\Hp\Desktop\\Project final\\Shane app\\assets\\uploadedfiles"):
        for file in files:
            # open the pdf file
            object = PyPDF2.PdfReader(os.path.join(foldername,file))

            # get number of pages
            NumPages = len(object.pages)

            
            # extract text and do the search
            for i in range(0, NumPages):
                # PageObj = object.getPage(i)
                PageObj = object.pages[i]
                print("this is page " + str(i)) 
                TextTemp = PageObj.extract_text()
                TextTemp = re.sub('[^A-Za-z0-9]+]'," ", TextTemp)
                if detect(TextTemp)!='en':
                    translated_text = GoogleTranslator(source='auto', target='en').translate(TextTemp)
                    Text+=translated_text
                else:
                    Text+=TextTemp
    return Text
    

def remove_stopwords(sen):
  stop_words = stopwords.words('english')
  sen_new = " ".join([i for i in sen if i not in stop_words])
  return sen_new

def preprocess(translated_text:str) -> str:
  
  sentences =[]
  sentences.append(sent_tokenize(translated_text))
  
  sentences = [y for x in sentences for y in x] # flatten the list
  clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z0-9]+]", " ")# remove punctuations, numbers and special characters

  # make alphabets lowercase
  clean_sentences = [s.lower() for s in clean_sentences]
  stop_words = stopwords.words('english')

  clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

  return sentences, clean_sentences

def sen_to_vectors(clean_sentences:str) -> list:
  word_embeddings = {}
  
  f = open('glove.6B.100d.txt', encoding='utf-8')
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      word_embeddings[word] = coefs
  f.close()

  sentence_vectors = []
  for i in clean_sentences:
    if len(i) != 0:
      v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
      v = np.zeros((100,))
    sentence_vectors.append(v)

  return sentence_vectors

def get_ranked_sentences(sentences, sentence_vectors):
  
  sim_mat = np.zeros([len(sentences), len(sentences)])
  for i in range(len(sentences)):
    for j in range(len(sentences)):
      if i != j:
        sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    nx_graph = nx.from_numpy_array(sim_mat)
  scores = nx.pagerank(nx_graph)
  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

  return ranked_sentences

def generate_summary(pdf:str) -> str:
  summary = ''
  
  sn = 8
  try: 
    english_text = translate_to_english(pdf)
    sent, clean_sent = preprocess(english_text)
    sent_vect = sen_to_vectors(clean_sent)
    ranked_sent = get_ranked_sentences(sent, sent_vect)

    # Generate summary
    for i in range(sn):
      summary += ranked_sent[i][1]

    return summary
  except Exception as e:
     st.error(f"Error generating: {e}")
     return None