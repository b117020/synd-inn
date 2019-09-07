# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 02:39:26 2019

@author: Devdarshan
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:34:25 2019

@author: Devdarshan
"""

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from ast import literal_eval
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle
#read data


#X_train, y_train = train['Issue'].values, train['Product'].values
#nltk.download('punkt')



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') 
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]') # take all words that contain characters other than 0-9,a-z,#,+
STOPWORDS = set(stopwords.words('english'))



#method to clean text
def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    #text = # lowercase text
    text =text.lower()
    #text = # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    #text = # delete symbols which are in BAD_SYMBOLS_RE from text
    text =  re.sub(BAD_SYMBOLS_RE, '', text)
    #text = # delete stopwords from text
    token_word=word_tokenize(text)
    filtered_sentence = [w for w in token_word if not w in STOPWORDS] # filtered_sentence contain all words that are not in stopwords dictionary
    lenght_of_string=len(filtered_sentence)
    text_new=""
    for w in filtered_sentence:
        if w!=filtered_sentence[lenght_of_string-1]:
             text_new=text_new+w+" " # when w is not the last word so separate by whitespace
        else:
            text_new=text_new+w
            
    text = text_new
    return text
STOPWORDS = set(stopwords.words('english')) 
#train test split

#encoding

#vectorisation

#print(Tfidf_vect.vocabulary_)

#print(Train_X_Tfidf)
# fit the training dataset on the NB classifier

with open('model.pkl', 'rb') as handle:
    model1 = pickle.load(handle)  
with open('vectorisation.pkl', 'rb') as handle:
    vect1 = pickle.load(handle)

with open('encodings.pkl', 'rb') as handle:
    Encoder = pickle.load(handle)  
#from sklearn.preprocessing import MultiLabelBinarizer

#series to list conversion
#Test_X = Test_X.tolist()

#mlb = MultiLabelBinarizer(classes=sorted(tag_counts.keys()))
#Train_Y = mlb.fit_transform(Train_Y)
'''
Test_X = []
query = input("enter query: ")

Test_X.insert(0,query)
Test_X = [text_prepare(x) for x in Test_X]
Test_X_Tfidf = vect1.transform(Test_X)
y_val_predicted_labels_tfidf = model1.predict(Test_X_Tfidf)
y_val_pred_inversed = Encoder.inverse_transform(y_val_predicted_labels_tfidf)
print(y_val_pred_inversed[0])
'''
#manual prediction
def manual_predict(query):
    #query = input("enter query: ")
    Test_X = []
    Test_X.insert(0,query)
    Test_X = [text_prepare(x) for x in Test_X]
    Test_X_Tfidf = vect1.transform(Test_X)
    y_val_predicted_labels_tfidf = model1.predict(Test_X_Tfidf)
    y_val_pred_inversed = Encoder.inverse_transform(y_val_predicted_labels_tfidf)
    return y_val_pred_inversed[0]
