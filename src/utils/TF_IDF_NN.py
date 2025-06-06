# -*- coding: utf-8 -*-
"""
Disease Detection using Symptoms and Treatment recommendation
Combines TF-IDF, Cosine Similarity, and Neural Networks with Adversarial Training
"""

import os
import pickle
import re
import math
import operator
import warnings
from time import time
from statistics import mean
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import neural_structured_learning as nsl

# Suppress warnings
warnings.simplefilter("ignore")
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'
DATA_DIR = 'data'
COMB_DATASET = 'dis_sym_dataset_comb.csv'
NORM_DATASET = 'dis_sym_dataset_norm.csv'

class DiseaseDiagnosis:
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
        self.splitter = RegexpTokenizer(r'\w+')
        
        # Load datasets
        self.df_norm = self._load_dataset(NORM_DATASET)
        self.df_comb = self._load_dataset(COMB_DATASET)
        
        # Initialize variables
        self.documentname_list = list(self.df_norm['label_dis'])
        self.columns_name = list(self.df_norm.columns[1:])
        self.N = len(self.df_norm)
        self.M = len(self.columns_name)
        
        # Precompute TF-IDF
        self._compute_tf_idf()
        
    def _load_dataset(self, filename):
        """Load dataset from CSV file"""
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return pd.read_csv(path)
    
    def _compute_tf_idf(self):
        """Compute TF-IDF matrices"""
        # IDF calculation
        self.idf = {}
        for col in self.columns_name:
            temp = np.count_nonzero(self.df_norm[col])
            self.idf[col] = np.log(self.N / temp) if temp != 0 else 0
        
        # TF calculation
        self.tf = {}
        for i in range(self.N):
            for col in self.columns_name:
                key = (self.documentname_list[i], col)
                self.tf[key] = self.df_norm.loc[i, col]
        
        # TF-IDF calculation
        self.tf_idf = {}
        for i in range(self.N):
            for col in self.columns_name:
                key = (self.documentname_list[i], col)
                self.tf_idf[key] = float(self.idf[col]) * float(self.tf[key])
        
        # TF-IDF vector
        self.D = np.zeros((self.N, self.M), dtype='float32')
        for i in self.tf_idf:
            sym = self.columns_name.index(i[1])
            dis = self.documentname_list.index(i[0])
            self.D[dis][sym] = self.tf_idf[i]
    
    @staticmethod
    def cosine_dot(a, b):
        """Calculate cosine similarity between two vectors"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def gen_vector(self, tokens):
        """Generate query vector for TF-IDF"""
        Q = np.zeros(self.M)
        counter = Counter(tokens)
        
        for token in np.unique(tokens):
            tf = counter[token]
            try:
                idf_temp = self.idf[token]
                ind = self.columns_name.index(token)
                Q[ind] = tf * idf_temp
            except (KeyError, ValueError):
                continue
        return Q
    
    def tf_idf_score(self, k, query):
        """Calculate TF-IDF scores"""
        query_weights = {}
        for key in self.tf_idf:
            if key[1] in query:
                query_weights[key[0]] = query_weights.get(key[0], 0) + self.tf_idf[key]
        
        return sorted(query_weights.items(), key=lambda x: x[1], reverse=True)[:k]
    
    def cosine_similarity(self, k, query):
        """Calculate cosine similarity scores"""
        query_vector = self.gen_vector(query)
        d_cosines = [self.cosine_dot(query_vector, d) for d in self.D]
        out = np.array(d_cosines).argsort()[-k:][::-1]
        
        return {lt: float(d_cosines[lt]) for lt in set(out)}
    
    def synonyms(self, term):
        """Get synonyms from Thesaurus.com and WordNet"""
        synonyms = set()
        
        # Get from Thesaurus.com
        try:
            response = requests.get(f'https://www.thesaurus.com/browse/{term}', timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")
            container = soup.find('section', {'class': 'MainContentContainer'})
            if container:
                row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'})
                if row:
                    for li in row.find_all('li'):
                        synonyms.add(li.get_text())
        except Exception:
            pass
        
        # Get from WordNet
        for syn in wordnet.synsets(term):
            synonyms.update(syn.lemma_names())
            
        return synonyms
    
    def preprocess_symptoms(self, symptoms_str):
        """Preprocess user input symptoms"""
        user_symptoms = symptoms_str.lower().split(',')
        processed = []
        
        for sym in user_symptoms:
            sym = sym.strip()
            sym = sym.replace('-', ' ').replace("'", "")
            sym = ' '.join([self.lemmatizer.lemmatize(word) 
                           for word in self.splitter.tokenize(sym)])
            processed.append(sym)
        
        return processed
    
    def expand_symptoms(self, symptoms):
        """Expand symptoms using synonyms"""
        expanded = []
        for sym in symptoms:
            sym_words = sym.split()
            str_sym = set()
            
            for comb in range(1, len(sym_words)+1):
                for subset in combinations(sym_words, comb):
                    subset = ' '.join(subset)
                    str_sym.update(self.synonyms(subset))
            
            str_sym.add(sym)
            expanded.append(' '.join(str_sym).replace('_', ' '))
        
        return expanded
    
    def match_symptoms(self, user_symptoms, dataset_symptoms):
        """Match user symptoms with dataset symptoms"""
        found = set()
        for idx, data_sym in enumerate(dataset_symptoms):
            data_sym_split = data_sym.split()
            for user_sym in user_symptoms:
                count = sum(1 for symp in data_sym_split if symp in user_sym.split())
                if count/len(data_sym_split) > 0.5:
                    found.add(data_sym)
        return list(found)
    
    def get_related_symptoms(self, selected_symptoms):
        """Find related symptoms based on co-occurrence"""
        dis_list = set()
        final_symp = []
        counter_list = []
        
        for symp in selected_symptoms:
            final_symp.append(symp)
            dis_list.update(set(self.df_norm[self.df_norm[symp]==1]['label_dis']))
        
        for dis in dis_list:
            row = self.df_norm.loc[self.df_norm['label_dis'] == dis].values.tolist()[0][1:]
            for idx, val in enumerate(row):
                if val != 0 and self.columns_name[idx] not in final_symp:
                    counter_list.append(self.columns_name[idx])
        
        return Counter(counter_list)
    
    def build_model(self):
        """Build neural network model"""
        inputs = keras.Input(shape=(len(self.columns_name),), 
                    dtype=tf.float32, 
                    name=IMAGE_INPUT_NAME)
        
        x = layers.Dense(1000, activation='relu', 
                        use_bias=True,
                        kernel_initializer=initializers.HeNormal())(inputs)
        x = layers.Dense(1000, activation='relu', 
                        use_bias=True,
                        kernel_initializer=initializers.HeNormal())(x)
        
        outputs = layers.Dense(len(self.documentname_list), 
                             activation='softmax')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='DiseaseDiagnosisModel')
    
    def train_model(self):
        """Train the neural network model"""
        Y = pd.get_dummies(self.df_comb['label_dis']).values
        X = self.df_comb.drop(columns=['label_dis']).values
        
        # Base model
        base_model = self.build_model()
        base_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
            ModelCheckpoint('best_model.h5', 
                          monitor='val_accuracy',
                          save_best_only=True)
        ]
        
        history = base_model.fit(X, Y,
                               validation_split=0.2,
                               epochs=20,
                               callbacks=callbacks)
        
        # Adversarial model
        adv_config = nsl.configs.make_adv_reg_config(
            multiplier=0.2,
            adv_step_size=0.0001
        )
        
        adv_model = nsl.keras.AdversarialRegularization(
            base_model,
            label_keys=[LABEL_INPUT_NAME],
            adv_config=adv_config
        )
        
        adv_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        
        train_data = {IMAGE_INPUT_NAME: X, LABEL_INPUT_NAME: Y}
        adv_model.fit(train_data,
                     validation_split=0.2,
                     epochs=15,
                     callbacks=callbacks)
        
        return base_model, adv_model

if __name__ == "__main__":
    try:
        diagnoser = DiseaseDiagnosis()
        
        # Example usage
        symptoms_input = "fever, headache, cough"
        processed = diagnoser.preprocess_symptoms(symptoms_input)
        expanded = diagnoser.expand_symptoms(processed)
        
        print("Expanded symptoms:", expanded)
        
        # Train models
        base_model, adv_model = diagnoser.train_model()
        
    except Exception as e:
        print(f"Error initializing DiseaseDiagnosis: {str(e)}")