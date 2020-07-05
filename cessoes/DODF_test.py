import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pandas as pd
import joblib

from predict import ner_model

df = pd.read_csv("data/cessoes.csv", sep=',')

# Preprocess sentence to separate commas, semi-colons, etc as independent tokens
def preprocess(sentence):
    sentence = sentence.replace(',', ' , ').replace(';', ' ; ').replace(':', ' : ').replace('. ', ' . ').replace('\n', ' ')
    if sentence[len(sentence)-2:] == '. ':
        sentence = sentence[:len(sentence)-2] + " ."
    return sentence.split()

# One text sample from cessoes.csv
txt = preprocess(df.iloc[0]['text'])

model = ner_model("cessoes_ner.pkl")
entities = model.prediction(txt)
print(entities)
