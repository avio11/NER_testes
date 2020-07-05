import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib
import pandas as pd

class ner_model():
    def __init__(self, model_path):
        # Load trained NER model
        self.model = joblib.load(model_path)

    def get_features(self, sentence):
        sent_features = []
        for i in range(len(sentence)):
            word_feat = {
                'word': sentence[i].lower(),
                'capital_letter': sentence[i][0].isupper(),
                'all_capital': sentence[i].isupper(),
                'isdigit': sentence[i].isdigit(),
                'word_before': sentence[i].lower() if i==0 else sentence[i-1].lower(),
                'word_after:': sentence[i].lower() if i+1>=len(sentence) else sentence[i+1].lower(),
                'BOS': i==0,
                'EOS': i==len(sentence)-1
            }
            sent_features.append(word_feat)
        return sent_features

    def prediction(self, sentence):
        if isinstance(sentence[0], list):
            feats = []
            for i in range(len(sentence)):
                feats.append(self.get_features(sentence[i]))
            predictions = self.model.predict(feats)
            return predictions
        else:
            feats = self.get_features(sentence)
            predictions = self.model.predict_single(feats)
            return self.dataFramefy(sentence, predictions)
            print("DFfy")
        # return predictions

    # TODO create dataframe with identified entities
    def dataFramefy(self, sentence, prediction):
        # Create dictionary of tags to save predicted entities
        tags = self.model.classes_
        tags.remove('O')
        tags.sort(reverse=True)
        while(tags[0][0] == 'I'):
            del tags[0]
        for i in range(len(tags)):
            tags[i] = tags[i][2:]
        dict = {}
        for i in tags:
            dict[i] = []

        last_tag = 'O'
        temp_entity = []
        for i in range(len(prediction)):
            if prediction[i] != last_tag and last_tag != 'O':
                dict[last_tag[2:]].append(temp_entity)
                temp_entity = []
            if prediction[i] == 'O':
                last_tag = 'O'
                continue
            else:
                temp_entity.append(sentence[i])
                last_tag = "I" + prediction[i][1:]

        if temp_entity:
            tags[last_tag[2:]].append(temp_entity)

        return dict
