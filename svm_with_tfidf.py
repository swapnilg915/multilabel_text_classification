import re
import os
import traceback
import json
import time
import numpy as np
from scipy import sparse
from scipy.stats import uniform
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import pickle
import itertools
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from cleaner import TextCleaner
text_cleaner_obj = TextCleaner()


class MultilabelTopicClassifier(object):

    def __init__(self):
        # self.clf = OneVsRestClassifier(SVC(class_weight="balanced",probability=True))
        self.tfidf_vect = TfidfVectorizer(sublinear_tf=True, use_idf=True, max_df=0.5, min_df=2)
        self.algo = LogisticRegression(class_weight="balanced")
        # self.algo = SGDClassifier(class_weight="balanced")
        self.clf = OneVsRestClassifier(self.algo)
        # self.clf = SVC(class_weight="balanced")
        # self.clf = svc = OneVsRestClassifier(LinearSVC())
        self.model_dir = "trained_models"
        self.make_dir(self.model_dir)        
    

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print("\n directory created for path : ",path)

    
    def save_models(self):
        model_dir = self.model_dir
        self.make_dir(model_dir)
        models_path = os.path.join(model_dir, "svm_multilabel.pickle")
        tfidf_vect_path = os.path.join(model_dir, "tfidf_vect_multilabel.pickle")
        multilabel_binarizer_path = os.path.join(model_dir, "svm_multilabel_binarizer.pickle")
        pickle.dump(self.clf, open(models_path, 'wb'))
        pickle.dump(self.tfidf_vect, open(tfidf_vect_path, 'wb'))
        pickle.dump(self.multilabel_binarizer, open(multilabel_binarizer_path, 'wb'))
        

    def read_models(self):
        model_dir = self.model_dir
        data = defaultdict()
        data["clf"] = pickle.load(open(os.path.join(model_dir, "svm_topic_clf_multilabel.pickle"), 'rb'))
        data["tfidf_vect"] = pickle.load(open(os.path.join(model_dir, "tfidf_vect_multilabel.pickle"), 'rb'))
        data["binarizer"] = pickle.load(open(os.path.join(model_dir, "svm_topic_clf_multilabel_binarizer.pickle"), 'rb'))
        return data


    def remove_quotes(self, text):
        return text.replace("'","").replace('"', '')


    def get_cleaned(self, lst):
        return [self.remove_quotes(word) for word in lst if self.remove_quotes(word)]


    def add_description_and_title(self, dic):
        return str(dic["Description"]) + str(dic["Title"])


    def read_data(self):
        with open("datasets/movie_multilabel_train.json", "r", encoding="utf-8") as fs:
            train_data = json.load(fs)

        with open("datasets/movie_multilabel_testing.json", "r", encoding="utf-8") as fs:
            test_data = json.load(fs)
        return train_data["TrainingData"], test_data


    def add_description_and_title(self, dic):
        return str(dic["Description"]) + str(dic["Title"])


    def transform_data(self, train_data, test_data):
        """ remove the objects in data if topics are not present or either of the description or title is not present """
        train_data = [dic for dic in train_data if (self.get_cleaned(dic["Topics"])) and ((text_cleaner_obj.cleaning_pipeline(dic["Description"]) or text_cleaner_obj.cleaning_pipeline(dic["Title"])) and dic["Topics"])]
        train_sentences = [self.add_description_and_title(dic) for dic in train_data]
        train_labels = [self.get_cleaned(dic["Topics"]) for dic in train_data]
        unique_labels = list(set([self.remove_quotes(topic) for dic in train_data for topic in dic["Topics"]]))
        
        test_data = [dic for dic in test_data if (self.get_cleaned(dic["Topics"])) and ((text_cleaner_obj.cleaning_pipeline(dic["Description"]) or text_cleaner_obj.cleaning_pipeline(dic["Title"])) and dic["Topics"])]
        test_sentences = [self.add_description_and_title(dic) for dic in test_data]
        test_labels = [self.get_cleaned(dic["Topics"]) for dic in test_data]

        return train_sentences, train_labels, unique_labels, test_sentences, test_labels


    def trainModel(self):
        """ read data """
        train_data, test_data = self.read_data()
        train_sentences, train_labels, unique_labels, test_sentences, test_labels = self.transform_data(train_data, test_data)

        en_time = time.time()
        train_vector = self.tfidf_vect.fit_transform(train_sentences)
        test_vector = self.tfidf_vect.transform(test_sentences)
        print("\n total time for sentence encoding : ", time.time() - en_time)

        # data["count_feature_names"] = self.count_vect.get_feature_names()
        self.multilabel_binarizer = MultiLabelBinarizer(classes = unique_labels)
        self.multilabel_binarizer.fit(train_labels)
        train_labels = self.multilabel_binarizer.fit_transform(train_labels)
        # self.multilabel_binarizer.fit(test_labels)
        test_labels = self.multilabel_binarizer.fit_transform(test_labels)

        #################### step 4 - train model / classifier
        self.clf.fit(train_vector, train_labels)
        print("\n Training score : ",self.clf.score(train_vector, train_labels))

        pred_vector = self.clf.predict(test_vector)
        print("\n Testing score :",accuracy_score(test_labels, pred_vector))
        print("\n f1 score micro: ", f1_score(test_labels, pred_vector, average="micro"))
        print("\n f1 score macro: ", f1_score(test_labels, pred_vector, average="macro"))
        self.save_models()



    def get_top_labels(self, data, pred_class):
        clf_labels_all = [(lab, float(round(score,2))) for lab, score in zip(data["binarizer"].classes_, pred_class[0])]
        clf_labels = [(lab, float(round(score,2))) for lab, score in zip(data["binarizer"].classes_, pred_class[0]) if score >= self.threshold_1]
        final_labels_lst = sorted(clf_labels, key=lambda item: item[1], reverse=True)
        return data


    def test(self, data):
        data = self.read_models(data)
        tfidf_vector = self.use_obj(data["Description"]).vector
        pred_class = data["clf"].predict_proba([tfidf_vector])
        data = self.get_top_labels(data, pred_class)
        # labels_score_list = [(lab, score) for lab, score in zip(binarizer.classes_, pred_class[0])]
        # final_labels = {tpl[0].lower():float(round(tpl[1],2)) for tpl in labels_score_list}
        return data


if __name__ == '__main__':
    obj = MultilabelTopicClassifier()
    obj.trainModel()

