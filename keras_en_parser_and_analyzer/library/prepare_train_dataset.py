"""
This script is aimed to have
line_labels = {0: 'experience', 1: 'education', 2: 'skills'}
line_types = {0: 'content', 1: 'footer'}
"""
import pandas as pd
from nltk import sent_tokenize, word_tokenize
import nltk
import re

from typing import List

nltk_stop_words = nltk.corpus.stopwords.words('english')
# text_without_stop_words = [t for t in word_tokenize(text) if t not in nltk_stop_words]
# [t for t in tokens if t not in string.punctuation]

import numpy as np
import multiprocessing as mp

import string
import en_core_web_sm
from sklearn.base import TransformerMixin, BaseEstimator
from normalise import normalise

nlp = en_core_web_sm.load()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 variety="BrE",
                 user_abbrevs={},
                 n_jobs=1):
        """
        Text preprocessing transformer includes steps:
            1. Text normalization
            2. Punctuation removal
            3. Stop words removal
            4. Lemmatization

        variety - format of date (AmE - american type, BrE - british format)
        user_abbrevs - dict of user abbreviations mappings (from normalise package)
        n_jobs - parallel jobs to run
        """
        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series, *_):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        pool.close()
        pool.join()

        return data

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text: str):
        if not pd.isnull(text):
            # Removing newline symbols and url links
            text = text.replace('\n', ' ').replace('\\', ' ')
            regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
            text = re.sub(regex, '', text)
            text = text.strip(' ')
            text = re.sub('[^A-Za-z0-9]+', ' ', text)
            normalized_text = self._normalize(text)
            doc = nlp(normalized_text)
            removed_punct = self._remove_punct(doc)
            removed_stop_words = self._remove_stop_words(removed_punct)
            return self._lemmatize(removed_stop_words)

    def _normalize(self, text) -> str:
        # some issues in normalise package
        try:
            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))
        except:
            return text

    def _remove_punct(self, doc) -> List[str]:
        return [t for t in doc if t.text not in string.punctuation]

    def _remove_stop_words(self, doc) -> List[str]:
        return [t for t in doc if not t.is_stop]

    def _lemmatize(self, doc)-> str:
        return ' '.join([t.lemma_ for t in doc])

    def prepare_dataset(self, file_name: str) -> pd.DataFrame:
        """
        Loading csv file with three columns : work, education, skills
        Final classes are :
        line_labels = {0: 'experience', 1: 'education', 2: 'skills'}
        line_types = {0: 'header', 1: 'content', 2: 'footer'}

        """
        df_temp = pd.read_csv(file_name)
        df_temp.loc[:, 'experience'] = self.transform(df_temp['work'])
        df_temp.loc[:, 'education'] = self.transform(df_temp['education'])
        df_temp.loc[:, 'skills'] = self.transform(df_temp['skills'])
        df_temp = df_temp[['education', 'experience', 'skills']]
        df_temp.to_csv('Train_'+file_name, index=False)
        return df_temp

'''
raw_text = pd.Series(['Developed and launched the application from scratch. The application has more than 60k users \
and a rating is 4.7. In the top 20 in Finance charts (AppStore) in Kazakhstan', 'Web app, which lets users create educational plans by\
creating schedules of classes, tracking the number of units required to successfully graduate in time.'])
df = TextPreprocessor(n_jobs=-1).prepare_dataset('indeed_resumes_cleaned.csv')
print(df.shape)
'''


def prepare_train_txt_dataset(file: str):
    line_labels = {0: 'experience', 1: 'education', 2: 'skills'}
    mapper = {'experience': 'content', 'education': 'content', 'skills': 'footer'}
    line_types = {0: 'content', 1: 'footer'}
    df = pd.read_csv(file)
    columns = df.columns
    for c in columns:
        # remove null values
        series = df[c]
        series.dropna(inplace=True)
        series = series[~series.isnull()]
        series = series.reindex(range(series.shape[0]))
        type_s = pd.Series([mapper[c]]*series.shape[0])
        type_label = pd.Series([c]*series.shape[0])
        frame = {'line_type': type_s, 'line_label': type_label, 'line': series}
        temp_df = pd.DataFrame(frame)
        temp_df = temp_df[~temp_df['line'].isnull()]
        file = file.split('/')[-1]
        temp_df.to_csv(c+'_FINAL_'+file.strip('csv')+'txt', sep='\t', index=False)



#prepare_train_txt_dataset('./TrainDataset/Train_resume_data_cleaned.csv')
