from copy import deepcopy

from typing import Dict
import pandas as pd
from keras_en_parser_and_analyzer.library.classifiers.lstm import WordVecBidirectionalLstmSoftmax
import os

from keras_en_parser_and_analyzer.library.prepare_train_dataset import TextPreprocessor
from keras_en_parser_and_analyzer.library.utility.parser_rules import extract_name, extract_email, extract_sex, \
    extract_ethnicity, extract_objective, extract_mobile
from keras_en_parser_and_analyzer.library.utility.pdf_utils import pdf_to_text
from keras_en_parser_and_analyzer.library.utility.tokenizer_utils import word_tokenize


class ResumeParser(object):

    def __init__(self):
        self.line_label_classifier = WordVecBidirectionalLstmSoftmax()
        self.line_type_classifier = WordVecBidirectionalLstmSoftmax()
        self.email = None
        self.name = None
        self.sex = None
        self.ethnicity = None
        self.education = []
        self.objective = None
        self.mobile = None
        self.experience = []
        self.knowledge = []
        self.project = []
        self.meta = list()
        self.header = list()
        self.unknown = True
        self.raw = None

    def load_model(self, model_dir_path):
        self.line_label_classifier.load_model(model_dir_path=os.path.join(model_dir_path, 'line_label'))
        self.line_type_classifier.load_model(model_dir_path=os.path.join(model_dir_path, 'line_type'))

    @staticmethod
    def extract_education(label, text):
        if label == 'education':
            return text
        return None

    @staticmethod
    def extract_project(label, text):
        if label == 'project':
            return text
        return None

    @staticmethod
    def extract_knowledge(label, text):
        if label == 'knowledge':
            return text
        return None

    @staticmethod
    def extract_experience(label, text):
        if label == 'experience':
            return text
        return None

    def parse(self, texts, print_line=False):
        self.raw = texts
        proc = TextPreprocessor(n_jobs=-0)
        predictions = {'line': [], 'type': [], 'label':[]}
        for p in texts:
            if len(p) > 10:
                s = word_tokenize(p)
                original_line = deepcopy(p).lower()
                p = proc._preprocess_text(p)
                line_label = self.line_label_classifier.predict_class(sentence=p)
                line_type = self.line_type_classifier.predict_class(sentence=p)
                predictions['line'].append(p)
                unknown = True
                # Find if the line belongs to header
                name = extract_name(s, original_line)
                email = extract_email(s, original_line)
                sex = extract_sex(s, original_line)
                race = extract_ethnicity(s, original_line)
                education = self.extract_education(line_label, p)
                project = self.extract_project(line_label, p)
                experience = self.extract_experience(line_label, p)
                objective = extract_objective(s, p)
                knowledge = self.extract_knowledge(line_label, original_line)
                mobile = extract_mobile(s, original_line)
                if mobile or name or email or sex or race:
                    predictions['type'].append('header')
                    predictions['label'].append('personal')
                else:
                    predictions['type'].append(line_type)
                    predictions['label'].append(line_label)
                if name is not None:
                    self.name = name
                    unknown = False
                if email is not None:
                    self.email = email
                    unknown = False
                if sex is not None:
                    self.sex = sex
                    unknown = False
                if race is not None:
                    self.ethnicity = race
                    unknown = False
                if education is not None:
                    self.education.append(education)
                    unknown = False
                if knowledge is not None:
                    self.knowledge.append(knowledge)
                    unknown = False
                if project is not None:
                    self.project.append(project)
                    unknown = False
                if objective is not None:
                    self.objective = objective
                    unknown = False
                if experience is not None:
                    self.experience.append(experience)
                    unknown = False
                if mobile is not None:
                    self.mobile = mobile
                    unknown = False

                if line_type == 'meta':
                    self.meta.append(p)
                    unknown = False
                if line_type == 'header':
                    self.header.append(p)

                if unknown is False:
                    self.unknown = unknown
        return predictions

    def detect_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        taking results from parse()
        Detecting header - personal data if found
        Data frame in the form:
        | line  | type  |   label |
        -----------------------
        |'same' | header | personal|
        """
        df.at[:, 'has_date'] = False
        date_idxs = []
        for i, row in df.iterrows():
            if has_date(row['line']):
                date_idxs.append(i)
                df.at[i, 'has_date'] = True
        return df

    def define_header_lines(self, df_predictions: pd.DataFrame):
        """
        If predictions contain personal/header information label all prior rows as personal
        """
        header_limit_percent = 0.1
        max_allowed_header_idx = int(df_predictions.shape[0]*header_limit_percent)
        personal_indexes = df_predictions[df_predictions['label'] == 'personal'].index
        personal_indexes = [i for i in personal_indexes if i <= max_allowed_header_idx]
        if len(personal_indexes)>0:
            last_idx = personal_indexes[-1]
            df_predictions.iloc[:last_idx, :][['type', 'label']] = ('header', 'personal')
        return df_predictions


def read_pdf(file:str, collected=None):
    if collected is None:
        collected = dict()
    if os.path.isfile(file):
        txt = None
        if file.lower().endswith('.pdf'):
            txt = pdf_to_text(file)
        if txt is not None and len(txt) > 0:
            collected[file] = txt
    return collected


def detect_date(token: str):
    """
    Attempts to convert string to date if found
    
    mask_test = r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+(\d{4})'

    """
    from re import search
    from dateutil.parser import parse
    from dateutil.parser._parser import ParserError
    token = token.lower()
    # Feb 2010
    mask1 = r"((jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec))\s([1-9]|([12][0-9])|(3[04]))"
    mask4 = r"((january)|(february)|(march)|(april)|(may)|(june)|(july)|(august)|(september)|(october)|(november)|(december))\s([1-9]|([12][0-9])|(3[04]))"
    date1 = search(mask1, token)
    # 12-09-1991
    mask2 = r'(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]|(?:january|march|may|july|august|october|december)))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2]|(?:january|march|april|may|january|july|august|september|october|november|december))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)(?:0?2|(?:february))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9]|(?:january|february|march|april|may|june|july|august|september))|(?:1[0-2]|(?:october|november|december)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$'
    date2 = search(mask2, token)
    # 09/2020, 09-2020, 09.2020 dates
    mask3 = r'[0-9]{2}(-|.|/)[0-9]{4}'
    date3 = search(mask3, token)
    date4 = search(mask4, token)
    if date1 or date2 or date3 or date4:
        try:
            # Case with 1999 2003 is faulty -> gives mask as 99 2003
            if len(token.split(' ')[0]) == len(token.split(' ')[1]) == 4:
                token = token.split(' ')[0]
            date = parse(token).date()
            return date
        except Exception as e:
            print("Date Parse Error: {}".format(e))
            return None


def has_date(line: str) -> bool:
    tokens = line.split(' ')
    for i in range(len(tokens)-1):
        result = detect_date(' '.join(tokens[i:i+2]))
        if result is not None:
            return True
    return False


def find_date(line: str) -> str:
    for token in line.split(' '):
        result = detect_date(token)
        if result is not None:
            return result
    return None
