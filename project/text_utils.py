from typing import List
from s2base.scispacy_util import SciSpaCyParser

import textacy
import spacy
import os
import json
from collections import defaultdict
from collections import Counter as mset
import numpy as np
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import re

DECADE_REGEX = re.compile(r' (\d+)s ')

def identity_function(x):
    """Identity function, used for the tfdif vectorizer"""
    return x

def get_tfidf_vectorizer():
    """Get or create tfidf vectorizer"""
    vectorizer_path = os.path.abspath(os.path.join("project", "tfidf_titles.pkl"))
    if os.path.isfile(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = build_tfidf_vectorizer()
        joblib.dump(vectorizer, vectorizer_path)
    
    return vectorizer

def build_tfidf_vectorizer():
    """Build the tfidf vectorizer based on dataset titles"""
    kb_path = os.path.abspath(os.path.join("project", "data", "data_sets.json"))
    with open(kb_path) as kb_file_:
        kb = json.load(kb_file_)

    nlp = spacy.load("en_scispacy_core_web_sm")
    token_lists = []
    for dataset in tqdm(kb, desc="Tokenizing kb titles for tfidf"):
        title_text = text_preprocess(dataset["title"])
        title_doc = nlp(title_text)
        title_tokens = [t.text.lower() for t in title_doc]
        token_lists.append(title_tokens)
    
    tfidf = TfidfVectorizer(analyzer='word', tokenizer=identity_function, preprocessor=identity_function, token_pattern=None, norm='l2')
    tfidf = tfidf.fit(token_lists)
    return tfidf

def text_preprocess(text: str):
    """Preprocess text to remove punctuation and lower case"""
    text = textacy.preprocess.remove_punct(text)
    text = text.replace("\n", " ").replace("\t", " ").replace(",", " ").replace("|", " ")
    text = text.replace(":", " ").replace(".", " ").replace("\xad", " ").replace("\\", " ")
    text = DECADE_REGEX.sub(r' \1 ', text)
    text = text.lower().rstrip()
    text = ' '.join(text.split())
    return text

def strip_numbers(text: str):
    """Strip numbers from a piece of text"""
    text = text.replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "")
    text = text.replace("6", "").replace("7", "").replace("8", "").replace("9", "")
    return text

def get_substring_candidates(all_ids: List[int],
                             all_titles: List[str],
                             all_titles_tokenized: List[List[str]],
                             mention: str,
                             sentence: List[str],
                             stopwords: set,
                             scispacy_parser: SciSpaCyParser,
                             tfidf_vectorizer: TfidfVectorizer):
    """Get candidate datasets for a given mention, based on tfidf weighted token overlap with dataset titles

       @param all_ids: list of all dataset ids
       @param all_titles: list of all datase titles preprocessed
       @param all_titles_tokenizer: list of all titles tokenized
       @param mention: the mention text
       @param sentence: the sentence that the mention text came from
       @param stopwords: set of stopwords to filter out
       @param scispacy_parser: instance of a scispacy parser
       @param tfidf_vectorizer: an already fit tfidf build_tfidf_vectorizer
    """
    tokens = [t.text for t in scispacy_parser.scispacy_create_doc(mention)]
    candidate_ids = []
    candidate_set = set()
    dataset_id_to_scores = defaultdict(list)
    sentence = text_preprocess(' '.join(sentence)).split()
    tfidf_sentence = tfidf_vectorizer.transform([sentence])
    tfidf_mention = tfidf_vectorizer.transform([tokens])
    sentence_candidate_scores = [(id, sum([tfidf_sentence[0, tfidf_vectorizer.vocabulary_[token]] for token in sentence 
                                    if token not in stopwords and token in tfidf_vectorizer.vocabulary_ and token in title]))
                                        for id, title in zip(all_ids, all_titles_tokenized)]
    mention_candidate_scores = [(id, sum([tfidf_mention[0, tfidf_vectorizer.vocabulary_[token]] for token in tokens 
                                    if token not in stopwords and token in tfidf_vectorizer.vocabulary_ and token in title]))
                                        for id, title in zip(all_ids, all_titles_tokenized)]
    candidate_scores = [(mention_score[0], sentence_score[1]*mention_score[1]) 
                            for sentence_score, mention_score in zip(sentence_candidate_scores, mention_candidate_scores)]

    ids = [candidate[0] for candidate in candidate_scores]
    scores = [candidate[1] for candidate in candidate_scores]
    return ids, scores