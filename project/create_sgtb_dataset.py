"""This file can be run to create the dataset linking dataset in the format expected by the
   Structured Gradient Tree Boosting model. It assumes that there are train/dev/test folders
   in project/data. We use this dataset for other downstream models as well, to minimize the
   number of times this dataset is regenerated. The create_dataset_input function can be used
   with any list of candidates.
"""

import json
import os
import numpy as np
from sklearn.externals import joblib
from s2base import scispacy_util
from spacy.matcher import Matcher
from spacy.vocab import Vocab
from spacy.tokens import Doc
from tqdm import tqdm
import re
import scipy
import math
from nltk.corpus import stopwords
import argparse

YEAR_PATTERN = r"((?<![.|\d])\d\d\d\d(?![\d]))"
SECTION_STRINGS = ["background", "methods", "results", "abstract", "intro", "introduction",
                   "keywords", "objectives", "conclusion", "measures", "discussion", "method",
                   "references", "contribution", "data"]
CONTEXT_WORDS = ["data", "information", "respondents", "survey"]
NLTK_STOPWORDS = stopwords.words("english")

def get_scispacy_doc(data_folder_path, publication_id, scispacy_parser):
    """Get spacy doc if cached, otherwise create it

       @param data_folder_path: path to the data folder
       @param publication_id: the publication id of the doc to load/create
       @param scispacy_parser: the SciSpaCyParser instance to use
    """
    spacy_doc_dir = os.path.join(data_folder_path, "spacy_tokenized_texts")
    # disabled caching because it wasn't working
    #if not os.path.exists(spacy_doc_dir):
    #    os.makedirs(spacy_doc_dir)

    spacy_doc_path = os.path.join(spacy_doc_dir, str(publication_id) + ".spacy")
    # try to get the spacy document from the cached location, otherwise create it
    # this is useful because processing a document with spacy is slow
    doc = None
    if os.path.exists(spacy_doc_dir):
        try:
            doc = Doc(scispacy_parser.nlp.vocab).from_disk(spacy_doc_path)
        except:
            pass
    if not doc:
        with open(os.path.join(data_folder_path, "input", "files", "text", str(publication_id) + ".txt"), mode='rt') as txt_fp:
            pub_text = txt_fp.read()
        doc = scispacy_parser.scispacy_create_doc(pub_text)
        # disabled caching because it wasn't workin
        #doc.to_disk(spacy_doc_path)
    return doc

def get_years_from_text(text):
    """Parses a set of candidate years included in text

       @param text: the text to search for years in
    """
    matches_year = re.findall(YEAR_PATTERN, text)
    hyphens = r'(?:\-+|\—+|\-|\–|\—|\~)'
    matches_range = re.findall(r"(?<!/)\d\d\d\d{h}\d\d\d\d(?!/)".format(h=hyphens), text)
    years_found = set([int(match) for match in matches_year])

    # also include all years in any range of years found
    for year_range in matches_range:
        try:
            start, end = re.split(hyphens, year_range)
        except:
            print("Failed to split:", year_range)
            continue
        for year in range(int(start), int(end)+1):
            years_found.add(year)

    # filter candidates years to be between 1000 and 2019
    filtered_years_found = set()
    for year in years_found:
        if not (year < 1000 or year > 2019):
            filtered_years_found.add(year)

    return filtered_years_found

def compute_entity_probabilities():
    """Computes p(dataset) based on the training set
    """
    train_path = os.path.abspath(os.path.join("project", "data", "train"))
    train_labels_path = os.path.join(train_path, "data_set_citations.json")
    with open(train_labels_path) as train_labels_file:
        train_labels_json = json.load(train_labels_file)

    dataset_id_to_count = {}
    set_of_docs = set()
    for label in train_labels_json:
        dataset_id = label["data_set_id"]
        publication_id = label["publication_id"]
        if dataset_id in dataset_id_to_count:
            dataset_id_to_count[dataset_id] += 1
        else:
            dataset_id_to_count[dataset_id] = 1
        set_of_docs.add(publication_id)

    total_docs = len(set_of_docs)
    # normalize at log scale to highlight differences between datasets that appear once or twice, and datasets that appear zero times
    normalized_dataset_id_to_count = {dataset_id: -1*np.log(dataset_id_to_count[dataset_id]/total_docs) for dataset_id in dataset_id_to_count}
    return normalized_dataset_id_to_count

def compute_entity_given_mention_probs():
    """Computes p(dataset|mention) based on the training set
    """
    train_path = os.path.abspath(os.path.join("project", "data", "train"))
    train_labels_path = os.path.join(train_path, "data_set_citations.json")
    with open(train_labels_path) as train_labels_file:
        train_labels_json = json.load(train_labels_file)

    mention_to_entity_to_count = {}
    for label in train_labels_json:
        dataset_id = label["data_set_id"]
        publication_id = label["publication_id"]
        mention_list = label["mention_list"]
        for mention in mention_list:
            if mention in mention_to_entity_to_count:
                if dataset_id in mention_to_entity_to_count[mention]:
                    mention_to_entity_to_count[mention][dataset_id] += 1
                else:
                    mention_to_entity_to_count[mention][dataset_id] = 1
            else:
                mention_to_entity_to_count[mention] = {dataset_id: 1}

    normalized_mention_to_entity_to_count = {}
    for mention in mention_to_entity_to_count:
        # normalize entity probabilities for each mention text
        entity_to_count = mention_to_entity_to_count[mention]
        total_count = sum([entity_to_count[entity] for entity in entity_to_count])
        normalized_entity_to_count = {entity: entity_to_count[entity]/total_count for entity in entity_to_count}
        normalized_mention_to_entity_to_count[mention] = normalized_entity_to_count
    return normalized_mention_to_entity_to_count

def compute_mention_given_entity_probs():
    """Computes p(mention|dataset) based on the training set
    """
    train_path = os.path.abspath(os.path.join("project", "data", "train"))
    train_labels_path = os.path.join(train_path, "data_set_citations.json")
    with open(train_labels_path) as train_labels_file:
        train_labels_json = json.load(train_labels_file)

    entity_to_mention_to_count = {}
    for label in train_labels_json:
        dataset_id = label["data_set_id"]
        publication_id = label["publication_id"]
        mention_list = label["mention_list"]
        for mention in mention_list:
            if dataset_id in entity_to_mention_to_count:
                if mention in entity_to_mention_to_count[dataset_id]:
                    entity_to_mention_to_count[dataset_id][mention] += 1
                else:
                    entity_to_mention_to_count[dataset_id][mention] = 1
            else:
                entity_to_mention_to_count[dataset_id] = {mention: 1}

    normalized_entity_to_mention_to_count = {}
    for entity in entity_to_mention_to_count:
        # normalize mention probabilities for each dataset id
        mention_to_count = entity_to_mention_to_count[entity]
        total_count = sum([mention_to_count[mention] for mention in mention_to_count])
        normalized_mention_to_count = {mention: mention_to_count[mention]/total_count for mention in mention_to_count}
        normalized_entity_to_mention_to_count[entity] = normalized_mention_to_count
    return normalized_entity_to_mention_to_count

def year_match_nearby(mention_contexts, kb_entry):
    """Searches the mention contexts for a year-like string in the title of dataset.
       Returns 0 if there are no years in the dataset title, otherwise 1 for a match
       and -1 for no match.

       @param mention_contexts: list of mention contexts
       @param kb_entry: the dataset entry from the knowledge base
    """
    feature_value = None
    years_in_dataset_name = get_years_from_text(kb_entry["title"])
    if len(years_in_dataset_name) == 0:
        return 0
    for context in mention_contexts:
        years_in_context = set()
        for sentence in context[0]:
            years_in_sentence = get_years_from_text(sentence.text)
            years_in_context = years_in_context.union(years_in_sentence)
        if len(years_in_context.intersection(years_in_dataset_name)) > 0:
            return 1
    return -1

def get_contextual_similarity(candidate_dataset_id,
                              kb_entry,
                              mention_contexts,
                              scispacy_parser,
                              glove):
    """Computes contextual similarity scores between the candidate dataset description and
       the mention contexts using glove embeddings and cosine similarity.

       @param candidate_dataset_id: the id of the candidate dataset
       @param kb_entry: the knowledge base entry for the candidate dataset
       @param mention_contexts: a list of mention contexts to compute similarity over
       @param scispacy_parser: a scispacy parser
       @param glove: a dictionary of glove word embeddings
    """
    glove_dim = 50
    bins = np.linspace(0, 1, 11)
    num_bins = bins.shape[0]

    description = kb_entry["description"]
    if description == "":
        return [0]*num_bins, [0]*num_bins
    
    description = scispacy_parser.scispacy_create_doc(description)

    # try both max pooling and average pooling of word embeddings to get sentence representation
    embedded_description_max = []
    embedded_description_avg = []
    for sentence in description.sents:
        tokens = [t.text.lower() for t in sentence]
        glove_tokens = [t for t in tokens if t in glove]
        embedded_sentence = [np.linalg.norm(glove[t], ord=2) for t in glove_tokens if t not in NLTK_STOPWORDS]
        # embedded_sentence = [embedding*idf_dict[t] if t in idf_dict else embedding*idf_dict["<MAX_VALUE>"] for embedding, t in zip(embedded_sentence, glove_token)]
        last_embedding_layer = embedded_sentence
        if last_embedding_layer == []:
            continue
        embedded_description_max.append(np.max(last_embedding_layer, axis=0))
        embedded_description_avg.append(np.mean(last_embedding_layer, axis=0))

    # try both max pooling and average pooling of word embeddings to get sentence representation
    embedded_contexts_max = []
    embedded_contexts_avg = []
    for context in mention_contexts:
        embedded_context_max = []
        embedded_context_avg = []
        for sentence in context[0]:
            tokens = [t.text.lower() for t in sentence]
            glove_tokens = [t for t in tokens if t in glove]
            embedded_sentence = [np.linalg.norm(glove[t], ord=2) for t in glove_tokens if t not in NLTK_STOPWORDS]
            # embedded_sentence = [embedding*idf_dict[t] if t in idf_dict else embedding*idf_dict["<MAX_VALUE>"] for embedding, t in zip(embedded_sentence, glove_token)]
            last_embedding_layer = embedded_sentence
            if last_embedding_layer == []:
                continue
            embedded_context_max.append(np.max(last_embedding_layer, axis=0))
            embedded_context_avg.append(np.mean(last_embedding_layer, axis=0))
        embedded_contexts_max.append(embedded_context_max)
        embedded_contexts_avg.append(embedded_context_avg)

    cosine_distances_max = []
    cosine_distances_avg = []
    for context_max, context_avg in zip(embedded_contexts_max, embedded_contexts_avg):
        for sentence_max, sentence_avg in zip(context_max, context_avg):
            for description_max, description_avg in zip(embedded_description_max, embedded_description_avg):
                max_cosine = scipy.spatial.distance.cosine(sentence_max, description_max)
                avg_cosine = scipy.spatial.distance.cosine(sentence_avg, description_avg)
                if not math.isnan(max_cosine):
                    cosine_distances_max.append(max_cosine)

                if not math.isnan(avg_cosine):
                    cosine_distances_avg.append(avg_cosine)

    # bin the similarity scores of description sentence and context sentence pairs
    digitized_max = np.digitize(cosine_distances_max, bins)
    digitized_avg = np.digitize(cosine_distances_avg, bins)

    binned_max = [0]*num_bins
    binned_avg = [0]*num_bins
    # use a one hot representation with a one for the largest similarity bin that has a pair in it
    binned_max[max(digitized_max)-1] = 1
    binned_avg[max(digitized_avg)-1] = 1

    return binned_max, binned_avg

def max_min_sentence_length(mention_contexts):
    """Get the max and min lengths of the sentence in which the mention text was found

       @param mention_contexts: a list of mention contexts
    """
    max_len = 0
    min_len = float('inf')
    for context in mention_contexts:
        # select the sentence in the context that contains the mention text
        sentence_idx = context[3]
        sentence = context[0][sentence_idx]
        if len(sentence) > max_len:
            max_len = len(sentence)
        if len(sentence) < min_len:
            min_len = len(sentence)

    return max_len, min_len

def get_section_features(spacy_doc, mention_context, section_matcher):
    """Get a one hot representation of which of a set of predefined section headers is
       closest before the first instance of the mention text. Each feature maps to a particular
       section header, except the last feature, which indicates that no section header was found
       prior to the mention text.

       @param spacy_doc: a spacy doc representation of the publication
       @param mention_context: a tuple representing the mention context
       @param section_matcher: a spacy matcher to match section headers followed by
                               new lines or colons
    """
    start_token_idx = mention_context[2][0]
    if start_token_idx == 0:
        return [0]*len(SECTION_STRINGS) + [1]
    doc_to_search = spacy_doc[:start_token_idx]
    matches = list(section_matcher(doc_to_search.as_doc()))
    if matches == []:
        return [0]*len(SECTION_STRINGS) + [1]
    matches = sorted(matches, key=lambda match: match[1], reverse=True)
    closest_match = matches[0]
    closest_section = section_matcher.vocab.strings[closest_match[0]]
    features = [0]*len(SECTION_STRINGS) + [0]
    features[SECTION_STRINGS.index(closest_section)] = 1
    return features

def context_word_overlap_count(mention_contexts, kb_entry):
    """Returns the count of overlapping words between mention contexts and the subjects
       field of the dataset's knowledge base entry

       @param mention_contexts: a list of mention contexts
       @param kb_entry: the knowledge base entry for the candidate dataset
    """
    subjects = re.split(",| ", kb_entry["subjects"])
    subjects = set([subject.lower() for subject in subjects])
    subjects.update(CONTEXT_WORDS)
    total_count = 0
    for context in mention_contexts:
        for sentence in context[0]:
            tokens = set([t.text.lower() for t in sentence])
            total_count += len(subjects.intersection(tokens))

    return total_count

def featurize_candidate_datasets(is_test,
                                 row,
                                 gold_dataset_id,
                                 prior_entity_probs,
                                 entity_to_prob,
                                 prior_mention_given_entity_probs,
                                 mention_text,
                                 dataset_id_to_kb_entry,
                                 mention_contexts,
                                 scispacy_parser,
                                 glove,
                                 spacy_doc,
                                 section_matcher):
    """Featurizes the list of dataset candidates, adding a null candidate, and outputs
       the expected format for sgtb

       @param is_test: if test data is being processed
       @param row: the row from the rule based candidates
       @param gold_dataset_id: the id of hte gold dataset for this mention
       @param prior_entity_probs: dictionary mapping dataset id to empirical probability
       @param entity_to_prob: dictionary containing dataset id to p(dataset id | mention)
       @param prior_mention_given_entity_probs: dictionary containing p(mention|dataset)
       @param mention_text: the text of the mention
       @param dataset_id_to_kb_entry: dictionary mapping dataset id to its entry in the knowledge base
       @param mention_contexts: list of mention contexts for this mention
       @param scispacy_parser: a scispacy parser
       @param glove: a dictionary of glove embeddings
       @param spacy_doc: a spacy tokenized doc
       @param section_matcher: a spacy matcher to match section headers followed by
                               new lines or colons
    """
    candidate_datasets = []
    for candidate_dataset_id in row["candidate_dataset_ids"]:
        candidate_dataset_id = str(candidate_dataset_id)
        label = int(candidate_dataset_id == gold_dataset_id)
        if int(candidate_dataset_id) in prior_entity_probs:
            prior_entity_probability = prior_entity_probs[int(candidate_dataset_id)]
        else:
            prior_entity_probability = 0

        if int(candidate_dataset_id) in entity_to_prob:
            prior_entity_given_mention_prob = entity_to_prob[int(candidate_dataset_id)]
        else:
            prior_entity_given_mention_prob = 0

        if int(candidate_dataset_id) in prior_mention_given_entity_probs:
            if mention_text in prior_mention_given_entity_probs[int(candidate_dataset_id)]:
                prior_mention_given_entity_prob = prior_mention_given_entity_probs[int(candidate_dataset_id)][mention_text]
            else:
                prior_mention_given_entity_prob = 0
        else:
            prior_mention_given_entity_prob = 0
        year_match = year_match_nearby(mention_contexts, dataset_id_to_kb_entry[int(candidate_dataset_id)])
        # the contextual similarity features were not found to improve performance and are slow to compute
        # contextual_similarity = get_contextual_similarity(candidate_dataset_id,
        #                                                   dataset_id_to_kb_entry[int(candidate_dataset_id)],
        #                                                   mention_contexts,
        #                                                   scispacy_parser,
        #                                                   glove)

        is_null_candidate = 0
        mention_length_chars = len(mention_text)
        mention_length_tokens = len(scispacy_parser.scispacy_create_doc(mention_text))
        max_len, min_len = max_min_sentence_length(mention_contexts)
        is_acronym = int(mention_text.isupper() and len(mention_text.split(' ')) == 1)
        section_features = get_section_features(spacy_doc, mention_contexts[0], section_matcher)
        context_word_overlap = context_word_overlap_count(mention_contexts, dataset_id_to_kb_entry[int(candidate_dataset_id)])
        # avoid log(0)
        context_word_overlap += 1
        context_word_overlap = np.log(context_word_overlap)

        # tfidf score for the ner produced candidates
        if "score" in row:
            score = row["score"]
        else:
            score = -1

        feats = [prior_entity_probability,
                 prior_entity_given_mention_prob,
                 prior_mention_given_entity_prob,
                 year_match,
                 mention_length_chars,
                 mention_length_tokens,
                 max_len,
                 min_len,
                 is_acronym] + \
                 section_features + \
                 [context_word_overlap,
                 score] + \
                 [is_null_candidate]
                 # contextual_similarity[0] + \
                 # contextual_similarity[1]
        candidate_datasets.append([(str(candidate_dataset_id), label), feats])

    # add a null candidate to every list of entity candidates for SGTB training
    null_is_null_candidate = 1
    if is_test:
        null_label = 0
    else:
        null_label = int(gold_dataset_id == "NULL")
    null_feats = [0]*(len(feats)-1) + [null_is_null_candidate]
    null_candidate = [("NULL", null_label), null_feats]
    candidate_datasets.append(null_candidate)
    return candidate_datasets

def create_output_mention(is_test,
                          row,
                          prior_entity_probs,
                          prior_entity_given_mention_probs,
                          mention_text,
                          prior_mention_given_entity_probs,
                          dataset_id_to_kb_entry,
                          mention_contexts,
                          scispacy_parser,
                          glove,
                          spacy_doc,
                          section_matcher):
    """Creates the output format and features for a single mention and its candidates

       @param is_test: if test data is being processed
       @param row: the row from the rule based candidates
       @param prior_entity_probs: dictionary mapping dataset id to empirical probability
       @param prior_entity_given_mention_probs: dictionary containing p(dataset|mention)
       @param mention_text: the text of the mention
       @param prior_mention_given_entity_probs: dictionary containing p(mention|dataset)
       @param dataset_id_to_kb_entry: dictionary mapping dataset id to its entry in the knowledge base
       @param mention_contexts: list of mention contexts for this mention
       @param scispacy_parser: a scispacy parser
       @param glove: a dictionary of glove embeddings
       @param spacy_doc: a spacy tokenized doc
       @param section_matcher: a spacy matcher to match section headers followed by
                               new lines or colons
    """
    if not is_test:
        gold_dataset_id = str(row["dataset_id"])
    else:
        gold_dataset_id = "NULL"

    # Note: dummy values to match SGTB expected format, but it is not actually used anywhere
    offset = [1, 2]

    if mention_text in prior_entity_given_mention_probs:
        entity_to_prob = prior_entity_given_mention_probs[mention_text]
    else:
        entity_to_prob = {}
    candidate_datasets = featurize_candidate_datasets(is_test,
                                                      row,
                                                      gold_dataset_id,
                                                      prior_entity_probs,
                                                      entity_to_prob,
                                                      prior_mention_given_entity_probs,
                                                      mention_text,
                                                      dataset_id_to_kb_entry,
                                                      mention_contexts,
                                                      scispacy_parser,
                                                      glove,
                                                      spacy_doc,
                                                      section_matcher)
    output_mention = [(mention_text, offset, gold_dataset_id), candidate_datasets]
    return output_mention

def create_idf_weighting(scispacy_parser):
    """Build a dictionary of inverse document frequency weights for all tokens in training set

       @param scispacy_parser: a scispacy parser
    """
    data_folder_path = os.path.abspath(os.path.join("project", "data", "train"))
    train_path = os.path.abspath(os.path.join("project", "data", "train", "input", "files", "text"))
    token_dict = {}
    num_docs = 0
    for txt_file_name in tqdm(os.listdir(train_path), desc='create idf weights in create_sgtb_dataset.py'):
        pub_id = txt_file_name.split(".")[0]
        spacy_doc = get_scispacy_doc(data_folder_path, pub_id, scispacy_parser)
        tokens_set = set([t.text for t in spacy_doc])
        for t in tokens_set:
            if t in token_dict:
                token_dict[t] += 1
            else:
                token_dict[t] = 1

        num_docs += 1

    idf_dict = {t: np.log(num_docs/(token_dict[t])) for t in token_dict}
    idf_dict["<MAX_VALUE>"] = max(idf_dict.values())
    return idf_dict

def create_dataset_input(rule_based_candidates, mention_context_cache_path, data_folder_path, overall_output_path=None, is_test=False, output_path=None, overwrite_dataset=False):
    """Function to take in the rule based candidates and create
       the input format for the SGTB model. This function is intended
       to be used for processing test data, as the main function in
       this file will convert and save train, dev, and test output.

       @param rule_based_candidates: a list of candidates from the rule based model
       @param mention_context_cache_path: path to a dictionary mapping <pub_id>:<mention_text> pairs to all contexts
       @param data_folder_path: path to the data folder
       @param overall_output_path: path to the overall output folder (optional, used for SGTB training)
       @param is_test: parameter indicating whether or not the data being processed is test data
       @param output_path: the path to write the output to (if not processing test data)
       @param overwrite_dataset: whether or not to overwrite the existing dataset (will be true for train
                                 and false for dev and test)
    """
    scispacy_parser = scispacy_util.SciSpaCyParser()
    prior_entity_probs = compute_entity_probabilities()
    prior_entity_given_mention_probs = compute_entity_given_mention_probs()
    prior_mention_given_entity_probs = compute_mention_given_entity_probs()

    glove_path = os.path.abspath(os.path.join("project", "data", "glove.6B.50d.txt"))
    with open(glove_path, "r") as lines:
        glove = {line.split()[0]: np.array([float(value) for value in line.split()[1:]])
                    for line in lines}

    # I haven't run the experiments to tell if having a cache actually helps or not, it takes a while to load
    # the cache when it is used
    # if is_test:
    #     mention_context_cache = {}
    # else:
    #     try:
    #         print("Loading cache...")
    #         mention_context_cache = joblib.load(mention_context_cache_path)["cache"]
    #         print("Cache loaded...")
    #     except:
    #         mention_context_cache = {}
    mention_context_cache = {}

    kb_path = os.path.abspath(os.path.join("project", "data", "data_sets.json"))
    with open(kb_path) as kb_file:
        kb_json = json.load(kb_file)

    dataset_id_to_kb_entry = {}
    for dataset in kb_json:
        dataset_id_to_kb_entry[dataset["data_set_id"]] = dataset

    matcher = Matcher(scispacy_parser.nlp.vocab)
    section_matcher = Matcher(scispacy_parser.nlp.vocab)
    for section_name in SECTION_STRINGS:
        section_matcher.add(section_name, None, [{"LOWER": section_name}, {"ORTH": "\n"}],
                                                [{"LOWER": section_name}, {"ORTH": ":"}],
                                                [{"ORTH": "\n"}, {"LOWER": section_name}, {"ORTH": "."}])

    output_docs = []
    pub_ids = []
    # we will write a new file on the first document, and amend to it afterwards
    first_doc = True
    cache_changed = False
    for pub_id in tqdm(rule_based_candidates, desc='create dataset in create_sgtb_dataset.py'):
        spacy_doc = get_scispacy_doc(data_folder_path, pub_id, scispacy_parser)

        pub_ids.append(pub_id)
        doc_candidates = rule_based_candidates[pub_id]
        output_doc = []

        dataset_id_to_longest_mention_text = {}
        for row in doc_candidates:
            mention_text = row["mention"]
            dataset_id = row["candidate_dataset_ids"][0]
            if dataset_id in dataset_id_to_longest_mention_text:
                if len(mention_text) > len(dataset_id_to_longest_mention_text[dataset_id]):
                    dataset_id_to_longest_mention_text[dataset_id] = mention_text
            else:
                dataset_id_to_longest_mention_text[dataset_id] = mention_text

        for row in doc_candidates:
            mention_text = row["mention"]
            dataset_id = row["candidate_dataset_ids"][0]
            # if mention_text != dataset_id_to_longest_mention_text[dataset_id]:
            #     continue

            mention_context_cache_key = str(pub_id) + "_" + mention_text
            if mention_context_cache_key in mention_context_cache:
                mention_contexts = mention_context_cache[mention_context_cache_key]
            else:
                # search for the mention text in the doc
                spacy_mention_text = scispacy_parser.scispacy_create_doc(mention_text)
                
                pattern = []
                for token in spacy_mention_text:
                    pattern.append({"ORTH": token.text})

                matcher.add("MENTION", None, pattern)
                matches = list(matcher(spacy_doc))

                # build and save a mapping of <pub_id>_<mention_text> to all contexts the mention
                # is found in
                cache_changed = True
                mention_contexts = []
                token_idx_to_sent_idx = {}
                sentences_list = list(spacy_doc.sents)
                context_size = 3
                for sent_idx, sent in enumerate(sentences_list):
                    for token in sent:
                        token_idx = token.i
                        token_idx_to_sent_idx[token_idx] = sent_idx

                for match_id, start, end in matches:
                    sentence_idx = token_idx_to_sent_idx[start]
                    start_context_sent_idx = max(0, sentence_idx-context_size)
                    if start_context_sent_idx == 0:
                        match_sentence_idx = sentence_idx
                    else:
                        match_sentence_idx = context_size
                    end_context_sent_idx = min(len(sentences_list), sentence_idx+context_size)
                    mention_context = sentences_list[start_context_sent_idx:end_context_sent_idx+1]
                    sentences_as_docs = []
                    for sentence in mention_context:
                        sentences_as_docs.append(sentence.as_doc())

                    start_context_token_idx = sentences_list[start_context_sent_idx].start
                    end_context_token_idx = sentences_list[end_context_sent_idx-1].end
                    context_with_offsets = (sentences_as_docs, (start_context_token_idx, end_context_token_idx), (start, end), match_sentence_idx)
                    mention_contexts.append(context_with_offsets)

                # limit featurizing to first 3 contexts in order of appearance
                mention_contexts = mention_contexts[:3]
                mention_context_cache[mention_context_cache_key] = mention_contexts

                matcher.remove("MENTION")

            if mention_contexts != []:
                output_mention = create_output_mention(is_test,
                                                       row,
                                                       prior_entity_probs,
                                                       prior_entity_given_mention_probs,
                                                       mention_text,
                                                       prior_mention_given_entity_probs,
                                                       dataset_id_to_kb_entry,
                                                       mention_contexts,
                                                       scispacy_parser,
                                                       glove,
                                                       spacy_doc,
                                                       section_matcher)
                output_doc.append(output_mention)

        # only write output to file if not processing test data
        if not is_test:
            if first_doc:
                with open(output_path, "w") as output_file:
                    json.dump(output_doc, output_file)
                    output_file.write("\n")
                first_doc = False

                if overwrite_dataset:
                    with open(overall_output_path, "w") as overall_output_file:
                        json.dump(output_doc, overall_output_file)
                        overall_output_file.write("\n")
            else:
                with open(output_path, "a") as output_file:
                    json.dump(output_doc, output_file)
                    output_file.write("\n")

                with open(overall_output_path, "a") as overall_output_file:
                    json.dump(output_doc, overall_output_file)
                    overall_output_file.write("\n")

        output_docs.append(json.loads(json.dumps(output_doc)))

    # if cache_changed and not is_test:
    #     joblib.dump({"cache": mention_context_cache}, mention_context_cache_path)
    return output_docs, pub_ids

def create_dataset(rule_candidate_path, data_folder_path, overall_output_path, output_path, overwrite_dataset=False):
    """Function to take in the rule based candidates and write the input format
       for SGTB to a file

       @param rule_candidate_path: a path to the rule based candidates json file
       @param data_folder_path: path to the data folder
       @param overall_output_path: path to the overall output folder
       @param output_path: the path to write the output to (if not processing test data)
       @param overwrite_dataset: whether or not to overwrite the existing dataset (will be true for train
                                 and false for dev and test)
    """
    mention_context_cache_path = os.path.join(data_folder_path, "mention_context_cache.pkl")

    with open(rule_candidate_path) as candidate_file:
        rule_candidate_json = json.load(candidate_file)
    
    candidates = rule_candidate_json

    create_dataset_input(candidates, mention_context_cache_path, data_folder_path, overall_output_path, is_test=False, output_path=output_path, overwrite_dataset=overwrite_dataset)

def main(dataset_root: str, output_root: str, overall_output_path: str):
    train_data_folder_path = os.path.join(dataset_root, "train")
    train_candidate_path = os.path.join(dataset_root, "train", "all_candidates_scores.json")
    train_output_path = os.path.join(output_root, "sgtb_train_scores.json")

    dev_data_folder_path = os.path.join(dataset_root, "dev")
    dev_candidate_path = os.path.join(dataset_root, "dev", "all_candidates_scores.json")
    dev_output_path = os.path.join(output_root, "sgtb_dev_scores.json")

    test_data_folder_path = os.path.join(dataset_root, "test")
    test_candidate_path = os.path.join(dataset_root, "test", "all_candidates_scores.json")
    test_output_path = os.path.join(output_root, "sgtb_test_scores.json")

    create_dataset(train_candidate_path, train_data_folder_path, overall_output_path, output_path=train_output_path, overwrite_dataset=True)
    create_dataset(dev_candidate_path, dev_data_folder_path, overall_output_path, output_path=dev_output_path)
    create_dataset(test_candidate_path, test_data_folder_path, overall_output_path, output_path=test_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_root',
    )

    parser.add_argument(
        '--output_root'
    )

    parser.add_argument(
        '--overall_output_path'
    )

    args = parser.parse_args()
    main(args.dataset_root, args.output_root, args.overall_output_path)
