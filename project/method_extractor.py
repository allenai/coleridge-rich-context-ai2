import os
import json
import re
from s2base import scispacy_util
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from create_sgtb_dataset import get_scispacy_doc
from math import log

class MethodExtractor():
    def __init__(self, train_path, dev_path, sage_methods_path, leipzig_word_counts_path):
        # path to the data folder for the train set
        self.train_path = train_path

        # path to the data folder for the dev set
        self.dev_path = dev_path

        # read the list of sage methods and prepare a regex to match them.
        sage_method_entries = json.load(open(sage_methods_path, mode='rt'))["@graph"]
        method_names = []
        for entry in sage_method_entries:
            if "skos:prefLabel" in entry:
                method_names.append(entry["skos:prefLabel"]["@value"])
            if "skos:altLabel" in entry:
                if type(entry["skos:altLabel"]) == list:
                    for label in entry["skos:altLabel"]:
                        method_names.append(label["@value"])
                else:
                    method_names.append(entry["skos:altLabel"]["@value"])                    
        # lowercase and remove duplicates.
        method_names = [name for name in set([name.lower() for name in method_names])]
        # remove very short names.
        method_regexes = [re.escape(method_name) for method_name in method_names]
        methods_regex_string = r'\b(?P<method_name>' + '|'.join(method_regexes) + r')\b'
        # to debug the regex: print(methods_regex_string)
        self.sage_methods_regex = re.compile(methods_regex_string, re.IGNORECASE)
                            
        # set of english stopwords
        self._stopwords =  set(stopwords.words('english'))

        # an instance of a scispacy parser
        self._scispacy_parser = scispacy_util.SciSpaCyParser()

        # read word counts in the Leipzig corpus.
        self._read_leipzig_word_counts_file(leipzig_word_counts_path)
        
    def _read_leipzig_word_counts_file(self, leipzig_word_counts_path):
        """Read word counts in a background corpus. This can be useful for estimating
           the confidence of extracted terms. Leipzig corpora are available for download
           at http://wortschatz.uni-leipzig.de/en/download/
        """
        self._background_word_counts = defaultdict(int)
        self._background_lowercased_word_counts = defaultdict(int)
        with open(leipzig_word_counts_path, mode='rt') as word_count_file:
            for line in word_count_file.readlines():
                splits = line.strip().split('\t')
                assert len(splits) == 4
                WORD_INDEX = 1
                FREQUENCY_INDEX = 3
                frequency = int(splits[3])
                self._background_word_counts[splits[1]] = frequency
                self._background_lowercased_word_counts[splits[1].lower()] += frequency
                
    def predict_from_publications_list(self, publication_list, predict_path):
        """Predict datasets for a list of publications, with each publication
           in the format provided in publications.json file.

           @param publication_list: the result of json.load('publications.json')
        """
        citation_list = []
        for publication in tqdm(publication_list, desc='predict methods'):            
            spacy_doc = get_scispacy_doc(predict_path, str(publication["publication_id"]), self._scispacy_parser)
            predictions = self.predict(publication["publication_id"], spacy_doc)
            citation_list += predictions

        return citation_list

    def is_sentence(self, spacy_sentence):
        """
        Checks if the string is an English sentence.
        :param sentence: spacy string
        :return: True / False
        """
        tokens = [t for t in spacy_sentence]
        
        # Minimum number of words per sentence
        MIN_TOKEN_COUNT = 6
        if len(tokens) < MIN_TOKEN_COUNT:
            return False
        
        # Most tokens should be words
        MIN_WORD_TOKENS_RATIO = 0.5
        if sum([t.is_alpha for t in tokens]) / len(tokens) < MIN_WORD_TOKENS_RATIO:
            return False

        text = spacy_sentence.text

        # A sentence has to end with a period
        if not text.strip().endswith('.'):
            return False
        
        # Most characters should be letters, not numbers and not special characters
        MIN_LETTER_CHAR_RATIO = 0.5
        if sum([c.isalpha() for c in text]) / len(text) < MIN_LETTER_CHAR_RATIO:
            return False
        
        return True
    
    def predict(self, pub_id, doc, debug=False):
        """Reads the text file of a publication, extracts methods, and returns 
           a list of dict objects such as: 
           { "publication_id": 876, "method": "opinion poll", "score": 0.680 }

           @param txt_file_path: path to the publication text file.
        """                                                                                        
        suffix_pattern = re.compile(r'\b(?P<method_name>([A-Z][-\w]+ )+([Aa]nalysis|[Mm]odel|[Tt]heory))\b')
        regex_list = [ self.sage_methods_regex, suffix_pattern ]
        # This dictionary maps the lowercased version of an extracted method to its original case,
        # which simultaneously removes embarrassing duplicates and report an easier to read casing
        # for human evaluation.
        methods_dict = {}
        methods_to_contexts = defaultdict(list)
        all_text = ' '.join([t.text for t in doc])
        all_text_lowercased = all_text.lower()
        for sent in doc.sents:
            if not self.is_sentence(sent): continue
            tokens = [t.text for t in sent]
            sent_str = ' '.join(tokens)            
            for regex in regex_list:
                for match in re.finditer(regex, sent_str):
                    match_str = match.group('method_name')
                    if match_str.lower() not in methods_dict:

                        # skip matches which include stopwords.
                        stopwords_count = len([token for token in match_str.lower().split() if token in self._stopwords])
                        if stopwords_count > 0: continue

                        # skip matches which appear in sentences with title casing.
                        all_non_stopwords_start_with_capital_letters = True
                        for token in sent_str.split():
                            if token in self._stopwords: continue
                            if len(token) > 0 and token[0].islower(): 
                                all_non_stopwords_start_with_capital_letters = False
                                break
                        if all_non_stopwords_start_with_capital_letters: continue
                            
                        methods_dict[match_str.lower()] = match_str
                        # record the context in which this match was found for debugging purposes.
                        methods_to_contexts[match_str.lower()].append(sent_str)
                        
        # filter out short method names, all lower case, single tokens.
        MIN_CHAR_COUNT = 3
        MIN_TOKEN_COUNT = 2
        methods_dict = { method_lower : method_name for method_lower, method_name in methods_dict.items() \
                         if len(method_name) >= MIN_CHAR_COUNT \
                         and len(method_name.split()) >= MIN_TOKEN_COUNT \
                         and method_name != method_lower }

        # score and prepare output.
        output = []
        for method_lower, method_name in methods_dict.items():
            # compute confidence score based on background frequencies, capitalization, length, term frequency.
            term_frequency = all_text_lowercased.count(method_lower) + 1.
            assert term_frequency > 0
            method_lowercased_tokens = [token.lower() for token in method_name.split()]
            background_frequencies = [self._background_lowercased_word_counts[token] for token in method_lowercased_tokens]
            min_background_frequency = min(background_frequencies) + 1
            capitalization_multiplier = 2. if method_name[0].isupper() else 1.
            length_multiplier = 0.5 if len(method_lowercased_tokens) < 2 else 1.
            score = length_multiplier * capitalization_multiplier * log(term_frequency) / (1. + log(min_background_frequency))
            MIN_THRESHOLD_FOR_METHODS = 0.2
            if score < MIN_THRESHOLD_FOR_METHODS:
                continue

            # normalize score
            score = log(1+score)
            if score > 1.0: score = 1.0
            record = { 'publication_id': int(pub_id), 'method': method_name, 'score': round(score, 2) }
            
            if debug:
                assert len(methods_to_contexts[method_lower]) > 0
                record['contexts'] = methods_to_contexts[method_lower]

            output.append(record)
        return output
                
    def _filter_references_section(self, text):
        """Helper function to return the text with the references section stripped out.
           It is probably not perfect, but it looks for the last instance of 'reference'
           and removes all text after it

           @param text: the text to filter the references section from
        """
        references_pattern = r"(REFERENCE)|(reference)|(Reference)"
        references_found = [i.start() for i in re.finditer(references_pattern, text)]
        if references_found != []:
            last_reference_index = references_found[-1]
            return text[:last_reference_index]
        else:
            return text

