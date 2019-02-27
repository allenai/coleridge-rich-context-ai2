"""This file can be run to convert all of the publications in train/dev/test to conll format,
   both for NER and linking. The output will be in folder called ner-conll and linking-conll.
   to_conll_test.py is used to produce conll formatted files for the test publications.
"""

import argparse
import json
import os
from s2base import scispacy_util
from spacy.matcher import Matcher
from sklearn.externals import joblib
from tqdm import tqdm
import re

DATASET_ENTITY_TAG = "DATA"

class ConllParser():
    """Class for parsing the rich context documents into conll 2003 format using scispacy
    """
    def __init__(self, data_folder_path, scispacy_parser):
        print(data_folder_path)
        os.system(f'ls {data_folder_path}/test')
        # dictionary mapping publication id to a list of (dataset id, mention list) tuples
        self.publication_to_datasets_and_mentions = None

        # path to the train_test folder, formatted in the same way as the dev fold data folder
        self.data_folder_path = data_folder_path

        # path to the publications.json file containing the list of publications to process
        self.pub_path = os.path.join(self.data_folder_path, "publications.json")

        # path to the data_set_citations.json file containing the publication dataset pairs
        self.citations_path = os.path.join(self.data_folder_path, "data_set_citations.json")

        # path to the data_sets.json file containing the dataset kb
        self.datasets_path = os.path.join(self.data_folder_path, os.pardir, "data_sets.json")

        # path to the folder containing the publication text files
        self.text_files_path = os.path.join(self.data_folder_path, "input", "files", "text")

        # instance of the scispacy parser class
        self.scispacy_parser = scispacy_parser

    def build_publication_to_datasets_and_mentions(self):
        """Builds the dictionary mapping publication to datasets and mentions
        """
        publication_to_datasets_and_mentions = {}
        with open(self.citations_path) as json_citations_file:
            citation_pairs = json.load(json_citations_file)
            for pair in citation_pairs:
                pub_id = pair["publication_id"]
                if pub_id in publication_to_datasets_and_mentions:
                    publication_to_datasets_and_mentions[pub_id].append((pair["data_set_id"],
                                                                         pair["mention_list"]))
                else:
                    publication_to_datasets_and_mentions[pub_id] = [(pair["data_set_id"],
                                                                     pair["mention_list"])]
        self.publication_to_datasets_and_mentions = publication_to_datasets_and_mentions

    def build_match_index_to_tag(self, doc, datasets_and_mentions, pub_id):
        """Builds the dictionary mapping the index of a match to the appropriate entity tag

           @param doc: a spacy doc to process matches on
           @param datasets_and_mentions: a list of (dataset id, mention list) tuples for this doc
        """
        matcher = Matcher(self.scispacy_parser.nlp.vocab)

        # add each pattern to match on to the matcher with id DATASET_<dataset_id>
        for dataset_id, mention_list in datasets_and_mentions:
            patterns = []
            for mention in mention_list:
                replaced_mention = mention
                # some mentions had extra escaping
                replaced_mention = replaced_mention.replace("\\", "")
                # this was a special character that was not in the actual text
                replaced_mention = replaced_mention.replace("\u2afd", "")
                # -\n in the actual text ends up as -<space> in the mention text
                replaced_mention = re.sub(r"(?<=\S)-\s", "", replaced_mention)
                # this unicode hyphen ends up as \xad in the actual text
                replaced_mention = replaced_mention.replace("\u2013", "\xad")
                # this unicode character is actually fi in the text
                replaced_mention = replaced_mention.replace("\ufb01", "fi")

                # we try replacing different variants of \xad and a hyphen
                xad_1 = replaced_mention.replace("\xad-", "\xad")
                xad_2 = replaced_mention.replace("-\xad", "\xad")

                # we try adding an s at the end of mentions
                plural_1 = replaced_mention + "s"
                plural_2 = xad_1 + "s"
                plural_3 = xad_2 + "s"

                mentions_to_try = [replaced_mention, xad_1, xad_2, plural_1, plural_2, plural_3]

                for mention_to_try in mentions_to_try:
                    mention_doc = self.scispacy_parser.nlp(mention_to_try)
                    pattern = []
                    for t in mention_doc:
                        pattern.append({"ORTH": t.text})
                        # allow new lines to be between tokens
                        pattern.append({"ORTH": "\n", "OP": "*"})
                    patterns.append(pattern)
            matcher.add("DATASET_" + str(dataset_id), None, *patterns)

        matches = matcher(doc)
        length_matches = []
        for match in matches:
            length_matches.append((match[0], match[1], match[2], match[2] - match[1]))

        # sort matches by length to capture the longest matches first
        length_matches = sorted(length_matches, key=lambda tup: tup[3])
        # loop over matches to create a dictionary mapping index in the document to the entity tag
        match_index_to_tag = {}
        indices_matched = set()
        for match_id, start, end, _ in length_matches:
            for i in range(start, end):
                dataset_id = self.scispacy_parser.nlp.vocab.strings[match_id]
                if (i == start) and (i-1) in indices_matched and (not i in indices_matched):
                    match_index_to_tag[i] = "B-" + DATASET_ENTITY_TAG + ":" + dataset_id
                else:
                    match_index_to_tag[i] = "I-" + DATASET_ENTITY_TAG + ":" + dataset_id
                indices_matched.add(i)

        return match_index_to_tag

    def create_conll_line(self, token, match_index_to_tag):
        """Create one line of the output conll file

           @param token: the token for the line being created
           @param match_index_to_tag: the dictionary mapping token index to entity tag
        """
        word = token.text
        pos = token.pos_
        tag = "O"
        linking_tag = "_"
        if token.i in match_index_to_tag:
            entity_tag = match_index_to_tag[token.i].split(":")[0]
            linking_tag = match_index_to_tag[token.i].split(":")[1]
        else:
            entity_tag = "O"

        output_line = word + " " + pos + " " + tag + " " + entity_tag

        extraction_line = output_line
        linking_line = output_line + " " + linking_tag

        return extraction_line, linking_line

    def create_conll_sentence(self, sentence, match_index_to_tag):
        """Creates one sentence of the output conll file

           @param sentence: the spacy sentence for the sentence being created
           @param match_index_to_tag: the dictionary mapping token index to entity tag
        """
        extraction_sentence = ""
        linking_sentence = ""
        for token in sentence:
            # spacy includes space tokens, which we can safely ignore
            if token.pos_ == "SPACE":
                continue
            extraction_line, linking_line = self.create_conll_line(token, match_index_to_tag)
            extraction_sentence += extraction_line + "\n"
            linking_sentence += linking_line + "\n"

        return extraction_sentence, linking_sentence

    def create_conll_text(self, doc, match_index_to_tag):
        """Creates one document of conll output

           @param doc: the spacy doc to process
           @param match_index_to_tag: the dictionary mapping token index to entity tag
        """
        extraction_text = ""
        linking_text = ""
        prev_sent = None
        for sent in doc.sents:
            extraction_sentence, linking_sentence = self.create_conll_sentence(sent, match_index_to_tag)
            # conll format includes an extra new line between each sentence
            # we will omit the line (merge sentences) if an entity spans sentences due to a spacy
            # sentence splitting error
            strip_new_line = False
            if prev_sent and prev_sent.endswith("-DATA\n"):
                # if previous sentence ends with -DATA, search for the end of the first token in
                # the next sentence and see if it ends with -DATA
                for i in range(len(extraction_sentence)):
                    if extraction_sentence[i] == "\n" and extraction_sentence[i-5:i] == "-DATA":
                        strip_new_line = True
                        break

            if strip_new_line:
                extraction_text = extraction_text[:-1]

            extraction_text += extraction_sentence + "\n"
            linking_text += linking_sentence + "\n"
            prev_sent = extraction_sentence

        return extraction_text, linking_text

    def parse_publication(self, publication):
        """Parses one raw text file into conll format and writes to
           ../conll/<publication_id>_<extraction or linking>.conll

           @param publication: the json publication being processed
        """
        try:
            publication_id = publication["publication_id"]
            datasets_and_mentions = []
            if publication_id in self.publication_to_datasets_and_mentions:
                datasets_and_mentions = self.publication_to_datasets_and_mentions[publication_id]
            publication_text_path = os.path.join(self.text_files_path, str(publication_id) + ".txt")
            with open(publication_text_path) as publication_text_file:
                full_text = publication_text_file.read()
                doc = self.scispacy_parser.scispacy_create_doc(full_text)

                match_index_to_tag = self.build_match_index_to_tag(doc, datasets_and_mentions, publication_id)

                extraction_file_path = os.path.join(self.text_files_path,
                                                    os.pardir,
                                                    os.pardir,
                                                    os.pardir,
                                                    "ner-conll",
                                                    str(publication_id) +
                                                    "_" +
                                                    "extraction" +
                                                    ".conll")
                
                linking_file_path = os.path.join(self.text_files_path,
                                                 os.pardir,
                                                 os.pardir,
                                                 os.pardir,
                                                 "linking-conll",
                                                 str(publication_id) +
                                                 "_" +
                                                 "linking" +
                                                 ".conll")

                extraction_text, linking_text = self.create_conll_text(doc, match_index_to_tag)
                with open(extraction_file_path, "w") as publication_conll_file:
                    publication_conll_file.write(extraction_text)
                with open(linking_file_path, "w") as publication_conll_file:
                    publication_conll_file.write(linking_text)
        except:
            print("-------------------Publication", publication["publication_id"], "failed-------------------")

    def parse_text_files_to_conll_format(self):
        """Parses all the input text files into conll format and writes them to ../conll
        """
        self.build_publication_to_datasets_and_mentions()
        # parse each text file into a conll file
        with open(self.pub_path) as json_publications_file:
            publications = json.load(json_publications_file)
            # publications = [publication for publication in publications if publication['publication_id'] == 3152]
            with joblib.Parallel(n_jobs=os.cpu_count() - 1) as pool:
                pool(joblib.delayed(self.parse_publication)(publications[i])
                     for i in tqdm(range(len(publications)), desc='convert text files to conll format in to_conll.py'))

def main(data_folder_path):
    scispacy_parser = scispacy_util.SciSpaCyParser()
    train_path = os.path.join(data_folder_path, "train")
    dev_path = os.path.join(data_folder_path, "dev")
    test_path = os.path.join(data_folder_path, "test")

    # parse train set
    conll_parser = ConllParser(train_path, scispacy_parser)
    conll_parser.parse_text_files_to_conll_format()

    # parse dev set
    conll_parser = ConllParser(dev_path, scispacy_parser)
    conll_parser.parse_text_files_to_conll_format()

    # parse test set
    conll_parser = ConllParser(test_path, scispacy_parser)
    conll_parser.parse_text_files_to_conll_format()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder_path',
        type=str,
        help='The path to the data folder, which should contain train, dev, and test'
    )

    parser.set_defaults()

    args = parser.parse_args()
    main(args.data_folder_path)
