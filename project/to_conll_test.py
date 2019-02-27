"""This file can be run to convert the test files to conll format, saved in the ner-conll folder.
   This is mostly copy pasted from to_conll.py as a quick workaround to run the conll parsing code
   at test time. A cleaner implementation would not need this file, and would just make use of
   to_conll.py
"""

import os
import json
from sklearn.externals import joblib
from s2base import scispacy_util
from tqdm import tqdm
import re
from create_sgtb_dataset import get_scispacy_doc
import logging
logging.basicConfig(level=logging.ERROR)

# the path to the test publications.json
PUB_PATH = os.path.abspath(os.path.join("data", "input", "publications.json"))
# the path to the test text files
TEXT_FILES_PATH = os.path.abspath(os.path.join("data", "input", "files", "text"))
# an instance of SciSpaCyParser
SCISPACY_PARSER = scispacy_util.SciSpaCyParser()

def create_conll_line(token):
    """Create one line of the output conll file

       @param token: the token for the line being created
       @param match_index_to_tag: the dictionary mapping token index to entity tag
    """
    word = token.text
    pos = token.pos_
    tag = "O"
    linking_tag = "_"
    entity_tag = "O"

    output_line = word + " " + pos + " " + tag + " " + entity_tag

    extraction_line = output_line
    linking_line = output_line + " " + linking_tag

    return extraction_line, linking_line

def create_conll_sentence(sentence):
    """Creates one sentence of the output conll file

       @param sentence: the spacy sentence for the sentence being created
       @param match_index_to_tag: the dictionary mapping token index to entity tag
    """
    extraction_sentence = ""
    linking_sentence = ""
    for token in sentence:
        # spacy includes space tokens, which we can safely ignore
        if token.pos_ == "SPACE" or token.text == "\n" or token.text == " ":
            continue
        extraction_line, linking_line = create_conll_line(token)
        extraction_sentence += extraction_line + "\n"
        linking_sentence += linking_line + "\n"

    return extraction_sentence, linking_sentence

def create_conll_text(doc):
    """Creates one document of conll output

       @param doc: the spacy doc to process
       @param match_index_to_tag: the dictionary mapping token index to entity tag
    """
    extraction_text = ""
    linking_text = ""
    prev_sent = None
    for sent in doc.sents:
        extraction_sentence, linking_sentence = create_conll_sentence(sent)
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

def parse_publication(publication):
    """Parses one raw text file into conll format and writes to
       ../conll/<publication_id>_<extraction or linking>.conll

       @param publication: the json publication being processed
    """
    publication_id = publication["publication_id"]
    datasets_and_mentions = []
    publication_text_path = os.path.join(TEXT_FILES_PATH, str(publication_id) + ".txt")
    with open(publication_text_path) as publication_text_file:
        full_text = publication_text_file.read()
        doc = get_scispacy_doc(os.path.join(TEXT_FILES_PATH, os.pardir, os.pardir, os.pardir), publication_id, SCISPACY_PARSER)

        if not os.path.isdir(os.path.join(TEXT_FILES_PATH, os.pardir, os.pardir, os.pardir, "ner-conll")):
            os.makedirs(os.path.join(TEXT_FILES_PATH, os.pardir, os.pardir, os.pardir, "ner-conll"))

        extraction_file_path = os.path.join(TEXT_FILES_PATH,
                                            os.pardir,
                                            os.pardir,
                                            os.pardir,
                                            "ner-conll",
                                            str(publication_id) +
                                            "_" +
                                            "extraction" +
                                            ".conll")

        extraction_text, _ = create_conll_text(doc)
        with open(extraction_file_path, "w") as publication_conll_file:
            publication_conll_file.write(extraction_text)


def parse_text_files_to_conll_format():
    """Parses all the input text files into conll format and writes them to ../conll
    """
    # parse each text file into a conll file
    with open(PUB_PATH) as json_publications_file:
        publications = json.load(json_publications_file)

        with joblib.Parallel(n_jobs=os.cpu_count() - 1) as pool:
            pool(joblib.delayed(parse_publication)(publications[i])
                 for i in tqdm(range(len(publications)), desc='convert text files to conll format in to_conll_test.py'))

        #for i in tqdm(range(len(publications)), desc='convert text files to conll format in to_conll_test.py'):
        #   parse_publication(publications[i])

if __name__ == "__main__":
    parse_text_files_to_conll_format()
