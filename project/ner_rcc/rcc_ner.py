"""Slightly modified subclass of the AllenNLP conll2003 dataset reader.

Allows pruning negative sentences given a percent value and a limiting
by a max length
"""
from typing import Dict, Sequence, Iterable, List
import itertools
import logging
logging.basicConfig(level=logging.ERROR)

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import Conll2003DatasetReader
from random import randint
from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False

_VALID_LABELS = {'ner', 'pos', 'chunk'}


@DatasetReader.register("rcc-ner")
class RccNerDatasetReader(Conll2003DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:
    WORD POS-TAG CHUNK-TAG NER-TAG
    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.
    percent_negatives: ``int``, optional (default=``100``)
        Represents the percentage of negative sentences included
    cutoff_sentence_length: ``int``, optional (default=``30``)
        Represents the max number of tokens for a sentence to be evaluated
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False,
                 percent_negatives: int = 100,
                 cutoff_sentence_length: int = 30,
                 coding_scheme: str = "IOB1",
                 filter_sections: bool = False) -> None:
        super().__init__(token_indexers, tag_label, feature_labels, lazy, coding_scheme)
        self.percent_negatives = percent_negatives
        self.cutoff_sentence_length = cutoff_sentence_length
        self.filter_sections = filter_sections

    def _is_title(self, tokens: List[Token], pos_tags: List[str], i: int):
        if pos_tags[i] == "PROPN":
            return True
        if len(tokens) <= 2:
            return True
        if i + 1 < len(tokens):
            if tokens[i+1].text == ":" or pos_tags[i+1] in ["PROPN", "NUM"]:
                return True

    def _is_sentence(self, tokens: List[Token], pos_tags: List[str]):
        if len(tokens) < 3:
            return False
        if "NOUN" not in pos_tags and "VERB" not in pos_tags:
            return False
        
        pos_counts = defaultdict(int)
        for pos_tag in pos_tags:
            pos_counts[pos_tag] += 1

        if (pos_counts["NUM"] + pos_counts["SYM"] + pos_counts["PUNCT"]) > 0.4*len(pos_tags) and pos_counts["VERB"] == 0:
            return False

        return True

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        start = False
        stop = False
        start_words_single = ["abstract",
                              "introduction",
                              "methods",
                              "data",
                              "method",
                              "intro",
                              "background",
                              "keywords"]
        end_words_single = ["references",
                            "acknowledgements"]
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            instance_count = 0
            yielded = False
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    instance_count += 1
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    tokens, pos_tags, chunk_tags, ner_tags = [list(field) for field in zip(*fields)]

                    # negative sentence
                    if len(set(ner_tags)) == 1 and ner_tags[0] == 'O':
                        if randint(0, 100) > self.percent_negatives:
                            continue
                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens]
                    for i, (token, pos_tag) in enumerate(zip(tokens, pos_tags)):
                        if token.text.lower() in start_words_single and self._is_title(tokens, pos_tags, i):
                            start = True
                        if instance_count >= 75:
                            start = True
                        if start and instance_count >= 150 and token.text.lower() in end_words_single and self._is_title(tokens, pos_tags, i):
                            stop = True
                    if self.filter_sections:
                        if not start:
                            continue
                        if stop:
                            break
                    # print(tokens)
                    # print(pos_tags)
                    # print(self._is_sentence(tokens, pos_tags))
                    # input('e.')
                    # if not self._is_sentence(tokens, pos_tags):
                    #     continue
                    
                    if self.cutoff_sentence_length and len(tokens) < self.cutoff_sentence_length:
                        yielded = True
                        yield self.text_to_instance(tokens, pos_tags, chunk_tags, ner_tags)
                
            if not yielded:
                yield self.text_to_instance([Token("The")], ["Det"], ["O"], ["O"])
