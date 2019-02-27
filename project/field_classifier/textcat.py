from typing import Dict, List
import logging
import numpy as np
import re
from overrides import overrides
import json

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common.checks import ConfigurationError
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_filter import StopwordFilter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("textcat")
class TextCatReader(DatasetReader):
    """
    Reads tokens and their topic labels from the AG News Corpus.

    The Stanford Sentiment Treebank comes with labels


    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 word_tokenizer: Tokenizer = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._word_tokenizer = word_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            columns = data_file.readline().strip('\n').split('\t')
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                tokens = items.get("title")
                if tokens is None:
                    continue
                category = items["category"]
                instance = self.text_to_instance(tokens=tokens,
                                                 category=category)
                if instance is not None:
                    yield instance
                

    @overrides
    def text_to_instance(self, tokens: List[str], category: str = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.

        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        category ``str``, optional, (default = None).
            The category for this sentence.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The category label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_fields = []
        tokens_ = self._word_tokenizer.tokenize(tokens)
        if not tokens_:
            return None
        fields['tokens'] = TextField(tokens_,
                                        self._token_indexers)
        if category is not None:
            if category in ('NA', 'None'):
                category = -1
            fields['label'] = LabelField(category)
        return Instance(fields)
