"""This file contains a class that can be used to predict dataset mentions using a trained
   named entity recognition (NER) model
"""

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from ner_rcc.rcc_ner import RccNerDatasetReader
from allennlp.common.params import Params
import torch.nn.functional as F
import torch


from tqdm import tqdm
import os


class NerModel():
    """Class for making predictions of dataset mentions using the NER model from AllenNLP
    """

    def __init__(self, data_path, model_path):
        # the path to the ner-conll folder of publications to predict on, this should generated
        # before using the ner model. This can be done by calling to_conll_test.py
        self.test_path = data_path
        self.model_path = model_path

    def predict_from_publication_list(self):
        citation_list = []
        filenum_mention_set = set()
        archive = load_archive(
            self.model_path
        )
        dataset_reader = RccNerDatasetReader.from_params(Params({
            "coding_scheme": "BIOUL",
            "cutoff_sentence_length": 50,
            "filter_sections": False ,
            "percent_negatives": 100,
            "tag_label": "ner",
            "token_indexers": {
                "token_characters": {
                    "type": "characters",
                    "character_tokenizer": {
                        "end_tokens": [
                            "@@PADDING@@",
                            "@@PADDING@@",
                            "@@PADDING@@"
                        ]
                    }
                },
                "tokens": {
                    "type": "single_id",
                    "lowercase_tokens": 'false'
                }
            }
        }))

        predictor = Predictor.from_archive(archive)
        for filename in tqdm(os.listdir(self.test_path), desc='extract dataset candidates with NER'):
            filenum = filename.replace("_extraction.conll", "")
            # using the AllenNLP command line tool to predict
            instances = dataset_reader.read(f'{self.test_path}/{filename}')
            mention = ""
            prob = -1
            for batch in range(0, len(instances), 16):
                instance_batch = instances[batch:min(batch+16, len(instances))]
                predicted_batch = predictor.predict_batch_instance(instance_batch)
                for instance, prediction in zip(instance_batch, predicted_batch):
                    for tag, word, logit in zip(prediction['tags'], prediction['words'], prediction['logits']):
                        # build up a mention based on BILOU tags and add to citation list
                        # when another O tag is found
                        if tag == 'O' and mention:
                            set_key = filenum + "_" + mention.rstrip()
                            if set_key not in filenum_mention_set:
                                citation_list.append({
                                    'publication_id': int(filenum),
                                    'mention': mention.rstrip(),
                                    'score': prob,
                                    'instance': [t.text for t in instance["tokens"].tokens]
                                })
                                filenum_mention_set.add(set_key)
                            mention = ""
                            prob = -1
                        elif tag != 'O':
                            if prob == -1:
                                probs = F.softmax(torch.tensor(logit), dim=-1)
                                prob_tensor, _ = torch.max(probs, 0)
                                prob = prob_tensor.data.item()
                            mention += str(word)
                            mention += " "
        return citation_list


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def main():
    # main function for debugging purposes
    model = NerModel("/data/ner-conll")
    model.predict_from_publication_list()


if __name__ == "__main__":
    main()
