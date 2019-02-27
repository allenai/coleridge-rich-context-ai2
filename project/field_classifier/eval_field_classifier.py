from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from field_classifier.classifier import Classifier
from field_classifier.predictor import ClassifierPredictor
from field_classifier.textcat import TextCatReader
import os
import json
import numpy as np

l0_archive = load_archive(
                os.path.abspath(os.path.join("data", "model_logs", "l0_model.tar.gz"))
            )
l0_predictor = Predictor.from_archive(l0_archive, 'classifier')
l1_archive = load_archive(
            os.path.abspath(os.path.join("data", "model_logs", "l1_model.tar.gz"))
        )
l1_predictor = Predictor.from_archive(l1_archive, 'classifier')
test_pubs = [{"title": "this is a test", "publication_id": 1}]
clf_output = []
l0_label_map = l0_archive.model.vocab.get_index_to_token_vocabulary("labels")
l1_label_map = l1_archive.model.vocab.get_index_to_token_vocabulary("labels")
for test_pub in test_pubs:
    l0_prediction = l0_predictor.predict_json({"title": test_pub['title']})
    l1_prediction = l1_predictor.predict_json({"title": test_pub['title']})
    pred = {}
    pred['publication_id'] = test_pub['publication_id']
    l0_score = np.max(l0_prediction['label_probs'])
    l1_score = np.max(l1_prediction['label_probs'])
    l0_field = l0_label_map[np.argmax(l0_prediction['label_probs'])]
    l1_field = l1_label_map[np.argmax(l1_prediction['label_probs'])]
    if l1_score > 0.4:
        output_score = "{}:{}".format(l0_score, l1_score)
        output_field = "{}:{}".format(l0_field, l1_field)
    else:
        output_score = "{}".format(l0_score)
        output_field = "{}".format(l0_field)
    pred['score'] = output_score
    pred['research_field'] = output_field
    clf_output.append(pred)
print(clf_output)
