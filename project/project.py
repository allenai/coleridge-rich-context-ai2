"""This is the main script for outputting predictions for the competition
"""
from rule_based_model import RuleBasedModel
from xgboost import XGBClassifier
from typing import Dict, Union, List
import json
import os
import create_linking_dataset
import create_sgtb_dataset
import structured_gradient_boosting
import structured_learner
from method_extractor import MethodExtractor
from sklearn.externals import joblib
import argparse
import xgboost_linking
from ner_model import NerModel

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from field_classifier.classifier import Classifier
from field_classifier.predictor import Seq2SeqPredictor
from field_classifier.textcat import TextCatReader
import os
import json
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import text_utils
from nltk.corpus import stopwords
from s2base import scispacy_util

def perform_evaluation(rule_based_model: RuleBasedModel,
                       linking_model: XGBClassifier,
                       publications_path: str,
                       labels_path: str,
                       data_folder_path: str,
                       ner_predicted_citations: List[Dict[str, Union[str, float, int]]],
                       rule_based_output_path: str = None,
                       rule_based_input_path: str = None,
                       predict_input_path: str = None,
                       verbose: bool = True):
    """Performs end task evaluation for the competition

       @param rule_based_model: the rule based model object to use
       @param linking_model: the linking model object to use (SGTB)
       @param publications_path: path to the publications.json to evaluate predictions on
       @param labels_path: path to the labels for the input publications
       @param data_folder_path: path to the data folder
       @param ner_predicted_citations: predicted citations based on predicted mentions from the NER model
       @param rule_based_output_path: (optional) path to save the rule based model output
       @param rule_based_input_path: (optional) path to the rule based model output if saved
       @param predict_input_path: (optional) path to the text input files if the rule based output is not saved
    """
    citation_list = []
    if rule_based_input_path:
        citation_list = joblib.load(rule_based_input_path)["citation_list"]
    else:
        print("Making rule based predictions...")
        with open(publications_path) as publications_file:
            json_publications = json.load(publications_file)
            citation_list = rule_based_model.predict_from_publications_list(json_publications, predict_input_path)
        joblib.dump({"citation_list": citation_list, "rule_based_version": "10"}, rule_based_output_path)
    citation_list += ner_predicted_citations
    print()
    if verbose:
        print("PRE LINKING EVALUATION")
        print()
        rule_based_model.evaluate(citation_list, labels_path)

    # convert the rule based candidates into the expected format for the linking model
    print("Preparing rule based input...")
    print(len(citation_list))
    rule_based_input = create_linking_dataset.create_rule_based_input(citation_list)
    print("Preparing sgtb input...")
    sgtb_input, pub_ids = create_sgtb_dataset.create_dataset_input(rule_based_input, os.path.join(data_folder_path, "mention_context_cache.pkl"), data_folder_path, is_test=True)
    print("Preparing xgboost input...")
    xgb_input_X, xgb_input_y, xgb_pub_ids, xgb_dataset_ids = xgboost_linking.processed_docs_to_xgboost_dataset(sgtb_input, pub_ids)

    # create a mapping of publication id to dataset ids predicted by the linking model
    print("Making linking predictions...")
    all_y_probs = linking_model.predict_proba(xgb_input_X)
    linking_predictions = {}
    for y_probs, pub_id, dataset_id in zip(all_y_probs, xgb_pub_ids, xgb_dataset_ids):
        # 0 is the "no link" label.
        if y_probs[0] > 0.75:
            continue

        if pub_id not in linking_predictions:
            linking_predictions[pub_id] = defaultdict(float)

        # add up linking probabilities when the same dataset id is linked with multiple
        # mentions in the same paper.
        assert y_probs[1] > -0.001 and y_probs[1] < 1.001
        linking_predictions[pub_id][dataset_id] += y_probs[1]
        
    # commented out code for working with the SGTB model as we are using XGBoost
    # linking_predictions = {}
    # for doc, pub_id in zip(sgtb_input, pub_ids):
    #     sgtb_idx_input = structured_learner.make_idx_data([doc])
    #     _, predictions = linking_model.predict(sgtb_idx_input[0], sgtb_idx_input[2], sgtb_idx_input[3])
    #     linking_predictions[pub_id] = set(predictions)

    # filter the rule based candidates based on the results of the linking model
    linking_filtered_citation_list = []
    for citation in citation_list:
        citation_dataset_id = str(citation["data_set_id"])
        if citation["publication_id"] in linking_predictions and \
           citation_dataset_id in linking_predictions[citation["publication_id"]]:
            # update score.
            citation['score'] = min(1., linking_predictions[citation["publication_id"]][citation_dataset_id])
            linking_filtered_citation_list.append(citation)

    print()
    print("POST LINKING EVALUATION")
    rule_based_model.evaluate(linking_filtered_citation_list, labels_path)

    return linking_filtered_citation_list

def generate_citations_from_ner_mentions(ner_mentions: List[Dict[str, Union[int, str, float]]],
                                         kb_path: str):
    """Generate candidate citations for the mentions produced by the ner model by using TFIDF
       weighted overlap with dataset titles

       @param ner_mentions: list of the ner_mentions
       @param kb_path: path to the knowledge base of datasets
    """
    nltk_stopwords = set(stopwords.words('english'))
    scispacy_parser = scispacy_util.SciSpaCyParser()
    substring_matches = set()
    tfidf_vectorizer = text_utils.get_tfidf_vectorizer()

    with open(kb_path) as kb_file_:
        kb = json.load(kb_file_)

    dataset_titles = []
    tokenized_dataset_titles = []
    dataset_ids = []
    dataset_id_to_title = {}
    for dataset in tqdm(kb, desc="processing kb"):
        dataset_title = text_utils.text_preprocess(dataset["title"])
        dataset_id = dataset["data_set_id"]
        dataset_titles.append(dataset_title)
        tokenized_dataset_titles.append(dataset_title.split(" "))
        dataset_ids.append(dataset_id)
        dataset_id_to_title[dataset_id] = dataset_title.split(" ")

    output_citations = []
    num_candidates = []
    i = 0
    mention_citations = []
    for mention in tqdm(ner_mentions, desc="Generating candidates from ner mentions"):
        publication_id = mention["publication_id"]
        mention_text = mention["mention"]
        instance = mention["instance"]

        if len(instance) - len(mention_text.split()) < 5:
            continue

        if len(mention_text.split()) == 1 and not mention_text.isupper():
            continue

        parsed_sentence = scispacy_parser.scispacy_create_doc(' '.join(instance))
        pos_counts = defaultdict(int)
        for t in parsed_sentence:
            pos_counts[t.pos_] += 1
        
        if pos_counts["NOUN"] + pos_counts["VERB"] == 0:
            continue
        
        if (pos_counts["NUM"] + pos_counts["SYM"] + pos_counts["PUNCT"]) > 0.4*len(parsed_sentence) and pos_counts["VERB"] == 0:
            continue

        mention_citations.append({"publication_id": publication_id, "mention": mention_text, "score": mention["score"]})

        mention_text = text_utils.text_preprocess(mention_text)
        dataset_candidates = text_utils.get_substring_candidates(dataset_ids,
                                                                 dataset_titles,
                                                                 tokenized_dataset_titles,
                                                                 mention_text,
                                                                 instance,
                                                                 nltk_stopwords,
                                                                 scispacy_parser,
                                                                 tfidf_vectorizer)
        num_candidates.append(0)
    
        sorted_candidates = []
        for dataset_id, match_count in zip(dataset_candidates[0], dataset_candidates[1]):
            sorted_candidates.append((dataset_id, match_count))

        sorted_candidates = sorted(sorted_candidates, key = lambda x: x[1], reverse=True)

        filtered_candidates = []
        for candidate in sorted_candidates:
            score = candidate[1]
            if score > 0.0:
                filtered_candidates.append((candidate[0], score))

        for top_candidate in range(0, min(30, len(filtered_candidates))):
            if sorted_candidates != []:
                num_candidates[i] += 1
                output_dict = {}
                output_dict["publication_id"] = publication_id
                output_dict["data_set_id"] = sorted_candidates[top_candidate][0]
                output_dict["score"] = sorted_candidates[top_candidate][1]
                output_dict["mention_list"] = [mention["mention"]]
                output_citations.append(output_dict)
        i += 1

    print("Num mentions:", len(num_candidates))
    print("Average candidates per mention:", np.mean(num_candidates))
    print("Min, median, max candidates per mention:", np.min(num_candidates), np.median(num_candidates), np.max(num_candidates))
    print("unique:", sum(np.unique(num_candidates, return_counts=True)[1]))
    return output_citations, mention_citations

def main(dev_evaluation, error_analysis, methods_only, holdout_eval, ner_only=False):
    random.seed(2018)
    train_path = os.path.abspath(os.path.join("project", "dataset_split_data", "train"))
    dev_path = os.path.abspath(os.path.join("project", "dataset_split_data", "dev"))
    kb_path = os.path.abspath(os.path.join("project", "data", "data_sets.json"))
    sage_methods_path = os.path.abspath(os.path.join("project", "data", "sage_research_methods.json"))
    leipzig_word_counts_path = os.path.abspath(os.path.join("project", "data", "eng_wikipedia_2016_1M-words.txt"))
    test_path = os.path.abspath(os.path.join("data"))

    sgtb_model = joblib.load(os.path.abspath(os.path.join("project", "linking_models", "linking_model_v4.pkl")))["clf"]
    xgboost_model = joblib.load(os.path.abspath(os.path.join("project", "linking_models", "xgboost_linking_model_v24_full.pkl")))["clf"]
    ner_model_path = os.path.abspath(os.path.join("project", "ner_model", "tweaked_model.tar.gz"))

    if holdout_eval:
        print("HOLDOUT SET EVALUATION - citations")
        holdout_ner_model = NerModel(os.path.abspath(os.path.join("project", "holdout_sampled", "ner-conll")), ner_model_path)
        # commented out code recreates the ner output for the sampled holdout set, this would need to happen if the ner model changes
        # or the holdout sample changes, but does not impact the submission code path
        holdout_ner_predictions = holdout_ner_model.predict_from_publication_list()
        with open(os.path.abspath(os.path.join("project", "holdout_sampled", "ner_output.json")), "w") as ner_output_file_:
            json.dump(holdout_ner_predictions, ner_output_file_)
        with open(os.path.abspath(os.path.join("project", "holdout_sampled", "ner_output_with_instances.json"))) as ner_output_file_:
            holdout_ner_predictions = json.load(ner_output_file_)
        ner_predicted_citations, ner_predicted_mentions = generate_citations_from_ner_mentions(holdout_ner_predictions, kb_path)
        with open(os.path.abspath(os.path.join("project", "holdout_sampled", "ner_candidates.json")), "w") as fp:
            json.dump(ner_predicted_citations, fp)

        holdout_rule_based_model = RuleBasedModel(train_path, dev_path, kb_path, test_path)
        holdout_publications_path = os.path.abspath(os.path.join("project", "holdout_sampled", "publications.json"))
        holdout_labels_path = os.path.abspath(os.path.join("project", "holdout_sampled", "data_set_citations.json"))
        holdout_rule_based_output_path = os.path.abspath(os.path.join("project", "holdout_sampled", "rule_based_output_v10.pkl"))
        holdout_predict_input_path = os.path.abspath(os.path.join("project", "holdout_sampled"))
        holdout_predictions = perform_evaluation(holdout_rule_based_model,
                                                 xgboost_model,
                                                 holdout_publications_path,
                                                 holdout_labels_path,
                                                 holdout_predict_input_path,
                                                 ner_predicted_citations,
                                                 rule_based_input_path=holdout_rule_based_output_path,
                                                 rule_based_output_path=holdout_rule_based_output_path,
                                                 predict_input_path=holdout_predict_input_path)
        return

    if not ner_only:
        # Predict methods.
        print("Predicting methods...")
        method_extractor = MethodExtractor(train_path, dev_path, sage_methods_path, leipzig_word_counts_path)
        output_file_path = os.path.abspath(os.path.join("data", "output", "methods.json"))
        test_publications_path = os.path.abspath(os.path.join("data", "input", "publications.json"))
        test_predict_input_path = os.path.abspath(os.path.join("data"))
        with open(test_publications_path) as publications_file, open(output_file_path, 'w') as output_file:
            json_publications = json.load(publications_file)
            method_predictions = method_extractor.predict_from_publications_list(json_publications, test_predict_input_path)
            json.dump(method_predictions, output_file)
        if methods_only: return
    
    if dev_evaluation and not ner_only:
        print("DEV SET EVALUATION - citations")
        dev_rule_based_model = RuleBasedModel(train_path, dev_path, kb_path)
        dev_publications_path = os.path.abspath(os.path.join("project", "data", "dev", "publications.json"))
        dev_labels_path = os.path.abspath(os.path.join("project", "data", "dev", "data_set_citations.json"))
        dev_rule_based_output_path = os.path.abspath(os.path.join("project", "data", "dev", "rule_based_output_v10.pkl"))
        dev_predict_input_path = os.path.abspath(os.path.join("project", "data", "dev"))
        dev_predictions = perform_evaluation(dev_rule_based_model,
                                             xgboost_model,
                                             dev_publications_path,
                                             dev_labels_path,
                                             dev_predict_input_path,
                                             rule_based_input_path=None,
                                             rule_based_output_path=dev_rule_based_output_path,
                                             predict_input_path=dev_predict_input_path)

    print("TEST SET EVALUATION - citations")
    test_predictions_mentions = []
    test_publications_path = os.path.abspath(os.path.join("data", "input", "publications.json"))
    test_labels_path = os.path.abspath(os.path.join("rich-context-competition", "evaluate", "data_set_citations.json"))
    test_rule_based_output_path = os.path.abspath(os.path.join("project", "data", "test", "rule_based_output_v10.pkl"))
    test_predict_input_path = os.path.abspath(os.path.join("data"))

    # make additional dataset mention predictions using the trained NER model
    model = NerModel("/data/ner-conll", ner_model_path)
    ner_predictions_list = model.predict_from_publication_list()
    ner_predicted_citations, ner_predicted_mentions = generate_citations_from_ner_mentions(ner_predictions_list, kb_path)

    test_rule_based_model = RuleBasedModel(train_path, dev_path, kb_path, test_path)
    test_predictions_original = perform_evaluation(test_rule_based_model,
                                            xgboost_model,
                                            test_publications_path,
                                            test_labels_path,
                                            test_predict_input_path,
                                            ner_predicted_citations,
                                            rule_based_input_path=None,
                                            rule_based_output_path=test_rule_based_output_path,
                                            predict_input_path=test_predict_input_path)

    test_predictions_dict = {}
    pub_dataset_to_longest_mention_list = {}
    for prediction in test_predictions_original:
        dataset_id = prediction["data_set_id"]
        publication_id = prediction["publication_id"]
        mention_list = prediction["mention_list"]
        mention_list_length = len(mention_list)
        key = str(dataset_id) + "_" + str(publication_id)
        if key in test_predictions_dict:
            if mention_list_length > pub_dataset_to_longest_mention_list[key]:
                pub_dataset_to_longest_mention_list[key] = mention_list_length
                test_predictions_dict[key] = prediction.copy()
        else:
            pub_dataset_to_longest_mention_list[key] = mention_list_length
            test_predictions_dict[key] = prediction.copy()

    test_predictions = []
    for prediction in test_predictions_dict:
        test_predictions.append(test_predictions_dict[prediction])

    # write dataset citation predictions to file
    output_file_path = os.path.abspath(os.path.join("data", "output", "data_set_citations.json"))
    with open(output_file_path, "w") as output_file:
        json.dump(test_predictions, output_file)

    print("Predicting dataset mentions...")
    # build up list of predicted mentions from predicted citations
    pub_id_to_mention = {}
    for test_prediction in test_predictions:
        output_mention_base = test_prediction.copy()
        pub_id = output_mention_base["publication_id"]
        output_mention_base.pop("mention_list")
        output_mention_base.pop("data_set_id")
        for mention in test_prediction["mention_list"]:
            output_mention = output_mention_base.copy()
            output_mention["mention"] = mention
            if pub_id in pub_id_to_mention:
                if mention in pub_id_to_mention[pub_id]:
                    continue
                else:
                    pub_id_to_mention[pub_id].add(mention)
            else:
                pub_id_to_mention[pub_id] = set([mention])

            test_predictions_mentions.append(output_mention)

    # write dataset mention predictions to file
    test_predictions_mentions += ner_predicted_mentions
    output_file_path_mentions = os.path.abspath(os.path.join("data", "output", "data_set_mentions.json"))
    with open(output_file_path_mentions, "w") as output_file:
        json.dump(test_predictions_mentions, output_file)
    if ner_only: return
        
    print("Predicting research fields...")
    # predict field of study of publications using a trained AllenNLP model
    l0_archive = load_archive(
                os.path.abspath(os.path.join("project", "data", "model_logs", "l0_model.tar.gz"))
            )
    l0_predictor = Predictor.from_archive(l0_archive, 'seq2seq')
    l1_archive = load_archive(
                os.path.abspath(os.path.join("project", "data", "model_logs", "l1_model.tar.gz"))
            )
    l1_predictor = Predictor.from_archive(l1_archive, 'seq2seq')
    with open(test_publications_path, 'r') as file_:
            test_pubs = json.load(file_)
    clf_output = []
    l0_label_map = l0_archive.model.vocab.get_index_to_token_vocabulary("labels")
    l1_label_map = l1_archive.model.vocab.get_index_to_token_vocabulary("labels")
    for test_pub in tqdm(test_pubs, desc="Predicting research fields"):
        if test_pub['title'] == '':
            continue
        l0_prediction = l0_predictor.predict_json({"title": test_pub['title']})
        l1_prediction = l1_predictor.predict_json({"title": test_pub['title']})
        pred = {}
        pred['publication_id'] = test_pub['publication_id']
        l0_score = np.max(l0_prediction['label_probs'])
        l1_score = np.max(l1_prediction['label_probs'])
        l0_field = l0_label_map[np.argmax(l0_prediction['label_probs'])]
        l1_field = l1_label_map[np.argmax(l1_prediction['label_probs'])]
        if l1_score > 0.4 and l0_field != l1_field:
            output_score = round((float(l0_score) + float(l1_score))/2., 2)
            output_field = "{}:{}".format(l0_field, l1_field)
        else:
            output_score = round(float(l0_score), 2)
            output_field = "{}".format(l0_field)
        pred['score'] = output_score
        pred['research_field'] = output_field
        clf_output.append(pred)

    # write predictions for research fields to file
    output_path = os.path.abspath(os.path.join("data", "output", "research_fields.json"))
    with open(output_path, "w") as output_file:
        json.dump(clf_output, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dev_evaluation',
        action='store_true',
        help="Whether to perform evaluation on the dev set in addition to the test set"
    )

    parser.add_argument(
        '--error_analysis',
        action='store_true'
    )

    parser.add_argument(
        '--methods-only',
        action='store_true',
        help="Only predict methods then halt."
    )

    parser.add_argument(
        '--holdout_eval',
        action='store_true',
        help="Evaluate on the phase 1 holdout set"
    )

    parser.add_argument(
        '--ner_only',
        action='store_true',
        help="Only output the mentions file then halt"
    )

    parser.set_defaults(dev_evaluation=False,
                        error_analysis=False,
                        methods_only=False,
                        holdout_eval=False,
                        ner_only=False)

    args = parser.parse_args()
    main(args.dev_evaluation,
         args.error_analysis,
         args.methods_only,
         args.holdout_eval,
         args.ner_only)
