"""This file can be run to create the rule based dataset candidates. It assumes that
   there are train/dev/test folders in project/data. The create_rule_based_input function
   can be used to convert a citation list (the competition output format) to the format expected
   by the next step of the system. The generate_rule_based_dataset function can be used to produce
   the rule based candidates for any publications list.
"""

from s2base.scispacy_util import SciSpaCyParser
import os
import json
import rule_based_model
import argparse
import project

def create_rule_based_input(citation_list, gold_labels_set=None):
    """Creates the expected input format for the rule based candidates from a citation list
       output of the rule based model. This function is intended to be used for processing
       test data, as the main function in this file will convert and save train, dev, and test
       output

       @param citation_list: a citation list formatted in the contest output format
       @param gold_labels_set: (optional) the set of <pub_id>_<dataset_id> pairs that are correct,
                               should not be passed in at test time
    """
    output = {}

    for citation in citation_list:
        pub_id = citation["publication_id"]
        dataset_id = citation["data_set_id"]
        mention_list = citation["mention_list"]
        if "score" in citation:
            score = citation["score"]
        else:
            score = -1

        for mention in mention_list:
            if gold_labels_set != None:
                gold_entity = "NULL"
                entity_key = str(pub_id) + "_" + str(dataset_id)
                if entity_key in gold_labels_set:
                    gold_entity = dataset_id
                row_to_add = {"mention": mention, "dataset_id": gold_entity, "candidate_dataset_ids": [dataset_id], "score": score}
            else:
                # if processing test data, do not include the gold label
                row_to_add = {"mention": mention, "candidate_dataset_ids": [dataset_id], "score": score}
            if pub_id in output:
                output[pub_id].append(row_to_add)
            else:
                output[pub_id] = [row_to_add]

    return output

def generate_labels_based_dataset(data_folder_path, gold_labels_path):
    """Create and save the linking dataset for the labels file. This dataset
       is a dictionary mapping publication id to a list of mentions and the dataset
       they are labeled as.

       Note: The context for all the occurrences of each mention will be genereated
             (and cached) as needed in a later step

       @param: data_folder_path: path to the data folder to run on (train, dev, or test)
       @param: gold_labels_path: path to the labels file to use in generating this dataset
    """
    with open(gold_labels_path) as gold_labels_file:
        gold_labels_json = json.load(gold_labels_file)

    output = {}
    for gold_label in gold_labels_json:
        pub_id = gold_label["publication_id"]
        dataset_id = gold_label["data_set_id"]
        mention_list = gold_label["mention_list"]

        for mention in mention_list:
            row_to_add = {"mention": mention, "dataset_id": dataset_id}
            if pub_id in output:
                output[pub_id].append(row_to_add)
            else:
                output[pub_id] = [row_to_add]

    out_path = os.path.join(data_folder_path, "linking_labels.json")
    with open(out_path, "w") as out_file:
        json.dump(output, out_file)

def generate_rule_based_dataset(output_path, data_folder_path, gold_labels_path, ner_output_path: str, is_dev=False):
    """Create and save the linking dataset for the rule based candidates. This dataset
       is a dictionary mapping publication id to a list of mentions, the dataset they
       are predicted as, and the dataset that is labeled for the publication if the prediction
       is correct, or NULL if the prediction is wrong. See the note in the code for more details

       Note: The context for all the occurrences of each mention will be genereated
             (and cached) as needed at train time

       @param: output_path: path to the file to write candidates to
       @param: data_folder_path: path to the data folder to run on (train, dev, or test)
       @param: gold_labels_path: path to the labels file to use in generating this dataset
       @param: ner_output_path: path to the candidate mentions from the NER model
       @param: is_dev: whether or not the data being processed is dev data. this impacts
                       which examples the rule based model is allowed to use
    """
    train_path = os.path.abspath(os.path.join("project", "dataset_split_data", "train"))
    dev_path = os.path.abspath(os.path.join("project", "dataset_split_data", "dev"))
    kb_path = os.path.abspath(os.path.join("project", "data", "data_sets.json"))
    test_path = os.path.abspath(os.path.join("data"))

    # without the test_path parameter, the rule based model will not use dev examples for training
    if is_dev:
        model = rule_based_model.RuleBasedModel(train_path, dev_path, kb_path)
    else:
        model = rule_based_model.RuleBasedModel(train_path, dev_path, kb_path, test_path)

    publications_path = os.path.join(data_folder_path, "publications.json")

    with open(publications_path) as publications_file:
        json_publications = json.load(publications_file)
        citation_list = model.predict_from_publications_list(json_publications, data_folder_path)

    with open(ner_output_path) as ner_file_:
        ner_citation_list = json.load(ner_file_)
    
    ner_citation_list, ner_mention_list = project.generate_citations_from_ner_mentions(ner_citation_list, kb_path)
    citation_list += ner_citation_list

    with open(gold_labels_path) as gold_labels_file:
        gold_labels_json = json.load(gold_labels_file)

    output = {}
    gold_labels_set = set()
    for gold_label in gold_labels_json:
        pub_id = gold_label["publication_id"]
        dataset_id = gold_label["data_set_id"]
        mention_list = gold_label["mention_list"]
        for mention in mention_list:
            # Note: we make the assumption here that if a publication-dataset pair
            # is correct, all mentions that the rule based model produces that support
            # that pair are correct. This is needed because the rule based model
            # produces mention-dataset pairs that match a mention-dataset pair in the knowledge
            # base, but does not necessarily match a mention-dataset pair in the provided labels.
            # Additionally, the rule based model finds some slight perturbations of the mentions listed
            # in the knowledge base/labels, and we want to count these as correct. We also already don't
            # know which occurrence of a mention in a text actually links to the dataset labeled, so this
            # type of noise is aleady present in the dataset
            gold_labels_set.add(str(pub_id) + "_" + str(dataset_id))

    output = create_rule_based_input(citation_list, gold_labels_set)

    with open(output_path, "w") as out_file:
        json.dump(output, out_file)

def main(dataset_root: str, output_root: str):
    train_folder_path = os.path.join(dataset_root, "train")
    dev_folder_path = os.path.join(dataset_root, "dev")
    test_folder_path = os.path.join(dataset_root, "test")
    kb_path = os.path.abspath(os.path.join("project", "data", "datasets.json"))

    train_ner_output_path = os.path.join(train_folder_path, "ner_output_split_train.json")
    dev_ner_output_path = os.path.join(dev_folder_path, "ner_output.json")
    test_ner_output_path = os.path.join(test_folder_path, "ner_output.json")

    train_output_path = os.path.join(output_root, "train_candidates.json")
    dev_output_path = os.path.join(output_root, "dev_candidates.json")
    test_output_path = os.path.join(output_root, "test_candidates.json")

    # generate_labels_based_dataset(test_folder_path, os.path.join(test_folder_path, "data_set_citations.json"))
    generate_rule_based_dataset(test_output_path, test_folder_path, os.path.join(test_folder_path, "data_set_citations.json"), test_ner_output_path)

    # generate_labels_based_dataset(dev_folder_path, os.path.join(dev_folder_path, "data_set_citations.json"))
    generate_rule_based_dataset(dev_output_path, dev_folder_path, os.path.join(dev_folder_path, "data_set_citations.json"), dev_ner_output_path, is_dev=True)

    # generate_labels_based_dataset(train_folder_path, os.path.join(train_folder_path, "data_set_citations.json"))
    generate_rule_based_dataset(train_output_path, train_folder_path, os.path.join(train_folder_path, "data_set_citations.json"), train_ner_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_root',
    )

    parser.add_argument(
        '--output_root'
    )

    args = parser.parse_args()
    main(args.dataset_root, args.output_root)