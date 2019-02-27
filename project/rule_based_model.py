"""This file contains a class for the rule based model for generating dataset extraction candidates
"""

import os
import json
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
import re
from s2base import scispacy_util
from tqdm import tqdm
from sklearn.externals import joblib
from spacy.lang import char_classes
import codecs

# Results on test set (organizers' dev fold)
# v1: simple exact string matching on uid, name, mentions with all dev fold mentions excluded: precision = 0.0556640625, recall = 0.4634146341463415, F1 = 0.09938971229293812
# v2: simple exact string matching on uid, name, mentions with all mentions in train set or not in dev fold (accounts for overlap between mentions in train and dev fold): precision = 0.06348314606741573, recall = 0.9186991869918699, F1 = 0.1187598528638991
# v3: simple exact string matching on uid, name, mentions with all mentions in train set (excludes mentions that are not present in either train or dev fold): precision = 0.06848484848484848, recall = 0.9186991869918699, F1 = 0.12746756909193455
# v4: exact string matching like v3, plus filter datasets with a year mentioned that is not present in the text: precision = 0.11098527746319366, recall = 0.7967479674796748, F1 = 0.194831013916501
# v5: exact string matching like v3, plus filter datasets with a year mentiond that is not present in the text, and no years mentioned that are present in the text: precision = 0.09403862300587741, recall = 0.9105691056910569, F1 = 0.17047184170471838
# v6: v5 + filter out mentions that are single, common english words: precision = 0.10526315789473684, recall = 0.9105691056910569, F1 = 0.18871103622577926
# v7: v6 + ignore case: Precision: 0.027991082486995295, Recall: 0.9186991869918699, Accuracy: 0.027921917469730665, F1: 0.05432692307692308
# v8 (best recall): v6 + extra modifications to the mentions based on to_conll work: Precision: 0.10079575596816977, Recall: 0.926829268292683, Accuracy: 0.1, F1: 0.18181818181818185
# v9: v8 + excluding shorter, overlapping (0.8 or 1.0 threshold) mention lists: Precision: 0.12280701754385964, Recall: 0.5691056910569106, Accuracy: 0.11235955056179775, F1: 0.20202020202020204
# v10 (best precision, best f1): v8 + try to exclude references section from text to search: Precision: 0.13820078226857888, Recall: 0.8617886178861789, Accuracy: 0.13520408163265307, F1: 0.2382022471910112

# Results on dev set
# v8: Precision: 0.06193330259720301, Recall: 0.8915929203539823, Accuracy: 0.061470408785845025, F1: 0.11582123868371894
# v10: Precision: 0.13820078226857888, Recall: 0.8617886178861789, Accuracy: 0.13520408163265307, F1: 0.2382022471910112

YEAR_PATTERN = r"((?<![.|\d])\d\d\d\d(?![.|\d]))"

class RuleBasedModel():
    """This class can be used to predict dataset extractions based on string search for mentions from the training set
    """
    def __init__(self, train_path, dev_path, kb_path, test_path = None):
        # path to the data folder for the train set
        self.train_path = train_path

        # path to the data folder for the dev set
        self.dev_path = dev_path

        # path to the json kb file
        self.kb_path = kb_path

        # optional path to the data folder for the test set
        # if this argument is passed in, the model will use mentions from the dev
        # and train set to make predictions on the test set.
        # Otherwise it will use mentions from the train set to make predictions on the
        # test set
        self.test_path = test_path

        # set of unique mentions in the dev set
        self._dev_set_mentions = set()
        self._build_dev_set_mentions()

        # set of unique mentions in the train set
        self._train_set_mentions = set()
        self._build_train_set_mentions()

        # set of unique mentions in the entire kb
        self._all_mentions = set()
        self._build_all_mentions()

        # dictionary mapping dataset id to a set of mentions of that dataset
        self._id_to_mentions = {}
        self._build_id_to_mentions()

        # set of english stopwords
        self._stopwords =  set(stopwords.words('english'))

        # an instance of a scispacy parser
        self._scispacy_parser = scispacy_util.SciSpaCyParser()

        # dictionary mapping mention to the number of datasets it is a mention for
        self._mention_dataset_count = {}

        # the total number of datasets
        self._dataset_count = 0
        self._build_mention_dataset_count()

        # precompile mention regexes
        self._dataset_id_to_regexes = {}
        for dataset_id in self._id_to_mentions:
            compiled_res = []
            for mention in self._id_to_mentions[dataset_id]:
                mention_patterns = self._build_mention_patterns(mention)
                for pattern in mention_patterns:
                    compiled_re = re.compile(pattern)
                    compiled_res.append(compiled_re)
            self._dataset_id_to_regexes[dataset_id] = compiled_res

    def compare_mention_lists(self):
        """Function to print out counts of unique mentions in train and
           dev sets.
        """
        print("Train:", len(self._train_set_mentions))
        print("Dev:", len(self._dev_set_mentions))
        print("Intersection:", len(self._train_set_mentions.intersection(self._dev_set_mentions)))
        print("Train - Dev:", len(self._train_set_mentions - self._dev_set_mentions))
        print("Dev - Train:", len(self._dev_set_mentions - self._train_set_mentions))
        total_set = set()
        for key in self._id_to_mentions:
            for mention in self._id_to_mentions[key]:
                total_set.add(mention)
        print("Total:", len(total_set))

    def _build_id_to_mentions(self):
        """Helper function to build the dictionary mapping dataset id
           to sets of mentions of that dataset
        """

        with open(self.kb_path) as kb_file:
            json_kb = json.load(kb_file)
            for dataset in json_kb:
                dataset_id = dataset["data_set_id"]
                name = dataset["name"]
                uid = dataset["unique_identifier"]
                mention_list = dataset["mention_list"]
                self._id_to_mentions[dataset_id] = set()

                # add uid as a "mention"
                self._id_to_mentions[dataset_id].add(uid)

                # add dataset name as a "mention"
                self._id_to_mentions[dataset_id].add(name)

                # add all actual mentions as mentions
                for mention in mention_list:
                    if (mention in self._train_set_mentions) or (self.test_path != None and mention in self._dev_set_mentions):
                        self._id_to_mentions[dataset_id].add(mention)

    def _build_dev_set_mentions(self):
        """Helper function to build the set of dev set mentions
        """
        dev_labels_path = os.path.join(self.dev_path, "data_set_citations.json")
        with open(dev_labels_path) as dev_labels_file:
            json_dev_labels = json.load(dev_labels_file)
            for pair in json_dev_labels:
                for mention in pair["mention_list"]:
                    self._dev_set_mentions.add(mention)

    def _build_train_set_mentions(self):
        """Helper funciton to build the set of train set mentions
        """
        train_labels_path = os.path.join(self.train_path, "data_set_citations.json")
        with open(train_labels_path) as train_labels_file:
            json_train_labels = json.load(train_labels_file)
            for pair in json_train_labels:
                for mention in pair["mention_list"]:
                    self._train_set_mentions.add(mention)

    def _build_all_mentions(self):
        """Helper funciton to build the set of all mentions in the kb
        """
        with open(self.kb_path) as kb_file:
            json_kb = json.load(kb_file)
            for dataset in json_kb:
                for mention in dataset["mention_list"]:
                    self._all_mentions.add(mention)

    def _make_citation_dict(self, pub_id, dataset_id, mention_list, score):
        """Helper function to create the dictionary format that evaluation script expects
           for one prediction

           @param pub_id: the publication id for this publication - dataset pair
           @param dataset_id: the dataset id for this publication - dataset pair
           @param mention_list: a list of mention texts supporting this pair
           @param score: the prediction score given to this prediction
        """
        return {"publication_id": int(pub_id), "data_set_id": int(dataset_id), "mention_list": mention_list, "score": float(score)} 

    def _get_years_from_text(self, text):
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

    def _build_mention_patterns(self, mention):
        """Builds patterns to search for in a text based on one mention

           @param mention: the raw text of the mention to search for
        """
        replaced_mention = mention
        replaced_mention = self._scispacy_parser.preprocess_text(replaced_mention)
        # replace tokens in the mention text with what they show up as in the actual text
        replaced_mention = replaced_mention.replace("\\", "")
        replaced_mention = replaced_mention.replace("\u2afd", " ")
        replaced_mention = replaced_mention.replace("\u2013", "\xad")
        replaced_mention = replaced_mention.replace("\ufb01", "fi")
        replaced_mention = re.sub(r"(?<=\S)-\s", "", replaced_mention)

        # commented out because it does not change performance on the test set, and makes the model 6x slower
        # but I still think it might be correct and the test set just isn't representative of every scenario
        # xad_1 = replaced_mention.replace("\xad-", "\xad")
        # xad_2 = replaced_mention.replace("-\xad", "\xad")

        # plural_mention = replaced_mention + "s"
        # plural_xad_1 = xad_1 + "s"
        # plural_xad_2 = xad_2 + "s"
        # patterns_without_new_lines = [replaced_mention, xad_1, xad_2, plural_mention, plural_xad_1, plural_xad_2]

        # build a regex pattern with an optional new line/space between each character to allow for mentions to be
        # split over lines
        patterns_without_new_lines = [replaced_mention]
        patterns_with_new_lines = []
        for pattern_without_new_line in patterns_without_new_lines:
            pattern = r""
            for c in pattern_without_new_line:
                if c == " ":
                    pattern += r"[\n|\s]*"
                else:
                    pattern += re.escape(c)
            patterns_with_new_lines.append(pattern)
        return patterns_with_new_lines

    def _build_dataset_id_to_mention_list_in_text_v8(self, text):
        """Builds a dictionary mapping dataset id to a list of mentions of that dataset in a text

           @param text: the text to search for mentions in
        """
        dataset_to_mention = {}
        self._document_count = 0
        for dataset_id in self._dataset_id_to_regexes:
            for regex in self._dataset_id_to_regexes[dataset_id]:
                    match = regex.search(text)
                    # string matching is significantly faster, but regex is required to allow for new
                    # lines between characters/words in a mention
                    # The commented out code below is the string matching version of this search
                    # if mention_pattern in text:
                    if match:
                        if dataset_id in dataset_to_mention:
                            dataset_to_mention[dataset_id].append(match.group(0))
                            # dataset_to_mention[dataset_id].append(mention_pattern)
                        else:
                            dataset_to_mention[dataset_id] = [match.group(0)]
                            # dataset_to_mention[dataset_id] = [mention_pattern]

        return dataset_to_mention

    def _build_dataset_id_to_mention_list_in_text(self, text):
        """Builds a dictionary mapping dataset id to a list of mentions of that dataset in a text
           Note: this function just looks for the actual mention text without any augmentation,
           as opposed to the above function that augments the raw mention text

           @param text: the text to search for mentions in
        """
        dataset_to_mention = {}
        self._document_count = 0
        for dataset_id in self._id_to_mentions:
            for mention in self._id_to_mentions[dataset_id]:
                mention = self._scispacy_parser.preprocess_text(mention)
                if mention in text:
                    if dataset_id in dataset_to_mention:
                        dataset_to_mention[dataset_id].append(mention)
                    else:
                        dataset_to_mention[dataset_id] = [mention]

        return dataset_to_mention

    def _build_mention_dataset_count(self):
        """Builds a dictionary mapping mention text to the number of datasets it is a mention for
        """
        with open(self.kb_path) as kb_file:
            json_kb = json.load(kb_file)
            for dataset in json_kb:
                self._dataset_count += 1
                dataset_id = dataset["data_set_id"]
                name = dataset["name"]
                uid = dataset["unique_identifier"]
                mention_list = dataset["mention_list"]

                for mention in mention_list:
                    if (mention in self._train_set_mentions) or (self.test_path != None and mention in self._dev_set_mentions):
                        if mention in self._mention_dataset_count:
                            self._mention_dataset_count[mention] += 1
                        else:
                            self._mention_dataset_count[mention] = 1

    def filter_common_words_keep(self, mentions):
        """Returns True if the list of mentions has at least one mention that is not
           a single, common, English word

           @param mentions: the list of mentions to search over
        """
        for mention in mentions:
            # uses the scispacy parser vocab as proxy for "single, common, English word"
            if not (mention in self._scispacy_parser.nlp.vocab):
                return True
        return False

    def dataset_has_year_not_in_text(self, years_found, dataset):
        """Returns True if the input dataset's mentions have a year in them that is
           not found in the text

           @param years_found: set of years found in the text
           @param dataset: dataset id of the dataset of interest
        """
        for mention in self._id_to_mentions[dataset]:
            years_in_mention = re.findall(YEAR_PATTERN, mention)
            years_in_mention = [int(year) for year in years_in_mention]
            for year in years_in_mention:
                if year not in years_found:
                    return True
        return False

    def dataset_year_filter_keep(self, years_found, dataset):
        """More conservative version of the above function. Returns True if the input
           dataset's mentions have a year in them that is not found in the text, and do
           not have any years in them that are found in the text

           @param years_found: set of years found in the text
           @param dataset: dataset id of the dataset of interest
        """
        bad_year_found = False
        for mention in self._id_to_mentions[dataset]:
            years_in_mention = re.findall(YEAR_PATTERN, mention)
            years_in_mention = [int(year) for year in years_in_mention]
            for year in years_in_mention:
                if year in years_found:
                    return True
                else:
                    bad_year_found = True
        return not bad_year_found

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

    def predict_v3(self, text_file_path):
        """Model v3: this version of the model does exact string matching for dataset names,
           dataset uids, and mentions from the training set

           @param text_file_path: path to the text file to predict for
        """
        pub_id = text_file_path.split(".")[-2].split("/")[-1]
        citation_list = []
        with open(text_file_path) as text_file:
            text = text_file.read()
            for dataset_id in self._id_to_mentions:
                for mention in self._id_to_mentions[dataset_id]:
                    if mention in text:
                        citation = self._make_citation_dict(pub_id, dataset_id, [mention], 1.0)
                        citation_list.append(citation)
                        break

        return citation_list

    def predict_v6(self, text_file_path):
        """Model v6: This version of the model does exact string matching like v3, plus
           filters out datasets that do not mention any years from the text and do mention
           a year not in the text, plus filters out datasets that are only supported by single,
           common word mentions

           @param text_file_path: path to the text file to predict for
        """
        pub_id = text_file_path.split(".")[-2].split("/")[-1]
        citation_list = []
        with open(text_file_path) as text_file:
            text = text_file.read()
            text = self._scispacy_parser.preprocess_text(text)

            dataset_to_mention = self._build_dataset_id_to_mention_list_in_text(text)
            filtered_years_found = self._get_years_from_text(text)

        for dataset in dataset_to_mention:
            if self.dataset_year_filter_keep(filtered_years_found, dataset):
                if self.filter_common_words_keep(dataset_to_mention[dataset]):
                    citation = self._make_citation_dict(pub_id, dataset, dataset_to_mention[dataset], 1.0)
                    citation_list.append(citation)

        return citation_list

    def predict_v8(self, text_file_path):
        """Model v8: This version of the model is v6, with some extra augmentations to the mentions
           that are searched for in the text. These augmentations are based on findings when creating
           the conll files

           @param text_file_path: path to the text file to predict for
        """
        pub_id = text_file_path.split(".")[-2].split("/")[-1]
        citation_list = []
        with open(text_file_path) as text_file:
            text = text_file.read()
            text = self._scispacy_parser.preprocess_text(text)

            # the difference from v6 is here
            dataset_to_mention = self._build_dataset_id_to_mention_list_in_text_v8(text)
            filtered_years_found = self._get_years_from_text(text)

        for dataset in dataset_to_mention:
            if self.dataset_year_filter_keep(filtered_years_found, dataset):
                if self.filter_common_words_keep(dataset_to_mention[dataset]):
                    citation = self._make_citation_dict(pub_id, dataset, dataset_to_mention[dataset], 1.0)
                    citation_list.append(citation)

        return citation_list

    def predict_v9(self, text_file_path):
        """Model v9: This version of the model is v8, pluse excluding datasets whose found mentions
           are a subset of another dataset's found mentions

           @param text_file_path: path to the text file to predict for
        """
        pub_id = text_file_path.split(".")[-2].split("/")[-1]
        citation_list = []
        with open(text_file_path) as text_file:
            text = text_file.read()
            text = self._scispacy_parser.preprocess_text(text)

            dataset_to_mention = self._build_dataset_id_to_mention_list_in_text_v8(text)
            filtered_years_found = self._get_years_from_text(text)

        filtered_dataset_to_mention = {}
        for dataset in dataset_to_mention:
            if self.dataset_year_filter_keep(filtered_years_found, dataset):
                if self.filter_common_words_keep(dataset_to_mention[dataset]):
                    filtered_dataset_to_mention[dataset] = dataset_to_mention[dataset]

        datasets_to_output = {}
        for dataset_1 in filtered_dataset_to_mention:
            better_dataset_found = False
            for dataset_2 in filtered_dataset_to_mention:
                if dataset_1 == dataset_2:
                    continue
                else:
                    mention_list_1 = set(filtered_dataset_to_mention[dataset_1])
                    mention_list_2 = set(filtered_dataset_to_mention[dataset_2])

                    overlap_percent = len(mention_list_1.intersection(mention_list_2)) / float(len(mention_list_1))

                    if overlap_percent == 1.0 and len(mention_list_2) > len(mention_list_1):
                        better_dataset_found = True

            if not better_dataset_found:
                citation = self._make_citation_dict(pub_id, dataset_1, filtered_dataset_to_mention[dataset_1], 1.0)
                citation_list.append(citation)

        return citation_list

    def predict_v10(self, text_file_path):
        """Model v10: This version of the model is v8 plus an attempt to not search for mentions
           in the references section

           @param text_file_path: path to the text file to predict for
        """
        pub_id = text_file_path.split(".")[-2].split("/")[-1]
        citation_list = []
        with open(text_file_path) as text_file:
            text = text_file.read()
            text = self._scispacy_parser.preprocess_text(text)
            text = self._filter_references_section(text)

            dataset_to_mention = self._build_dataset_id_to_mention_list_in_text_v8(text)
            filtered_years_found = self._get_years_from_text(text)

        for dataset in dataset_to_mention:
            if self.dataset_year_filter_keep(filtered_years_found, dataset):
                if self.filter_common_words_keep(dataset_to_mention[dataset]):
                    citation = self._make_citation_dict(pub_id, dataset, dataset_to_mention[dataset], 1.0)
                    citation_list.append(citation)

        return citation_list

    def predict(self, text_file_path):
        """Predict datasets for one text file. Returns a list of citation
           dictionaries.

           @param text_file_path: path to the text file to predict for
        """
        return self.predict_v10(text_file_path)

    def predict_from_publications_list(self, publication_list, predict_path):
        """Predict datasets for a list of publications, with each publication
           in the format provided in publications.json file.

           @param publication_list: the result of json.load('publications.json')
        """
        citation_list = []
        for publication in tqdm(publication_list, desc='extract regex-based dataset candidates'):
            predictions = self.predict(os.path.join(predict_path, "input", "files", "text", str(publication["publication_id"]) + ".txt"))
            citation_list += predictions

        return citation_list

    def evaluate(self, predicted_citations_list, labels_path):
        """Run evaluation on the input predictions and labels. This is the same evaluation setup
           as the organizers' evaluate script.

           @param predicted_citations_list: a list of dictionaries in the correct output format for a citation
           @param labels_path: the path to the json file of labels for the predictions being input
        """

        if not os.path.isfile(labels_path):
            return

        with open(labels_path) as labels_json_file:
            labels_json = json.load(labels_json_file)

        predicted_set = set()
        for prediction in predicted_citations_list:
            predicted_set.add(str(prediction["publication_id"]) + ":" + str(prediction["data_set_id"]))

        actual_set = set()
        for actual in labels_json:
            actual_set.add(str(actual["publication_id"]) + ":" + str(actual["data_set_id"]))

        tp = len(actual_set.intersection(predicted_set))
        fp = len(predicted_set - actual_set)
        tn = 0
        fn = len(actual_set - predicted_set)
        accuracy = (tp+tn)/(tp+fp+tn+fn)
        precision = tp/(fp+tp) if (tp+fp) != 0 else 1
        recall = tp/(tp+fn)
        f1 = (2 * precision * recall)/(precision + recall) if (precision + recall) != 0 else 0

        # We expect these metrics to give the same results on the test set as ./rcc.sh evaluate
        print("Precision:", precision)
        print("Recall:", recall)
        print("Accuracy:", accuracy)
        print("F1:", f1)

    def error_analysis(self, output_path, labels_path, pre_linking_candidates):
        with open(labels_path) as labels_json_file:
            labels_json = json.load(labels_json_file)

        with open(output_path) as predictions_json_file:
            predicted_citations_list = json.load(predictions_json_file)

        kb_path = os.path.abspath(os.path.join("project", "data", "data_sets.json"))
        with open(kb_path) as kb_file:
            kb_json = json.load(kb_file)

        dataset_id_to_kb_entry = {}
        for dataset in kb_json:
            dataset_id_to_kb_entry[dataset["data_set_id"]] = dataset

        pub_id_to_pre_linking = {}
        pub_id_to_predicted = {}
        pub_id_to_actual = {}
        pub_dataset_to_mention_list = {}
        all_pub_ids = set()
        for prediction in predicted_citations_list:
            pub_id = prediction["publication_id"]
            dataset_id = prediction["data_set_id"]
            all_pub_ids.add(pub_id)
            if pub_id in pub_id_to_predicted:
                pub_id_to_predicted[pub_id].add(dataset_id)
            else:
                pub_id_to_predicted[pub_id] = set([dataset_id])

        for actual in labels_json:
            pub_id = actual["publication_id"]
            dataset_id = actual["data_set_id"]
            all_pub_ids.add(pub_id)
            if pub_id in pub_id_to_actual:
                pub_id_to_actual[pub_id].add(dataset_id)
            else:
                pub_id_to_actual[pub_id] = set([dataset_id])

        for pre_linking in pre_linking_candidates:
            pub_id = pre_linking["publication_id"]
            dataset_id = pre_linking["data_set_id"]
            all_pub_ids.add(pub_id)
            pub_dataset_to_mention_list[str(pub_id) + "_" + str(dataset_id)] = pre_linking["mention_list"]
            if pub_id in pub_id_to_pre_linking:
                pub_id_to_pre_linking[pub_id].add(dataset_id)
            else:
                pub_id_to_pre_linking[pub_id] = set([dataset_id])

        for pub_id in all_pub_ids:
            if pub_id in pub_id_to_predicted:
                predicted = pub_id_to_predicted[pub_id]
            else:
                predicted = set()
            if pub_id in pub_id_to_actual:
                actual = pub_id_to_actual[pub_id]
            else:
                actual = set()
            if pub_id in pub_id_to_pre_linking:
                pre_linking = pub_id_to_pre_linking[pub_id]
            else:
                pre_linking = set()

            pre_linking_titles = [(dataset_id_to_kb_entry[dataset_id]["title"], dataset_id, pub_dataset_to_mention_list[str(pub_id) + "_" + str(dataset_id)]) for dataset_id in pre_linking]
            predicted_titles = [(dataset_id_to_kb_entry[dataset_id]["title"], dataset_id, pub_dataset_to_mention_list[str(pub_id) + "_" + str(dataset_id)]) for dataset_id in predicted]
            actual_titles = [(dataset_id_to_kb_entry[dataset_id]["title"], dataset_id) for dataset_id in actual]
            print("Publication id:", pub_id)
            print()
            print("Pre linking:", pre_linking_titles)
            print()
            print("Post linking:", predicted_titles)
            print()
            print("Actual:", actual_titles)
            print()

def main():
    # keeping main function around to use for debugging
    train_path = os.path.abspath(os.path.join("project", "data", "train"))
    dev_path = os.path.abspath(os.path.join("project", "data", "dev"))
    kb_path = os.path.abspath(os.path.join("project", "data", "data_sets.json"))
    test_path = os.path.abspath(os.path.join("data"))
    model = RuleBasedModel(train_path, dev_path, kb_path, test_path)
    model.compare_mention_lists()
    ex_file_path = os.path.join(os.getcwd(), "project/data/test/input/files/text/143.txt")
    predictions = model.predict(ex_file_path)
    print(len(predictions))
    print(predictions)

    # publications_path = os.path.abspath(os.path.join("data", "input", "publications.json"))

    # with open(publications_path) as publications_file:
    #     json_publications = json.load(publications_file)
    #     citation_list = model.predict_from_publications_list(json_publications)

    # labels_path = os.path.abspath(os.path.join("rich-context-competition", "evaluate", "data_set_citations.json"))
    # model.evaluate(citation_list, labels_path)

if __name__ == "__main__":
    main()
