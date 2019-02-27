"""This script splits up the papers such that the datasets in train, dev, and test are disjoint.
   Note: It assumes that the old splits and their file structure remain the same as before"""

import os
import json
from collections import defaultdict
import numpy as np
import random

def build_dataset_id_to_papers(old_train_path, old_dev_path, old_test_path):
    """Creates a dictionary mapping dataset id to a list of papers that reference that dataset
       according to the labels.

       @param old_train_path: path to the old train split folder
       @param old_dev_path: path to the old dev split folder
       @param old_test_path: path to the old test split folder
    """
    train_citations_path = os.path.join(old_train_path, "data_set_citations.json")
    dev_citations_path = os.path.join(old_dev_path, "data_set_citations.json")
    test_citations_path = os.path.join(old_test_path, "data_set_citations.json")

    dataset_id_to_papers = defaultdict(list)
    with open(train_citations_path) as fp:
        train_citations_json = json.load(fp)

    with open(dev_citations_path) as fp:
        dev_citations_json = json.load(fp)

    with open(test_citations_path) as fp:
        test_citations_json = json.load(fp)

    citations_json = train_citations_json + dev_citations_json + test_citations_json
    for citation in citations_json:
        dataset_id = citation["data_set_id"]
        publication_id = citation["publication_id"]
        dataset_id_to_papers[dataset_id].append(publication_id)

    return dataset_id_to_papers

def get_old_splits(old_train_path, old_dev_path, old_test_path):
    """Returns the set of papers in the original train, dev, and test splits
       respectively

       @param old_train_path: path to the old train split folder
       @param old_dev_path: path to the old dev split folder
       @param old_test_path: path to the old test split folder
    """
    train_publications_path = os.path.join(old_train_path, "publications.json")
    dev_publications_path = os.path.join(old_dev_path, "publications.json")
    test_publications_path = os.path.join(old_test_path, "publications.json")

    with open(train_publications_path) as fp:
        train_publications_json = json.load(fp)

    with open(dev_publications_path) as fp:
        dev_publications_json = json.load(fp)

    with open(test_publications_path) as fp:
        test_publications_json = json.load(fp)

    train_papers = set()
    dev_papers = set()
    test_papers = set()

    for publication in train_publications_json:
        publication_id = publication["publication_id"]
        train_papers.add(publication_id)

    for publication in dev_publications_json:
        publication_id = publication["publication_id"]
        dev_papers.add(publication_id)

    for publication in test_publications_json:
        publication_id = publication["publication_id"]
        test_papers.add(publication_id)

    print()
    print("Original splits:")
    print("Num train papers:", len(train_papers))
    print("Num dev papers:", len(dev_papers))
    print("Num test papers:", len(test_papers))
    assert len(train_papers) + len(dev_papers) + len(test_papers) == 5100, "There should be exactly 5100 papers in the old splits"

    return train_papers, dev_papers, test_papers

def create_splits(dataset_id_with_papers_sorted):
    """Returns the set of papers in the new splits for train, dev, and test
       respectively

       @param dataset_id_with_paper_sorted: a sorted list of (dataset id, list of papers) tuples
    """
    train_papers = set()
    dev_papers = set()
    test_papers = set()
    all_papers = set()

    for dataset_id, papers in dataset_id_with_papers_sorted:
        all_papers.update(papers)
        # take any datasets that appear in many papers as training data
        if len(papers) > 20:
            train_papers.update(papers)
            continue

        # if any of the current dataset's papers are already in train, dev, or test,
        # put all of this dataset's papers in that split
        if any(paper in train_papers for paper in papers):
            train_papers.update(papers)
        elif any(paper in dev_papers for paper in papers):
            dev_papers.update(papers)
        elif any(paper in test_papers for paper in papers):
            test_papers.update(papers)
        else:
            # randomly assign this dataset's papers to train, dev, or test
            random_flip = random.randint(0, 100)
            if random_flip <= 70:
                train_papers.update(papers)
            elif random_flip <= 85:
                dev_papers.update(papers)
            else:
                test_papers.update(papers)

    # resolve conflicts by preferring dev over train, and test over dev
    train_papers = train_papers - dev_papers
    train_papers = train_papers - test_papers
    dev_papers = dev_papers - test_papers

    print()
    print("New splits:")
    print("Num train papers:", len(train_papers))
    print("Num dev papers:", len(dev_papers))
    print("Num test papers:", len(test_papers))
    assert len(train_papers) + len(dev_papers) + len(test_papers) == 2550, "There should be exactly 2550 papers with datasets in them"

    return train_papers, dev_papers, test_papers

def write_split(old_data_path,
                new_split_path,
                new_papers_path,
                old_train_papers,
                old_dev_papers,
                old_test_papers,
                new_papers):
    """Writes a concatenated conll file for a given data fold

       @param old_data_path: path to the old data folder
       @param new_split_path: path to write the concatenated conll file to
       @param new_papers_path: path to write the list of papers in the fold to
       @param old_train_papers: set of papers in the old train set
       @param old_dev_papers: set of papers in the old dev set
       @param old_test_papers: set of papers in the old test set
       @param new_papers: set of papers in the new fold
    """
    old_train_conll_path = os.path.join(old_data_path, "train", "ner-conll")
    old_dev_conll_path = os.path.join(old_data_path, "dev", "ner-conll")
    old_test_conll_path = os.path.join(old_data_path, "test", "ner-conll")

    with open(new_split_path, "w") as new_split_fp,\
         open(new_papers_path, "w") as new_papers_fp:
        for paper in new_papers:
            if paper in old_train_papers:
                base_conll_path = old_train_conll_path
            elif paper in old_dev_papers:
                base_conll_path = old_dev_conll_path
            elif paper in old_test_papers:
                base_conll_path = old_test_conll_path
            else:
                raise Exception("Paper {} was not found in old train, dev, or test".format(paper))

            old_conll_file_path = os.path.join(base_conll_path, str(paper) + "_extraction.conll")
            with open(old_conll_file_path) as old_conll_fp:
                new_split_fp.write(old_conll_fp.read())

            new_papers_fp.write(str(paper))
            new_papers_fp.write('\n')

def write_splits(old_data_path,
                 new_splits_base_path,
                 old_train_papers,
                 old_dev_papers,
                 old_test_papers,
                 new_train_papers,
                 new_dev_papers,
                 new_test_papers):
    """Writes concatenated conll  files for train, dev, and test based on the new splits

       @param old_data_path: path to the old data folder
       @param new_splits_base_path: path to the folder where the concatenated files will be written
       @param old_train_papers: set of papers in the old train set
       @param old_dev_papers: set of papers in the old dev set
       @param old_test_papers: set of papers in the old test set
       @param new_train_papers: set of papers in the new train fold
       @param new_dev_papers: set of papers in the new dev fold
       @param new_test_papers: set of papers in the new test fold
    """
    train_concat_path = os.path.join(new_splits_base_path, "train_concat.conll")
    dev_concat_path = os.path.join(new_splits_base_path, "dev_concat.conll")
    test_concat_path = os.path.join(new_splits_base_path, "test_concat.conll")
    train_papers_path = os.path.join(new_splits_base_path, "train_papers.txt")
    dev_papers_path = os.path.join(new_splits_base_path, "dev_papers.txt")
    test_papers_path = os.path.join(new_splits_base_path, "test_papers.txt")

    write_split(old_data_path, train_concat_path, train_papers_path, old_train_papers, old_dev_papers, old_test_papers, new_train_papers)
    write_split(old_data_path, dev_concat_path, dev_papers_path, old_train_papers, old_dev_papers, old_test_papers, new_dev_papers)
    write_split(old_data_path, test_concat_path, test_papers_path, old_train_papers, old_dev_papers, old_test_papers, new_test_papers)

def main():
    old_data_path = os.path.abspath(os.path.join("project", "data"))
    old_train_path = os.path.join(old_data_path, "train")
    old_dev_path = os.path.join(old_data_path, "dev")
    old_test_path = os.path.join(old_data_path, "test")

    old_train_papers, old_dev_papers, old_test_papers = get_old_splits(old_train_path, old_dev_path, old_test_path)

    dataset_id_to_papers = build_dataset_id_to_papers(old_train_path, old_dev_path, old_test_path)
    dataset_id_with_papers_sorted = sorted([(k, v) for k, v in dataset_id_to_papers.items()], key=lambda x: len(x[1]))
    new_train_papers, new_dev_papers, new_test_papers = create_splits(dataset_id_with_papers_sorted)

    new_splits_base_path = os.path.abspath(os.path.join("project", "ner_retraining", "data"))
    write_splits(old_data_path,
                 new_splits_base_path,
                 old_train_papers,
                 old_dev_papers,
                 old_test_papers,
                 new_train_papers,
                 new_dev_papers,
                 new_test_papers)


if __name__ == "__main__":
    main()