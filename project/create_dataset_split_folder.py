"""Script to write all the needed files to a new folder based on splits provided in text files"""

import os
import json
from typing import Dict, Union, List
from collections import defaultdict
from tqdm import tqdm

def load_all_publications(old_base_path: str):
    train_path = os.path.join(old_base_path, "train")
    train_pubs_path = os.path.join(train_path, "publications.json")
    train_citations_path = os.path.join(train_path, "data_set_citations.json")

    dev_path = os.path.join(old_base_path, "dev")
    dev_pubs_path = os.path.join(dev_path, "publications.json")
    dev_citations_path = os.path.join(dev_path, "data_set_citations.json")

    test_path = os.path.join(old_base_path, "test")
    test_pubs_path = os.path.join(test_path, "publications.json")
    test_citations_path = os.path.join(test_path, "data_set_citations.json")

    with open(train_pubs_path) as fp:
        train_pubs = json.load(fp)

    with open(train_citations_path) as fp:
        train_citations = json.load(fp)

    with open(dev_pubs_path) as fp:
        dev_pubs = json.load(fp)

    with open(dev_citations_path) as fp:
        dev_citations = json.load(fp)

    with open(test_pubs_path) as fp:
        test_pubs = json.load(fp)

    with open(test_citations_path) as fp:
        test_citations = json.load(fp)

    all_pubs = train_pubs + dev_pubs + test_pubs
    all_citations = train_citations + dev_citations + test_citations

    pub_id_to_pub = {}
    pub_id_to_citation = defaultdict(list)
    for pub_entry in all_pubs:
        publication_id = pub_entry["publication_id"]
        pub_id_to_pub[publication_id] = pub_entry

    for citation_entry in all_citations:
        publication_id = citation_entry["publication_id"]
        pub_id_to_citation[publication_id].append(citation_entry)

    return pub_id_to_pub, pub_id_to_citation

def make_split_folder(papers: set,
                      new_folder_path: str, 
                      old_base_path: str,
                      all_pubs: Dict[str, Union[str, int]],
                      all_citations: Dict[str, Union[int, List, float]]):
    os.system("mkdir {}".format(new_folder_path))
    linking_conll_path = os.path.join(new_folder_path, "linking-conll")
    ner_conll_path = os.path.join(new_folder_path, "ner-conll")
    input_path = os.path.join(new_folder_path, "input")
    files_path = os.path.join(input_path, "files")
    pdf_path = os.path.join(files_path, "pdf")
    text_path = os.path.join(files_path, "text")
    os.system("mkdir {}".format(linking_conll_path))
    os.system("mkdir {}".format(ner_conll_path))
    os.system("mkdir {}".format(input_path))
    os.system("mkdir {}".format(files_path))
    os.system("mkdir {}".format(pdf_path))
    os.system("mkdir {}".format(text_path))

    new_publications = []
    new_citations = []
    for paper in tqdm(papers):
        text_file_name = paper + ".txt"
        pdf_file_name = paper + ".pdf"
        ner_file_name = paper + "_extraction.conll"
        linking_file_name = paper + "_linking.conll"
        publication_id = int(paper)
        publication_entry = all_pubs[publication_id]
        citation_entries = all_citations[publication_id]
        new_publications.append(publication_entry)
        new_citations += citation_entries

        if os.path.isfile(os.path.join(old_base_path, "train", "input", "files", "text", text_file_name)):
            folder_path = os.path.join(old_base_path, "train")
        elif os.path.isfile(os.path.join(old_base_path, "dev", "input", "files", "text", text_file_name)):
            folder_path = os.path.join(old_base_path, "dev")
        else:
            folder_path = os.path.join(old_base_path, "test")

        os.system("cp {} {}".format(os.path.join(folder_path, "input", "files", "text", text_file_name),
                                    os.path.join(text_path, text_file_name)))
        os.system("cp {} {}".format(os.path.join(folder_path, "input", "files", "pdf", pdf_file_name),
                                    os.path.join(pdf_path, pdf_file_name)))
        os.system("cp {} {}".format(os.path.join(folder_path, "ner-conll", ner_file_name),
                                    os.path.join(ner_conll_path, ner_file_name)))
        os.system("cp {} {}".format(os.path.join(folder_path, "linking-conll", linking_file_name),
                                    os.path.join(linking_conll_path, linking_file_name)))

    with open(os.path.join(new_folder_path, "publications.json"), "w") as fp:
        json.dump(new_publications, fp, indent=4)

    with open(os.path.join(new_folder_path, "data_set_citations.json"), "w") as fp:
        json.dump(new_citations, fp, indent=4)

def load_papers_set_from_file(path: str) -> set:
    paper_ids = set()
    with open(path) as fp:
        line = fp.readline()
        while line:
            paper_ids.add(line.rstrip())
            line = fp.readline()
    return paper_ids

def main():
    old_folder_base_path = os.path.abspath(os.path.join("project", "data"))
    new_folder_path = os.path.abspath(os.path.join("project", "dataset_split_data"))
    new_split_path = os.path.abspath(os.path.join("project", "ner_retraining", "data"))
    train_papers_path = os.path.join(new_split_path, "train_papers.txt")
    dev_papers_path = os.path.join(new_split_path, "dev_papers.txt")
    test_papers_path = os.path.join(new_split_path, "test_papers.txt")

    train_papers = load_papers_set_from_file(train_papers_path)
    dev_papers = load_papers_set_from_file(dev_papers_path)
    test_papers = load_papers_set_from_file(test_papers_path)

    pub_id_to_pub, pub_id_to_citation = load_all_publications(old_folder_base_path)
    make_split_folder(train_papers,
                      os.path.join(new_folder_path, "train"),
                      old_folder_base_path,
                      pub_id_to_pub,
                      pub_id_to_citation)
    make_split_folder(dev_papers,
                      os.path.join(new_folder_path, "dev"),
                      old_folder_base_path,
                      pub_id_to_pub,
                      pub_id_to_citation)
    make_split_folder(test_papers,
                      os.path.join(new_folder_path, "test"),
                      old_folder_base_path,
                      pub_id_to_pub,
                      pub_id_to_citation)
    


if __name__ == '__main__':
    main()