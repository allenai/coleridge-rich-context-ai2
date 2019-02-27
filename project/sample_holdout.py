"""Script to sample some publications from the phase 1 holdout set"""

import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json

def main():
    SAMPLE_COUNT = 400
    holdout_path = os.path.abspath(os.path.join("project", "holdout", "data"))
    holdout_publications_path = os.path.abspath(os.path.join(holdout_path, "publications.json"))
    holdout_citations_path = os.path.abspath(os.path.join(holdout_path, "data_set_citations.json"))
    holdout_ner_path = os.path.abspath(os.path.join(holdout_path, "ner-conll"))
    holdout_text_path = os.path.abspath(os.path.join(holdout_path, "input", "files", "text"))
    holdout_pdf_path = os.path.abspath(os.path.join(holdout_path, "input", "files", "pdf"))

    with open(holdout_publications_path) as fp:
        holdout_publications = json.load(fp)

    with open(holdout_citations_path) as fp:
        holdout_citations = json.load(fp)

    pub_id_to_pub = {}
    for publication in holdout_publications:
        publication_id = publication["publication_id"]
        pub_id_to_pub[publication_id] = publication

    pub_id_to_citation_entry = defaultdict(list)
    for citation in holdout_citations:
        publication_id = citation["publication_id"]
        pub_id_to_citation_entry[publication_id].append(citation)

    holdout_paper_ids = list(pub_id_to_pub.keys())
    sampled_ids = np.random.choice(holdout_paper_ids, size=SAMPLE_COUNT, replace=False)

    new_folder_path = os.path.abspath(os.path.join("project", "holdout_sampled"))
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
    for paper_id in tqdm(sampled_ids):
        text_file_name = str(paper_id) + ".txt"
        pdf_file_name = str(paper_id) + ".pdf"
        ner_file_name = str(paper_id) + "_extraction.conll"
        linking_file_name = str(paper_id) + "_linking.conll"
        publication_entry = pub_id_to_pub[paper_id]
        citation_entries = pub_id_to_citation_entry[paper_id]
        new_publications.append(publication_entry)
        new_citations += citation_entries

        folder_path = os.path.abspath(os.path.join("project", "holdout", "data"))
        os.system("cp {} {}".format(os.path.join(folder_path, "input", "files", "text", text_file_name),
                                    os.path.join(text_path, text_file_name)))
        os.system("cp {} {}".format(os.path.join(folder_path, "input", "files", "pdf", pdf_file_name),
                                    os.path.join(pdf_path, pdf_file_name)))
        os.system("cp {} {}".format(os.path.join(folder_path, "ner-conll", ner_file_name),
                                    os.path.join(ner_conll_path, ner_file_name)))

    with open(os.path.join(new_folder_path, "publications.json"), "w") as fp:
        json.dump(new_publications, fp, indent=4)

    with open(os.path.join(new_folder_path, "data_set_citations.json"), "w") as fp:
        json.dump(new_citations, fp, indent=4)


if __name__ == '__main__':
    main()