import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join("project")))
import ner_model
import json

def main(conll_path, output_path, model_path):
    ner = ner_model.NerModel(conll_path, model_path)
    citations_list = ner.predict_from_publication_list()
    with open(output_path, "w") as fp:
        json.dump(citations_list, fp)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--conll_path',
    )

    parser.add_argument(
        '--output_path',
    )

    parser.add_argument(
        '--model_path'
    )

    args = parser.parse_args()
    main(args.conll_path,
         args.output_path,
         args.model_path)