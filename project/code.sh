#!/usr/bin/env bash

# create ner-conll directory if it doesn't exist
mkdir -p ./data/ner-conll
# convert test files to conll format, which the NER model expects
python3 ./project/to_conll_test.py
# run the main competition scripts, which outputs all the expected output files
python3 ./project/project.py
