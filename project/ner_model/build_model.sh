#/bin/bash

rm model/*
allennlp train allennlp-ner-config.json -s model --include-package ner_rcc