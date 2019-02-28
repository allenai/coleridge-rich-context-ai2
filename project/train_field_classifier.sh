#!/bin/bash

# command to train the research field classifier
allennlp train --include-package project.field_classifier.classifier --include-package project.field_classifier.predictor --include-package project.field_classifier.textcat -s /project/model_logs_l0/ /project/field_classifier/classifier_l0.json
allennlp train --include-package project.field_classifier.classifier --include-package project.field_classifier.predictor --include-package project.field_classifier.textcat -s /project/model_logs_l1/ /project/field_classifier/classifier_l1.json