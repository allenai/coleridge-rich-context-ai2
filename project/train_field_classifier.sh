#!/bin/bash

# command to train the research field classifier
# Note: the current setup reads data from S3 that is not publicly accessible (although the data source is publicly accessible)
allennlp train --include-package field_classifier.classifier --include-package field_classifier.predictor --include-package field_classifier.textcat -s ./model_logs/ ./field_classifier/classifier.json