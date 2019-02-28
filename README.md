# Download data before running things
There are large files and data folders that cannot be put on github, so they are on s3. Before running things, you will need to download the `data`, `dataset_split_data`, and `holdout_sampled` folders from [this s3 bucket](https://s3.console.aws.amazon.com/s3/buckets/ai2-s2-rcc/?region=us-west-2&tab=overview) and place them at the top level of the project folder.

# Estimated runtime

On the phase 2 set (5000 papers), our model takes approximately 11 hours to run to completion on a fixed performance (note that the T series are burstable, and it will likely take much longer on a burstable instance) EC2 instance with 8 CPUs and 32G of RAM.

# Overall structure

The data folder contains the input files to make predictions on and the output files that the predictions are written to.

The project folder contains all the code/data that our model needs to run. When the organizers run our code, it will run through the code.sh shell script which runs project/project.py.

The general approach to dataset extraction is to use a combination of string matching between the text and the knowledge base and named entity recognition to generate potential dataset extractions, and then run a classifier on those candidates to decide which candidates to output as predicted extractions.

# Useful commands

All commands should be run through docker

- Build docker image using cache (tagged my_rcc): `./rcc.sh build-use-cache`
- Build docker image (tagged my_rcc): `./rcc.sh build`
- Run model and evaluate: `./rcc.sh run-stop`
- Run organizers' evaluation (after running run-stop): `./rcc.sh evaluate`
- Run whatever command you want using a docker container: `docker run --rm -it -v` pwd`/project:/project -v `pwd`/data:/data --name my_rcc_run my_rcc <your_command_here>`
- Create and save the rule based candidates: `project/create_linking_dataset.py`
- Create and save the NER produced mentions:
`project/generate_ner_output.py`
- Createa and save the featurized dataset of mentions and candidates (this is in the format the Structured Gradient Tree Boosting model expects): `project/create_sgtb_dataset.py`
- Train the Structured Gradient Tree Boosting model (Note: you may want to change the name of the model that gets saved to increment version number): `project/structured_learner.py`
- Train the XGBoost model (Note: you may want to change the name of the model that gets saved to increment version number): `project/xgboost_linking.py`
- Train the Field Classifier: `project/train_field_classifier.sh`

# Training of different components
All commands are expected to be run through Docker

## NER model
This command was run on [Beaker](https://beaker.org/)
`allennlp train project/ner_model/tweaked_parameters_config.json -s <where_to_output_the_model> --include-package ner_rcc`

## Linking model
Before running this command, `project/create_linking_dataset.py` and `project/create_sgtb_dataset.py` need to be run.
`project/xgboost_linking.py`

## Methods prediction
No training required for this component

## Fields prediction
`./project/train_field_classifier.sh`


## Prepare for a new submission by executing the following steps (these can be executed on any machine):
1. cd to your local clone of [our github repo](https://github.com/allenai/rich-context-composition-s2) then: `git checkout master && git pull`. If you don't already have a clone, you'll need to [install git-lfs](https://git-lfs.github.com) then clone it as usual: `git clone git@github.com:allenai/rich-context-composition-s2.git && cd rich-context-composition-s2` but be prepared to wait for ~8 hours for all the PDF files to be cloned locally.
2. prepare the submission file: `cd ../ && zip -r rich-context-composition-s2.zip rich-context-composition-s2/ -x *.git* -x *.spacy -x *.pdf`
3. upload the zipped file to [our team's directory on box](https://app.box.com/folder/55203588026).

## Test the submission file.
1. copy the zipped file to the EC2 instance: `scp rich-context-composition-s2.zip rcc.dev.ai2.in:~/`
2. `ssh rcc.dev.ai2.in`, and start a tmux session: `tmux` or `tmux attach`
3. delete the previous submission (if you had one): `rm -r rich-context-composition-s2/`, unzip the submission file `unzip rich-context-composition-s2.zip`, and cd into the unzipped directory: `cd rich-context-composition-s2`
4. remove previous docker images: `sudo ./rcc.sh remove-docker-image`, then build a new one: `sudo ./rcc.sh build-use-cache` or `sudo ./rcc.sh build`.
5. change the content of the `data/` subdirectory to the test set of interest.
6. check what time it is, then start processing the test set: `./rcc.sh run-stop` and go take a long nap.
7. run automatic evaluation scripts: `./rcc.sh evaluate`

# Component Descriptions

## Dataset citations

We first constructed a set of rule based candidate citations by exact string matching mentions and dataset names from the provided knowledge base. We found this to have high recall on the provided dev fold and our own dev fold that we created. However, after our test submission, it became clear that there were many datasets in the actual test set that did not have mentions in the provided knowledge base. In an attempt to address this issue, we generated more candidates using a Named Entity Recognition (NER) model (https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py) trained on the provided mentions in the knowledge base. Since these mentions were just text, not identifying where in the publication they came from, the training data for this task is quite noisy. The linking candidates for the NER mentions are generated by TFIDF weighted token overlap with the dataset titles. These linking candidates are then passed on to a linking model, an XGBoost (gradient boosted trees) classifier, with a set of hand engineered features. The set of features can be seen in `project/create_sgtb_dataset.py:featurize_candidate_datasets()`. This classifer predicts whether each mention-dataset pair is a correct dataset citation or not. The candidates that are predicted as positive by the linking model are output as citations. The rule based model takes no time to "train" as there is no real training involved, and the XGBoost model is extremely fast to train (it takes a few minutes to do a hyperparameter search, each individual XGBoost classifier takes a few seconds to train).

## Dataset mentions

We construct an NER dataset using the mentions in the provided knowledge base as examples. We then train the crf tagger NER model in AllenNLP on our NER dataset. We use a custom dataset reader based on the conll2003 version from AllenNLP such that we are able to filter out very long sentences and train using a defined percentage of sentences that do not include datasets. Training using the [config that can be found in the ner_model directory](https://github.com/allenai/rich-context-composition-s2/blob/master/project/ner_model/model/config.json) takes approximately 12 hours on a single GPU.

## Research fields

We first build training data for a research fields classifier by collecting paper titles and associated fields from the Microsoft Academic Graph. We perform many preprocessing methods on the training data: 1) filtering the fields that are most relevant to the training data provided, 2) identifying the strongest field of study to paper title relationships and 3) identifying fields of study that have a sufficient amount of associated papers for training. The training data we ended up with included approximately 100K paper titles among two levels, L0 (coarse fields) and L1 (granular fields), with 7 research fields and 32 research fields respectively. For each level, we trained a bi-directional LSTM on the paper titles to predict the corresponding research fields. We additionally incorporate ELMO embeddings to improve performance. For the evaluation publications, we report both the L0 and L1 classification, and only report the L1 classification for a publication if the corresponding score is above some threshold. It takes approximately 1.5hr for the L0 classifier to converge, and it takes approximately 3.5hr for the L1 classifier to converge.

## Research methods

We started by inspecting a subset of the provided papers to get a better understanding of what kind of methods are used in social science and how they are referred to within papers. Based on the inspection, we designed regular expressions which capture common contextual patterns as well as the list of provided SAGE methods. In order to score candidates, we used a background corpus to estimate the salience of candidate methods in a paper. Two additional strategies were attempted but proved unsuccessful: a weakly-supervised model for named entity recognition, and using open information extraction (openIE) to further generalize the list of candidate methods.
