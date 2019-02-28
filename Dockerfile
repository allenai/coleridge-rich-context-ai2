from ubuntu:18.04

# Run apt to install OS packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y tree vim curl python3 python3-pip python3.6-tk git locales locales-all

ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Python 3 package install example
RUN pip3 install ipython==7.1.1 matplotlib==2.2.3 numpy==1.15.4 pandas==0.23.4 scikit-learn==0.20.0 scipy==1.1.0 six==1.11.0 fuzzywuzzy==0.17.0 fuzzysearch==0.5.0 python-Levenshtein==0.12.0 nltk==3.3

RUN python3 -c 'import nltk; nltk.download("stopwords")'

# create directory for "work".
RUN mkdir /work

RUN pip3 install spacy==2.0.18 && \
	python3 -m spacy download en_core_web_sm

RUN git clone https://github.com/allenai/SciSpaCy

COPY project/dist/ dist

RUN pip3 install dist/scispacy-0.1.0-unreleased.tar.gz && \
	pip3 install dist/en_scispacy_core_web_sm-1.0.0.tar.gz
# clone the rich context repo into /rich-context-competition
RUN git clone https://github.com/Coleridge-Initiative/rich-context-competition.git /rich-context-competition

RUN pip3 install xgboost==0.81 allennlp==0.8.1 textacy==0.6.2

RUN pip3 install msgpack==0.5.6 msgpack-numpy==0.4.4.0 thinc==6.12.0
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
LABEL maintainer="jonathan.morgan@nyu.edu"
