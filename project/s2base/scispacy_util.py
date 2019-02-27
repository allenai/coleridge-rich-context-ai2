import spacy

class SciSpaCyParser(object):
    def __init__(self):
        self.nlp = spacy.load('en_scispacy_core_web_sm')

    def remove_new_lines(self, text):
        """Used to preprocess away new lines in the middle of words. This function
           is intended to be called on a raw string before it is passed through a
           spaCy pipeline

        @param text: a string of text to be processed
        """
        text = text.replace("-\n\n", "")
        text = text.replace("- \n\n", "")
        text = text.replace("-\n", "")
        text = text.replace("- \n", "")
        return text

    def preprocess_text(self, text):
        """Function to preprocess text before passing it on to spacy

        @param text: the raw text to process
        """
        text = self.remove_new_lines(text)
        return text

    def postprocess_doc(self, doc):
        """Function to postprocess a doc before returning it for use.
           This post processing could be done by converting the doc to an array,
           processing out what you don't want, and then converting the array back
           to a doc.

        @param doc: a spacy processed doc
        """
        return doc

    def scispacy_create_doc(self, text):
        """Function to use SciSpaCy instead of spaCy. Intended usage is to replace
           instances of `nlp = spacy.load("<model_name>")` with `nlp = scispacy_create_doc`
        
        @param text: the text to be processed into a spacy doc
        """
        text = self.preprocess_text(text)
        doc = self.nlp(text)
        doc = self.postprocess_doc(doc)
        return doc
