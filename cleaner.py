import os
import re
import json
import spacy
spacy_nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
from spacy.lang.en.stop_words import STOP_WORDS as stopwords_en



class TextCleaner(object):

	def __init__(self):
		self.normalize_mapping = json.load(open("data/normalize_mapping.json"))

	def clean_text(self, text):
			try:
				text = str(text)
				text = re.sub(r"[^a-zA-Z]", " ", text)
				text = re.sub(r"\s+", " ", text)
				text = text.lower().strip()
			except Exception as e:
				print("\n Error in clean_text --- ", e,"\n ", traceback.format_exc())
				print("\n Error sent --- ", text)
			return text

	def get_nouns(self, text):
		return [tok.text for tok in spacy_nlp(text) if tok.tag_ in ["NN"]]

	def normalize(self, text):
		return " ".join([self.normalize_mapping.get(tok, tok) for tok in text.split()])

	def get_lemma_tokens(self, text):
		return [tok.lemma_.lower().strip() for tok in spacy_nlp(text) if (tok.lemma_ != '-PRON-' and tok.lemma_ not in stopwords_en and len(tok.lemma_)>1)]

	def cleaning_pipeline(self, text):
		text = self.normalize(text)
		text = self.clean_text(text)
		text = self.get_lemma_tokens(text)
		return text