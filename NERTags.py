from nltk.tag.stanford import StanfordNERTagger
from nltk import word_tokenize

#gets the NER tags for a probable answer setnence
def get_ner_tags(answer):

	#download teh stanford ner zip file and extract it .
	#Change the directory location in the folllowing path
	modelfile = '/Users/karthik/Development/QA/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz'
	st = StanfordNERTagger(model_filename=modelfile, path_to_jar='/Users/karthik/Development/QA/stanford-ner-2017-06-09/stanford-ner.jar')
	sentence = word_tokenize(answer)
	return st.tag(sentence)