from CosineSimilarity import getCosineSimilarity
from NERTags import get_ner_tags
import spacy
import nltk

window_length = 10

from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

def testFuncOld():
    text = 'hello bye the the hi'
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])

def testFuncNew():
    text = 'hello bye the the hi'
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])

if __name__ == "__main__":
    for i in xrange(10000):
        testFuncOld()
        testFuncNew()

#generated list of paragraph strings of a particular window length
def get_sliding_window(para):

	a=para.split()
	b = [a[i:i+window_length] for i in range(len(a)- (window_length -1))]
	c = []
	for i  in b:
		c.append(" ".join(i))
	return c

def get_most_probable_answer(probable_ans):
	ner = get_ner_tags(probable_ans)
	print(ner)
	return probable_ans

#Uses Cosine Similarity to get the most probable sentenc, where we expect to discover our answer
def get_answer(window, q):
	maxSimilarity = -float("inf")
	for i in window:
		if getCosineSimilarity(i, q) > maxSimilarity:
			maxSimilarity = getCosineSimilarity(i, q)
			probable_ans = i

	#probable_ans = get_most_probable_answer(probable_ans)
	return probable_ans

def get_similarity_answer(para, question):
	qtokens = nltk.word_tokenize(question)
	para_sentences = para.split(".")

	sentence = ""
	maxValue = 0
	context_span = 0
	for i in range(len(para_sentences)):
		doc = para_sentences[i]
		#dtokens = nltk.word_tokenize(doc)
		#common_word_count = len(set.intersection(set(qtokens), set(dtokens)))
		common_word_count = getCosineSimilarity(doc, question)
		if common_word_count > maxValue:
			sentence = doc
			maxValue = common_word_count
			context_span = i

	return sentence 
	#+ ". " + para_sentences[context_span+1]


#calls the get_answer method for every questions and stores the answer sentence corresponding to every Question id
def get_para_answer(index, para, q_dict):
	output_dict = dict()

	window =get_sliding_window(para)
	
	for i in q_dict:
		id = q_dict[i]
		#output_dict[id] = get_answer(window, i)
		output_dict[id] = get_similarity_answer(para, i)
	return output_dict
	