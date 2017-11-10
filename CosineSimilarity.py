import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

def stem_tokens(tokens):
	stemmer = nltk.stem.porter.PorterStemmer()
	return [stemmer.stem(item) for item in tokens]


'''remove punctuation, lowercase, stem'''
def normalize(text):
	remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
	#return stem_tokens(nltk.word_tokenize(text.lower().translate(string.punctuation)))
	return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


def getCosineSimilarity(text1, text2):

	vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')	
	tfidf = vectorizer.fit_transform([text1, text2])
	return ((tfidf * tfidf.T).A)[0,1]


def main():
	print (getCosineSimilarity('a little bird', 'a little bird'))
	print (getCosineSimilarity('a little bird', 'a little bird chirps'))
	

if __name__ == "__main__":
	main()