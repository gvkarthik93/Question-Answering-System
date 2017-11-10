from nltk.tag.stanford import StanfordNERTagger
import json

def read_json(file):
    with open(file, 'r') as f:
        text = json.load(f)
    return text

def main():
	st = StanfordNERTagger('/Users/shubhambarhate/Desktop/project3/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz',
           '/Users/shubhambarhate/Desktop/project3/stanford-ner-2017-06-09/stanford-ner.jar')
	sample_file = "sample.json"
	train_file = "training.json"
	test_file = "testing.json"
    
    
	sample_text = read_json(sample_file)
	train_text = read_json(train_file)
	test_text = read_json(test_file)
	#print(test_text['data'][0]['paragraphs'][0]['qas'][1]['question'])
	print(test_text['data'][0]['paragraphs'][0]['context'])
	#print(test_text['data'][0]['paragraphs'][0]['qas'][0])
	#print(test_text['data'][0]['paragraphs'][0]['qas'][0]['question'])
	print(test_text['data'][0]['paragraphs'][0]['qas'][0]['id'])
	print(test_text['data'][0]['paragraphs'][0]['qas'][0]['answers'])
	#print(test_text['data'][0]['paragraphs'][0]['qas'][0]['answers_start'])

	#print(st.tag('Rami Eid is studying at Stony Brook University in Bengaluru'.split()))
    
main()


