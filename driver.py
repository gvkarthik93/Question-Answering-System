
# coding: utf-8

# In[102]:

import json
from nltk.tag.stanford import StanfordNERTagger
from QuestionAnswer import get_para_answer
from NERTags import get_ner_tags

def read_json(file):
    with open(file, 'r') as f:
        text = json.load(f)
    return text

def convert_input_to_dict(text):
    data_length = len(text['data'])
    context_list = list()
    test_dict = dict()
    for i in range(data_length):
        data = text['data'][i]
        para_length = len(data['paragraphs'])
        para = data['paragraphs']
        #print(para_length)
        
        for j in range(para_length):
            key = len(context_list)
            context_list.insert(key, para[j]['context'])
            if key not in test_dict:
                test_dict[key] = dict()
                
            qas_length = len(para[j]['qas'])
            qas = para[j]['qas']
            
            for k in range(qas_length):
                if qas[k]['question'] not in test_dict[key]:
                    test_dict[key][qas[k]['question']] = dict()
                test_dict[key][qas[k]['question']] = str(qas[k]['id'])
                #print(str(qas[k]['question']) +" : " +str(qas[k]['id']))
        
    return test_dict, context_list
    

def generate_output_json(dictionary):
    filename = "output4.json"
    with open(filename, 'w') as f:
        json.dump(dictionary, f)
    
# In[103]:

def main():
    
    sample_file = "sample.json"
    train_file = "training.json"
    test_file = "testing.json"

    #sample_text = read_json(sample_file)
    #train_text = read_json(train_file)
    
    #test_text contains the json text for testing.json
    test_text = read_json(test_file)

    #context_list contains paragraphs. context_list[0] contains first paragraph and so on
    #test_dict is a nested dictionary , For the first paragraph, 0 is the key, and value is a dictionary of question and ids pairs
    # 0 : 
    #      "Quesgion 1" : "id1"
    #      "Quesgion 2" : "id2"

    # Reading and processing Testing data
    test_dict, context_list = convert_input_to_dict(test_text)
    
    #ans_dict will have q id as key and answer phrase as the value
    traing_ans_dict = dict()
    ans_dict = dict()
    index = 0
    training_index = 0

    #this just gets answers for the questions associated with the first paragraph
    dictionary = get_para_answer(0, context_list[0], test_dict[0])
    ans_dict.update(dictionary)

    #generic loop for iterating through every paragraph and and getting answers for its questions
    #for i in context_list:
    #    dictionary = get_para_answer(index, i, test_dict[index])
    #    ans_dict.update(dictionary)
    #    index += 1
    #    print ("Processed Number of Paragraphs: ",index)

    question_count = 0
    questionList = []
    for key,value in test_dict.items():
        for i,j in value.items():
            question_count += 1
            questionList.append(i)
    print ("Number of questions: ",question_count)

    anscount = 0
    ner_tag_list = list()
    dev_index = 0
    
    for key, value in ans_dict.items():
        processed_answer = str(value)
        processed_question = questionList[dev_index]

        print ("Question: ", processed_question)
        print ("Baseline Answer: ", processed_answer)

        question_split_list = processed_question.split()        
        nerlist = get_ner_tags(processed_answer)

        final_answer = ""
        for word, tag in nerlist:
            if tag != "O":
                if "where" in question_split_list:
                    if tag == "ORGANIZATION" or tag == "LOCATION" and word not in question_split_list:
                        final_answer = " " + final_answer + " " + word

                elif "whose" in question_split_list or "who" in question_split_list or "whom" in question_split_list:
                    if tag == "ORGANIZATION" or tag == "PERSON" and word not in question_split_list:
                        final_answer = " " + final_answer+ " " + word

                elif "which" in question_split_list or "what" in question_split_list:
                    if tag == "ORGANIZATION" or tag == "PERSON" or tag == "LOCATION":
                        final_answer = " " + final_answer + " " + word

                elif "when" in question_split_list :
                    if tag == "DATE" or tag == "TIME" and word not in question_split_list:
                        final_answer = " " + final_answer + " " + word

                else:
                    final_answer = " " + final_answer + " " + word

        if final_answer == "":
            final_answer = processed_answer
        ans_dict[key] = final_answer.strip()
        print ("Final Answer: ", final_answer.strip())
        print ("Processed Answer: ",dev_index)
        dev_index += 1

    print ("Number of answers: ", dev_index)    
    print ("Done Processing")
    
    #for i in ner_tag_list:
    #    print (i)

    #Used to generate the output json file
    generate_output_json(ans_dict)
    
    
    
if __name__ == "__main__":
    main()


# In[ ]:



