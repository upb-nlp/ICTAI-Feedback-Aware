from tqdm import tqdm
import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import spacy
import benepar


benepar.download('benepar_en3')
nlp = spacy.load("en_core_web_md")
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

def get_syntax_tree_nodes(sentence):
    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    
    # Function to recursively collect texts from the nodes
    def collect_node_texts(node):
        texts.append(node.text)
        for child in node._.children:
            collect_node_texts(child)
    
    texts = []
    collect_node_texts(sent)
    
    return texts

for PARTITION in ['train', 'test', 'val']:
    dataset = json.load(open(f"Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_{PARTITION}.json", 'r'))

    for data in tqdm(dataset):
        sentences = sent_tokenize(data['story_section'])

        answers_list = []
        for sentence in sentences:
            answers_list += get_syntax_tree_nodes(sentence)
        answers_list = list(set(answers_list))
        
        answers_dicts = [{'answer': answer} for answer in answers_list]    
        data['answers_questions'] = answers_dicts

    json.dump(dataset, open(f"Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_{PARTITION}_syntax_tree.json", 'w'), indent=4)