import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import numpy as np
import random
import spacy

nlp = spacy.load("en_core_web_md")
def extract_lemmas(text):
    doc = nlp(text)
    lemmatized_words = []
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADV', 'ADJ', 'AUX']:
            lemmatized_words.append(token.lemma_)

    bag_of_words = set(lemmatized_words)
    return bag_of_words

def is_response_repeated(response, gen_list_responses):
    lemma_response = extract_lemmas(response)
    for gen_response in gen_list_responses:
        lemma_gen_response = extract_lemmas(gen_response)
        if response == gen_response or len(lemma_response.intersection(lemma_gen_response)) > 0:
            return True
    return False

def levenshtein_distance(s1, s2):
    dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1
                )
    
    return dp[-1][-1]

def get_label_from_response(keyword, keyword_type, gen_list):
    if is_response_repeated(keyword, gen_list):
        return 'REPEATED'
    return keyword_type

def get_keyword_score(keyword, ground_truth):
    for gt in ground_truth:
        if (levenshtein_distance(keyword.lower(), gt.lower()) < 3 and len(keyword) > 3 and len(gt) > 3) or keyword.lower() == gt.lower():
            return 'GOOD'
    return 'BAD'

# Set the seed to 42 for random and torch and transformers
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_test = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_test.json', 'r'))
gpt_samples = json.load(open("Feedback-Aware-Keywords/experiments/gpt4o_all_one_shot.json", "r"))

for i in tqdm(range(len(data_test))):
    data = data_test[i]
    sample = gpt_samples[i]
    test_generated_list = []

    for key_data in sample:
        keyword = key_data['keyword']
        key_data['keyword_type'] = get_keyword_score(keyword, data['keywords'])
        key_data['label'] = get_label_from_response(keyword, key_data['keyword_type'], test_generated_list)

        test_generated_list.append(keyword)

    gpt_samples[i] = sample

json.dump(gpt_samples, open("Feedback-Aware-Keywords/experiments/test_keywords_all_gpt4o_one_shot.json", "w"), indent=4)