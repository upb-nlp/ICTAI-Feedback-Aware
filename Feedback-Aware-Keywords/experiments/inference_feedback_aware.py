import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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

# Set the seed to 42 for random and torch and transformers
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_test = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_test.json', 'r'))
#start_idx = 1000
#end_idx = 1998
#data_test = data_test[start_idx:end_idx]

model_name = "ORGANIZATION_NAME/llama3_8b_keywords_feedback_aware_orpo"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY')
tokenizer.pad_token_id = 128002
eos_ids = [id for key, id in tokenizer.vocab.items() if "}" in key]

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY')

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

GENERATION_START = "GOOD {"

def get_initial_prompt_for_inference(data):
    task_description_prompt = f"Iteratively select keywords for the following text."
    prompt_sentences = data['abstract']
    
    prompt = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response: \n"

    return prompt

def tokenize_prompt(list_prompt):
    return tokenizer(list_prompt, return_tensors="pt", max_length=2048*4, truncation=True).to(device)

def generate_response(model_inputs):
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        generation_config=GenerationConfig(
            max_new_tokens=30,
            do_sample=True,
            temperature=None,
            top_p=0.8,
            top_k=20,
            num_return_sequences=10,
            eos_token_id=eos_ids,
            pad_token_id=eos_ids[0],
            output_logits=True,
            return_dict_in_generate=True,
        )
    )
    
    generated_ids_sequences = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip([model_inputs.input_ids[0]] * 10, generated_ids['sequences'])
    ]


    generated_ids_logits = torch.transpose(torch.stack(generated_ids['logits']), 0, 1)

    losses = []

    for i in range(len(generated_ids_sequences)):
        end_idx_sequence = len(generated_ids_sequences[i])
        for eos_id in eos_ids:
            if eos_id in generated_ids_sequences[i]:
                end_idx_sequence = min(generated_ids_sequences[i].tolist().index(eos_id) + 1, end_idx_sequence)
        generated_ids_sequences[i] = generated_ids_sequences[i][:end_idx_sequence]

        good_logit = generated_ids_logits[i][:end_idx_sequence]

        loss = loss_fn(
                good_logit,
                generated_ids_sequences[i],
        )
        losses.append(loss.item())

    response = tokenizer.batch_decode(generated_ids_sequences, skip_special_tokens=True)
    return response, losses

def get_top_responses(responses, losses, gen_list_responses):
    all_responses = [responses[i] for i in np.argsort(losses)]
    all_responses = [extract_keyword_from_response(response) for response in all_responses]
    # Remove repeated from the responses

    all_responses_filtered = [response for response in all_responses if not is_response_repeated(response, gen_list_responses)]
    if len(all_responses_filtered) == 0:
        return all_responses
    return all_responses_filtered

def get_keyword_score(keyword, ground_truth):
    for gt in ground_truth:
        if (levenshtein_distance(keyword.lower(), gt.lower()) < 3 and len(keyword) > 3 and len(gt) > 3) or keyword.lower() == gt.lower():
            return 'GOOD'
    return 'BAD'

def extract_keyword_from_response(response):
    # Remove the rest of the string response after '}'
    response = response.split('}')[0]
    response = response.replace('}', '')
    return response.strip()

def get_label_from_response(keyword, keyword_type, gen_list):
    if is_response_repeated(keyword, gen_list):
        return 'REPEATED'

    return keyword_type

def get_next_prompt(prompt, keyword, label):
    return f"{prompt}{label}" + " { " + keyword + ' }' + "\n"

max_iter = 10
test_generated_list = []
for data in tqdm(data_test):
    text = data['abstract']
    iter = 0
    generated_keywords = []
    prompt = get_initial_prompt_for_inference(data)
    gen_list = []
    while iter < max_iter:
        iter += 1
        try:
            model_inputs = tokenize_prompt(prompt + GENERATION_START)
            responses, losses = generate_response(model_inputs)
            response = get_top_responses(responses, losses, [x['keyword'] for x in gen_list])[0]
            keyword = response
            keyword_type = get_keyword_score(keyword, data['keywords'])
            label = get_label_from_response(keyword, keyword_type, [x['keyword'] for x in gen_list])
            gen_list.append({
                'response': response,
                'keyword': keyword,
                'keyword_type': keyword_type,
                'label': label,
            })

            prompt = get_next_prompt(prompt, keyword, label)
        except Exception as e:
            print(e)
            break
    test_generated_list.append(gen_list)

json.dump(test_generated_list, open(f"Feedback-Aware-Keywords/experiments/test_keywords_feedback_aware_orpo_smart_duplicates.json", 'w'), indent=4)
