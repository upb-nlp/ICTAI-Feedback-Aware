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

# Set the seed to 42 for random and torch and transformers
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_test = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_test.json', 'r'))

data_test = data_test

model_name = "ORGANIZATION_NAME/llama3_8b_keywords_baseline_single_keyword_generation_sft"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY')
tokenizer.pad_token_id = 128002

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY')

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

def generate_response(model_inputs):
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=30,
        do_sample=True,
        temperature=None,
        top_p=0.8,
        top_k=20,
        num_return_sequences=25,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        output_logits=True,
        return_dict_in_generate=True,
    )
    
    generated_ids_sequences = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip([model_inputs.input_ids[0]] * 25, generated_ids['sequences'])
    ]

    generated_ids_logits = torch.transpose(torch.stack(generated_ids['logits']), 0, 1)

    losses = []

    for i in range(len(generated_ids_sequences)):
        end_idx_sequence = generated_ids_sequences[i].tolist().index(tokenizer.eos_token_id) + 1 if tokenizer.eos_token_id in generated_ids_sequences[i] else len(generated_ids_sequences[i])
        generated_ids_sequences[i] = generated_ids_sequences[i][:end_idx_sequence]

        good_logit = generated_ids_logits[i][:end_idx_sequence]

        loss = loss_fn(
                good_logit,
                generated_ids_sequences[i],
        )
        losses.append(loss.item())

    response = tokenizer.batch_decode(generated_ids_sequences, skip_special_tokens=True)
    return response, losses

def get_topk_responses(responses, losses, k):
    return [responses[i] for i in np.argsort(losses)[:k]]

def remove_duplicates(responses, losses):
    new_responses = []
    new_losses = []
    for response, loss in zip(responses, losses):
        if not is_response_repeated(response.strip(), new_responses):
            new_responses.append(response.strip())
            new_losses.append(loss)
    return new_responses, new_losses

def get_keyword_score(keyword, ground_truth):
    for gt in ground_truth:
        if (levenshtein_distance(keyword.lower(), gt.lower()) < 3 and len(keyword) > 3 and len(gt) > 3) or keyword.lower() == gt.lower():
            return 'GOOD'
    return 'BAD'

def get_label_from_response(keyword, keyword_type, gen_list):
    if is_response_repeated(keyword, gen_list):
        return 'REPEATED'
    return keyword_type


test_generated_list = []
for data in tqdm(data_test):
    try:
        gen_list = []
        text = data['abstract']

        task_description_prompt = "Select keywords for the following text."
        prompt = f"{task_description_prompt}\nText:\n{text}\n### Response:"

        model_inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(device)
        responses = []
        losses = []
        for i in range(4):
            rsp, lss = generate_response(model_inputs)
            responses += rsp
            losses += lss
        responses, losses = remove_duplicates(responses, losses)
        generated_keywords = get_topk_responses(responses, losses, 10)
        keyword_types = [get_keyword_score(gk, data['keywords']) for gk in generated_keywords]

        for keyword, keyword_type in zip(generated_keywords, keyword_types):
            label = get_label_from_response(keyword, keyword_type, [x['keyword'] for x in gen_list])
            gen_list.append({
                'keyword': keyword,
                'keyword_type': keyword_type,
                'label': label
            })

        test_generated_list.append(gen_list)
    except Exception as e:
        test_generated_list.append([])
        print(f"Error at {len(test_generated_list)}")

json.dump(test_generated_list, open('Feedback-Aware-Keywords/experiments/test_keywords_baseline_single_keyword_sft_smart_duplicates.json', 'w'), indent=4)
