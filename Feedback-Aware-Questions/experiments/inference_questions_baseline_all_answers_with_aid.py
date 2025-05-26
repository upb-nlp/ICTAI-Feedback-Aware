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

# Set the seed to 42 for random and torch and transformers
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_test = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_test.json', 'r')) + json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_test.json', 'r'))

model_name = "ORGANIZATION_NAME/llama3_8b_questions_answer_selection_baseline_all_pairs_with_aid_generation_v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY')
tokenizer.pad_token_id = 128002
eos_id = tokenizer.encode(' }')[1]

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY')

tokenizer_qans = AutoTokenizer.from_pretrained("ORGANIZATION_NAME/llama3_8b_qans", padding_side="left", token='TOKEN_KEY')
tokenizer_qans.pad_token_id = 128002
model_qans = AutoModelForCausalLM.from_pretrained("ORGANIZATION_NAME/llama3_8b_qans", device_map="auto", token='TOKEN_KEY', torch_dtype=torch.bfloat16)

tokenizer_qgen = AutoTokenizer.from_pretrained("ORGANIZATION_NAME/llama3_8b_qgen", padding_side="left", token='TOKEN_KEY')
tokenizer_qgen.pad_token_id = 128002
model_qgen = AutoModelForCausalLM.from_pretrained("ORGANIZATION_NAME/llama3_8b_qgen", device_map="auto", token='TOKEN_KEY', torch_dtype=torch.bfloat16)

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

GENERATION_START = "{"

def get_initial_prompt_for_inference(data):
    task_description_prompt = f"Iteratively select a span from the following text that would serve as good answer for generating a question."
    prompt_sentences = data['text'] if 'text' in data else data['story_section']
    
    prompt = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response: \n"

    return prompt

def tokenize_prompt(list_prompt):
    return tokenizer(list_prompt, return_tensors="pt", max_length=2048*4, truncation=True).to(device)

def generate_response(model_inputs):
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=30,
        do_sample=True,
        temperature=None,
        top_p=0.8,
        top_k=20,
        num_return_sequences=10,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
        output_logits=True,
        return_dict_in_generate=True,
    )
    
    generated_ids_sequences = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip([model_inputs.input_ids[0]] * 10, generated_ids['sequences'])
    ]

    generated_ids_logits = torch.transpose(torch.stack(generated_ids['logits']), 0, 1)

    losses = []

    for i in range(len(generated_ids_sequences)):
        end_idx_sequence = generated_ids_sequences[i].tolist().index(eos_id) + 1 if eos_id in generated_ids_sequences[i] else len(generated_ids_sequences[i])
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
    all_responses = [extract_answer_from_response(response) for response in all_responses]
    # Remove repeated from the responses

    all_responses_filtered = [response for response in all_responses if not is_response_repeated(response, gen_list_responses)]
    if len(all_responses_filtered) == 0:
        return all_responses
    return all_responses_filtered

def generate_question(text, answer):
    model_inputs = tokenizer_qgen(f"Generate a question based on the context and the answer.\nContext: {text}\nAnswer: {answer}\n### Response:", return_tensors="pt", max_length=2048*4, truncation=True).to(device)
    generated_ids = model_qgen.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=30,
        eos_token_id=tokenizer_qgen.eos_token_id,
        pad_token_id=tokenizer_qgen.eos_token_id,
        do_sample=False,
        temperature=None,
        top_p=None,
        penalty_alpha=0.5,
        top_k=4,
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response[0]

def get_answer_loss(text, question, answer):
    texts_input = [f"Answer the following question based on the context.\nContext: {text}\nQuestion: {question}\n### Response:"]
    texts_whole = [f"Answer the following question based on the context.\nContext: {text}\nQuestion: {question}\n### Response: {answer}"]

    input_prompts = tokenizer_qans(texts_input, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)
    whole_prompts = tokenizer_qans(texts_whole, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)

    # Add the eos token to the whole prompts, for the whole batch
    whole_prompts['input_ids'] = torch.cat((whole_prompts['input_ids'], torch.tensor([[tokenizer_qans.eos_token_id]] * whole_prompts['input_ids'].shape[0], device=device)), dim=1)
    whole_prompts['attention_mask'] = torch.cat((whole_prompts['attention_mask'], torch.tensor([[1]] * whole_prompts['attention_mask'].shape[0], device=device)), dim=1)


    with torch.no_grad():
        outputs = model_qans(input_ids=whole_prompts['input_ids'], attention_mask=whole_prompts['attention_mask'])
        logits = outputs.logits

    for logit, input, whole in zip(logits, input_prompts['input_ids'], whole_prompts['input_ids']):
        # Remove padding
        padding = torch.count_nonzero(whole == tokenizer_qans.pad_token_id)
        whole = whole[padding:]
        padding = torch.count_nonzero(input == tokenizer_qans.pad_token_id)
        input = input[padding:]

        # Remove the last logit (unnecessary, automatically added by the model)
        logit = logit[:-1]

        # Get from the logits just the ones corresponding to the actual generation (label)
        good_logit = logit[-(len(whole) - len(input)):]

        # Get the label
        good_label = whole[len(input):]

        loss = loss_fn(
            good_logit,
            good_label,
        )
        return loss.item()

def prob(loss):
    return np.exp(-loss)

def get_answer_score(text, question, answer):
    QA_PROB_TRESHOLD = 0.03
    PROB_RAPPORT_TRESHOLD = 1.0
    prob_qa = prob(get_answer_loss(text, question, answer))
    prob_without_context = prob(get_answer_loss("No context.", question, answer))

    if prob_qa < QA_PROB_TRESHOLD or prob_without_context/prob_qa > PROB_RAPPORT_TRESHOLD:
        return 'BAD', prob_qa
    else:
        return 'GOOD', prob_qa

def extract_answer_from_response(response):
    # Remove the rest of the string response after '}'
    response = response.split('}')[0]
    response = response.replace('}', '')
    return response.strip()

def get_label_from_response(answer, answer_type, gen_list):
    if is_response_repeated(answer, gen_list):
        return 'REPEATED'

    return answer_type

def get_next_prompt(prompt, answer, question):
    return f"{prompt}" + "{ " + answer + ' } ' + question + "\n"

max_iter = 25
test_generated_list = []
for data in tqdm(data_test):
    text = data['text'] if 'text' in data else data['story_section']
    iter = 0
    generated_answers = []
    prompt = get_initial_prompt_for_inference(data)
    gen_list = []
    while iter < max_iter:
        iter += 1
        try:
            model_inputs = tokenize_prompt(prompt + GENERATION_START)
            responses, losses = generate_response(model_inputs)
            response = get_top_responses(responses, losses, [x['response'] for x in gen_list])[0]
            answer = response
            question = generate_question(text, answer)
            answer_type, prob_qa = get_answer_score(text, question, answer)
            label = get_label_from_response(answer, answer_type, [x['answer'] for x in gen_list])
            gen_list.append({
                'response': response,
                'answer': answer,
                'question': question,
                'answer_type': answer_type,
                'prob_qa': prob_qa,
                'label': label,
            })

            prompt = get_next_prompt(prompt, answer, question)
        except Exception as e:
            print(e)
            break
    test_generated_list.append(gen_list)

json.dump(test_generated_list, open(f"Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_all_answers_with_aid_sft_smart_duplicates.json", 'w'), indent=4)
