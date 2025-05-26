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

data_test = data_test

model_name = "ORGANIZATION_NAME/llama3_8b_questions_baseline_single_answer_orpo"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY')
tokenizer.pad_token_id = 128002

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY')

tokenizer_qans = AutoTokenizer.from_pretrained("ORGANIZATION_NAME/llama3_8b_qans", padding_side="left", token='TOKEN_KEY')
tokenizer_qans.pad_token_id = 128002
model_qans = AutoModelForCausalLM.from_pretrained("ORGANIZATION_NAME/llama3_8b_qans", device_map="auto", token='TOKEN_KEY', torch_dtype=torch.bfloat16)

tokenizer_qgen = AutoTokenizer.from_pretrained("ORGANIZATION_NAME/llama3_8b_qgen", padding_side="left", token='TOKEN_KEY')
tokenizer_qgen.pad_token_id = 128002
model_qgen = AutoModelForCausalLM.from_pretrained("ORGANIZATION_NAME/llama3_8b_qgen", device_map="auto", token='TOKEN_KEY', torch_dtype=torch.bfloat16)

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

def generate_questions(text, answers, batch_size):
    all_questions = []
    for i in range(0, len(answers), batch_size):
        end_interval = min(i+batch_size, len(answers))
        batch_answers = answers[i:end_interval]

        text_inputs = [f"Generate a question based on the context and the answer.\nContext: {text}\nAnswer: {a}\n### Response:" for a in batch_answers]
        model_inputs = tokenizer_qgen(text_inputs, return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(device)
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
        all_questions += tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return all_questions

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

def get_label_from_response(answer, answer_type, gen_list):
    if is_response_repeated(answer, gen_list):
        return 'REPEATED'
    return answer_type


test_generated_list = []
for data in tqdm(data_test):
    try:
        gen_list = []
        text = data['text'] if 'text' in data else data['story_section']

        task_description_prompt = "Select a span from the following text that would serve as good answer for generating a question."
        prompt = f"{task_description_prompt}\nText:\n{text}\n### Response:"

        model_inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(device)
        responses = []
        losses = []
        for i in range(4):
            rsp, lss = generate_response(model_inputs)
            responses += rsp
            losses += lss
        responses, losses = remove_duplicates(responses, losses)
        generated_answers = get_topk_responses(responses, losses, 25)
        generated_questions = generate_questions(text, generated_answers, 2)
        answer_types_probs = [get_answer_score(text, gq, ga) for gq, ga in zip(generated_questions, generated_answers)]
        answer_types = [at for at, _ in answer_types_probs]
        probs_qa = [prob for _, prob in answer_types_probs]

        for answer, question, answer_type, prob_qa in zip(generated_answers, generated_questions, answer_types, probs_qa):
            label = get_label_from_response(answer, answer_type, [x['answer'] for x in gen_list])
            gen_list.append({
                'answer': answer,
                'question': question,
                'answer_type': answer_type,
                'prob_qa': prob_qa,
                'label': label
            })

        test_generated_list.append(gen_list)
    except Exception as e:
        test_generated_list.append([])
        print(f"Error at {len(test_generated_list)}")

json.dump(test_generated_list, open('Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_single_answer_orpo_smart_duplicates.json', 'w'), indent=4)
