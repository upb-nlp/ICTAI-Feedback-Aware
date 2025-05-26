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
gpt_samples = json.load(open("Feedback-Aware-Questions/experiments/gpt4o_all_one_shot.json", "r"))
data_test = data_test

tokenizer_qans = AutoTokenizer.from_pretrained("ORGANIZATION_NAME/llama3_8b_qans", padding_side="left", token='TOKEN_KEY')
tokenizer_qans.pad_token_id = 128002
model_qans = AutoModelForCausalLM.from_pretrained("ORGANIZATION_NAME/llama3_8b_qans", device_map="auto", token='TOKEN_KEY', torch_dtype=torch.bfloat16)

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

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
for i in tqdm(range(len(data_test))):
    data = data_test[i]
    sample = gpt_samples[i]
    try:
        gen_list = []
        text = data['text'] if 'text' in data else data['story_section']

        generated_answers = [x['answer'] for x in sample]
        generated_questions = [x['question'] for x in sample]

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

json.dump(test_generated_list, open('Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_all_gpt4o_one_shot.json', 'w'), indent=4)
