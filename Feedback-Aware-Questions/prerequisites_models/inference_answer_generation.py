import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_hotpot = json.load(open("Feedback-Aware-Questions/prerequisites_models/reshaped_hotpotqa_test.json", 'r'))
dataset_narrativeqa = json.load(open("Feedback-Aware-Questions/prerequisites_models/reshaped_narrativeqa_test.json", 'r'))
dataset_squad = json.load(open("Feedback-Aware-Questions/prerequisites_models/reshaped_squad_test.json", 'r'))
dataset_fairytaleqa = json.load(open("Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_test.json", 'r'))

for data in dataset_fairytaleqa:
    data['context'] = data['story_section']
    data['question'] = data['question']
    data['answer'] = data['answer1']

model_name = "ORGANIZATION_NAME/llama3_8b_qans"
access_token = 'TOKEN_KEY'

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=access_token)
tokenizer.pad_token_id = 128002
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=access_token, torch_dtype=torch.bfloat16)

def generate_answer(texts, questions):
    model_inputs = tokenizer([f"Answer the following question based on the context.\nContext: {text}\nQuestion: {question}\n### Response:" for text, question in zip(texts, questions)], return_tensors="pt", max_length=2048, truncation=True, padding='longest').to(device)
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=30,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
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
    return response


for data_test, data_name in zip([dataset_fairytaleqa, dataset_hotpot, dataset_narrativeqa, dataset_squad], ["fairytaleqa", "hotpotqa", "narrativeqa", "squad"]):
    batch_size = 4
    responses = []
    for i in tqdm(range(0, len(data_test), batch_size)):
        end_interval = min(i+batch_size, len(data_test))

        texts = [data['context'] for data in data_test[i:end_interval]]
        questions = [data['question'] for data in data_test[i:end_interval]]

        response = generate_answer(texts, questions)
        responses += response

    for data, response in zip(data_test, responses):
        data['generated_answer'] = response

    json.dump(data_test, open(f"Feedback-Aware-Questions/prerequisites_models/generated_answers_{data_name}_deterministic.json", "w"), indent=4)
