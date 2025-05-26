import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import random

random.seed(42)

PARTITION = 'val' # 'train' 'test' 'val'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "ORGANIZATION_NAME/llama3_8b_qgen"
access_token = 'TOKEN_KEY'

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, device_map="auto", torch_dtype=torch.bfloat16)

dataset = json.load(open(f"Failed-EMNLP/tasa_dataset_preparation/tasa_{PARTITION}_syntax_tree.json", 'r'))

batch_size = 8

for data in tqdm(dataset):
    # Sample 10% of the answers
    data['sampled_answers_questions'] = random.sample(data['answers_questions'], int(len(data['answers_questions'])*0.1))
    answers_list = data['sampled_answers_questions']

    responses = []
    for i in range(0, len(answers_list), batch_size):
        end_interval = min(i+batch_size, len(answers_list))

        texts = [f"Generate a question based on the context and the answer.\nContext: {data['text']}\nAnswer: {ans['answer']}\n### Response:" for ans in answers_list[i:end_interval]]

        model_inputs = tokenizer(texts, return_tensors="pt", padding='longest', truncation=True, max_length=2048*2).to(device)

        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=300,
            pad_token_id=128002,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        responses += response

    for pair, response in zip(answers_list, responses):
        pair['generated_question'] = response
    
    data['answers_questions'] = answers_list

json.dump(dataset, open(f"Failed-EMNLP/tasa_dataset_preparation/tasa_{PARTITION}_answers_questions.json", 'w'), indent=4)