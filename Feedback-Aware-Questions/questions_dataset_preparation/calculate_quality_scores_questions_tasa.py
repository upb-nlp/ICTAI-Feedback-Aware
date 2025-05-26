import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm


PARTITION = 'val' # 'train' 'test' 'val'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = json.load(open(f'Failed-EMNLP/tasa_dataset_preparation/tasa_{PARTITION}_answers_questions.json'))

model_name = "ORGANIZATION_NAME/llama3_8b_qans"
access_token = 'TOKEN_KEY'

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer.pad_token_id = 128002

batch_size = 8
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
for data in tqdm(dataset):
    all_losses = []
    for i in range(0, len(data['sampled_answers_questions']), batch_size):
        end_interval = min(i+batch_size, len(data['sampled_answers_questions']))

        texts_input = [f"Answer the following question based on the context.\nContext: No context.\nQuestion: {ans['generated_question']}\n### Response:" for ans in data['sampled_answers_questions'][i:end_interval]]
        texts_whole = [f"Answer the following question based on the context.\nContext: No context.\nQuestion: {ans['generated_question']}\n### Response: {ans['answer']}" for ans in data['sampled_answers_questions'][i:end_interval]]

        input_prompts = tokenizer(texts_input, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)
        whole_prompts = tokenizer(texts_whole, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)

        # Add the eos token to the whole prompts, for the whole batch
        whole_prompts['input_ids'] = torch.cat((whole_prompts['input_ids'], torch.tensor([[tokenizer.eos_token_id]] * whole_prompts['input_ids'].shape[0], device=device)), dim=1)
        whole_prompts['attention_mask'] = torch.cat((whole_prompts['attention_mask'], torch.tensor([[1]] * whole_prompts['attention_mask'].shape[0], device=device)), dim=1)

        with torch.no_grad():
            outputs = model(input_ids=whole_prompts['input_ids'], attention_mask=whole_prompts['attention_mask'])
            logits = outputs.logits

        for logit, input, whole in zip(logits, input_prompts['input_ids'], whole_prompts['input_ids']):
            # Remove padding
            padding = torch.count_nonzero(whole == tokenizer.pad_token_id)
            whole = whole[padding:]
            padding = torch.count_nonzero(input == tokenizer.pad_token_id)
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
            all_losses.append(loss.item())

    for sentence_pair, loss in zip(data['sampled_answers_questions'], all_losses):
        sentence_pair['quality_scores'] = {}
        sentence_pair['quality_scores']['qa_loss'] = loss

json.dump(dataset, open(f'Failed-EMNLP/tasa_dataset_preparation/tasa_{PARTITION}_answers_questions_quality_scores_sum_no_context.json', 'w'), indent=4)