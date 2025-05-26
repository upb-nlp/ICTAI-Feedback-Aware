import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, Trainer
import wandb
import random
from peft import LoraConfig, get_peft_model
from prepare_dataset_keywords import get_dataset_keywords_feedback_aware_sft_v4
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key='TOKEN_KEY')
wandb.init(
    project="Feedback-Aware-Keywords",
    config={
        'model': 'meta-llama/Meta-Llama-3-8B',
        'task': 'Keywords, Feedback Aware, SFT',
    }
)

model_name = "meta-llama/Meta-Llama-3-8B"

new_model_name = "llama3_8b_keywords_feedback_aware_sft_lw"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY')
tokenizer.pad_token_id = 128002

data_train, data_test = get_dataset_keywords_feedback_aware_sft_v4()

print(len(data_train))
data_train = [data for data in data_train if len(tokenizer(data['prompt'])['input_ids']) < 2500]
data_test = [data for data in data_test if len(tokenizer(data['prompt'])['input_ids']) < 2500]
print(len(data_train))


random.shuffle(data_train)
random.shuffle(data_test)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
test_dataset = Dataset.from_list(data_test)


def _extract_text_between_braces(text):
    words_indexes_between_braces = []
    current_indexes = []

    for i, t in enumerate(text):
        if t == '{':
            current_indexes = [i]
        elif t == '}':
            current_indexes.append(i)
            words_indexes_between_braces.append(current_indexes)

    return words_indexes_between_braces[-1:]

def tokenize_function(examples):
    inputs = tokenizer(examples, add_special_tokens=False, return_tensors="pt")
    inputs_offsets = inputs.encodings[0].offsets
    inputs['input_ids'] = inputs['input_ids'].squeeze()
    inputs['attention_mask'] = inputs['attention_mask'].squeeze()
    inputs['labels'] = copy.deepcopy(inputs['input_ids'])

    indexes_braces = _extract_text_between_braces(examples)

    for i in range(len(inputs['labels'])):
        is_between_braces = False
        for indexes_brace in indexes_braces:
            if inputs_offsets[i][0] > indexes_brace[0] and inputs_offsets[i][1] - 1 <= indexes_brace[1]:
                is_between_braces = True
                break
        if not is_between_braces:
            inputs['labels'][i] = -100

    return inputs

train_dataset_tokenized = train_dataset.map(lambda x: tokenize_function(x['prompt']))
test_dataset_tokenized = test_dataset.map(lambda x: tokenize_function(x['prompt']))

# Drop all columns except input_ids, attention_mask and labels
train_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dpo_trainer = Trainer(
    model,
    train_dataset=train_dataset_tokenized,
    eval_dataset=test_dataset_tokenized,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        dataloader_num_workers=16,
        remove_unused_columns=False,
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=0.1,
        optim="adamw_8bit",
        report_to="wandb",
        output_dir="logs",
        save_strategy="no",
        save_steps=0.1,
        save_total_limit=1,
        fp16=True,
    ),
    data_collator=DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer, return_tensors="pt"),
)
dpo_trainer.train()
dpo_trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)