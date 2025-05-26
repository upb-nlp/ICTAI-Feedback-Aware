import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import wandb
import random
from peft import LoraConfig
from prepare_dataset_keywords import get_dataset_baseline_single_keyword_sft

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key='TOKEN_KEY')
wandb.init(
    project="Feedback-Aware-Keywords",
    config={
        'model': 'meta-llama/Meta-Llama-3-8B',
        'task': 'V2, Keywords, Single Keyword, SFT, Lora, Half, Base',
    }
)

data_train, data_test = get_dataset_baseline_single_keyword_sft()

random.shuffle(data_train)
random.shuffle(data_test)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
test_dataset = Dataset.from_list(data_test)

model_name = "meta-llama/Meta-Llama-3-8B"

new_model_name = "llama3_8b_keywords_baseline_single_keyword_generation_sft_lw"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY')
tokenizer.pad_token_id = 128002

def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], return_tensors="pt", max_length=2048*4, truncation=True)
    inputs["no_tokens"] = inputs["input_ids"].shape[1]
    inputs["attention_mask"] = inputs["attention_mask"].squeeze()
    inputs["input_ids"] = inputs["input_ids"].squeeze()

    # Add tokenizer.bos_token_id to the beginning of the input_ids tokenizer.eos_token_id to the end of the input_ids. Update the attention mask
    inputs["input_ids"] = torch.cat((inputs["input_ids"], torch.tensor([tokenizer.eos_token_id])))
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.tensor([1])))

    return inputs

train_dataset_tokenized = train_dataset.map(lambda x: tokenize_function(x))
test_dataset_tokenized = test_dataset.map(lambda x: tokenize_function(x))

print(len(train_dataset_tokenized))
print(len(test_dataset_tokenized))

# Filter out examples that are too long
train_dataset_tokenized = train_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 2000)
test_dataset_tokenized = test_dataset_tokenized.filter(lambda x: x['no_tokens'] <= 2000)

print(len(train_dataset_tokenized))
print(len(test_dataset_tokenized))

# Drop all columns except input_ids, attention_mask and labels
train_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
test_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])


model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY')

dpo_trainer = SFTTrainer(
    model,
    train_dataset=train_dataset_tokenized,
    eval_dataset=test_dataset_tokenized,
    tokenizer=tokenizer,
    dataset_text_field='prompt',
    args=TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        dataloader_num_workers=16,
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
    max_seq_length=2000,
    peft_config=LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    ),
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False, return_tensors="pt", response_template="### Response:")
)
dpo_trainer.train()
dpo_trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)