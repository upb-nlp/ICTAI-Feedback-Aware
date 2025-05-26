import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import wandb
import random
from peft import LoraConfig, get_peft_model
from trl import DataCollatorForCompletionOnlyLM
from prepare_dataset_qall import get_dataset_for_olmo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key='TOKEN_KEY')
wandb.init(
    project="Feedback-Aware",
    config={
        'model': 'Llama-3',
        'task': 'llama3-8b_qans',
    }
)

data_train, data_test = get_dataset_for_olmo()
data_train = [data for data in data_train if data['task'] == 'qa']
data_test = [data for data in data_test if data['task'] == 'qa']
random.shuffle(data_train)
random.shuffle(data_test)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
test_dataset = Dataset.from_list(data_test)


model_name = "meta-llama/Meta-Llama-3-8B"
new_model_name = "llama3-8b_qans_lw"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY')
tokenizer.pad_token_id = 128002

def tokenize_function(examples):
    inputs = tokenizer(examples["prompt"], return_tensors="pt", max_length=2048, truncation=True)
    inputs["no_tokens"] = inputs["input_ids"].shape[1] + 1
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

lora_config = LoraConfig(
    r=64,  # the rank of the LoRA matrices
    lora_alpha=16, # the weight
    lora_dropout=0.1, # dropout to add to the LoRA layers
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #, "o_proj"], # the name of the layers to add LoRA
    modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainer = Trainer(
    model=model,
    train_dataset=train_dataset_tokenized,
    eval_dataset=test_dataset_tokenized,
    tokenizer=tokenizer,
    args=TrainingArguments(
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=True,
        dataloader_num_workers=16,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        optim="adamw_8bit",
        report_to="wandb",
        output_dir="logs_qwen",
        save_steps=0.2,
        save_total_limit=1,
    ),
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False, return_tensors="pt", response_template="### Response:")
)

trainer.train()

# Save trained model
trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)