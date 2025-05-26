import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import ORPOConfig
from feedback_aware_orpo_trainer_v3 import FeedbackAwareORPOTrainerv3
from peft import LoraConfig
import wandb
import random
from prepare_dataset_questions import get_dataset_questions_feedback_aware_orpo_v4

wandb.login(key='TOKEN_KEY')
wandb.init(
    project="Feedback-Aware-Questions",
    config={
        'model': 'meta-llama/Meta-Llama-3-8B',
        'task': 'Questions, Feedback Aware V4, ORPO, Beta 0.3',
    }
)

data_train, data_test = get_dataset_questions_feedback_aware_orpo_v4()

model_name = "meta-llama/Meta-Llama-3-8B"
new_model_name = "llama3_8b_questions_feedback_aware_orpo_v4_lw"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token='TOKEN_KEY')
tokenizer.pad_token_id = 128002

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token='TOKEN_KEY')

print(len(data_train))
# Only keep the samples that the data['prompt'], data['chosen'] and data['rejected'] have less than 2000 tokens each
data_train = [data for data in data_train if len(tokenizer(data['prompt'])['input_ids']) < 1000 and len(tokenizer(data['chosen'])['input_ids']) < 1500 and len(tokenizer(data['rejected'])['input_ids']) < 1500]
data_test = [data for data in data_test if len(tokenizer(data['prompt'])['input_ids']) < 1000 and len(tokenizer(data['chosen'])['input_ids']) < 1500 and len(tokenizer(data['rejected'])['input_ids']) < 1500]
print(len(data_train))
############################################### ORPO ###############################################

random.shuffle(data_train)
random.shuffle(data_test)

# Convert to dataset
train_dataset = Dataset.from_list(data_train)
test_dataset = Dataset.from_list(data_test)


dpo_trainer = FeedbackAwareORPOTrainerv3(
    model,
    args=ORPOConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        dataloader_num_workers=16,
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        optim="adamw_8bit",
        report_to="wandb",
        output_dir="orpo_logs_lora_half_beta_03",
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=1,
        max_length=3000,
        max_prompt_length=1000,
        beta=0.3,
        fp16=True,
    ),
    peft_config=LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    ),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
dpo_trainer.save_model(new_model_name)
tokenizer.save_pretrained(new_model_name)