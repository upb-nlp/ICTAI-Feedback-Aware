import json
import os
import random
random.seed(42)

def get_dataset_baseline_all_keywords_sft():
    task_description_prompt = f"Select keywords for the following text."

    new_dataset_train = []
    # dataset_train = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_train_with_negative_keywords.json', 'r'))
    dataset_train = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/clean_train.json', 'r'))
    for data in dataset_train:
        text = data['abstract']
        prompt_generation = '\n'.join([keyword for keyword in data['keywords']])
        new_dataset_train.append({'prompt': f"{task_description_prompt}\nText:\n{text}\n### Response: \n{prompt_generation}"})

    new_dataset_val = []
    dataset_val = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_val_with_negative_keywords.json', 'r'))
    for data in dataset_val:
        text = data['abstract']
        prompt_generation = '\n'.join([keyword for keyword in data['keywords']])
        new_dataset_val.append({'prompt': f"{task_description_prompt}\nText:\n{text}\n### Response: \n{prompt_generation}"})

    new_dataset_test = []
    dataset_test = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_test.json', 'r'))
    for data in dataset_test:
        text = data['abstract']
        new_dataset_test.append({'prompt': f"{task_description_prompt}\nText:\n{text}\n### Response:"})

    return new_dataset_train, new_dataset_val, new_dataset_test

def get_dataset_baseline_single_keyword_orpo():
    task_description_prompt = f"Select keywords for the following text."

    new_dataset_train = []
    dataset_train = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_train_with_negative_keywords.json', 'r'))
    for data in dataset_train:
        text = data['abstract']
        good_keywords = [keyword for keyword in data['keywords']]
        bad_keywords = [keyword for keyword in data['negative_keywords']]

        for gk, bk in zip(good_keywords, bad_keywords):
            good_keyword = f" {gk}"
            bad_keyword = f" {bk}"
            new_dataset_train.append(
                {
                    'prompt': f"{task_description_prompt}\nText:\n{text}\n### Response:", 
                    'chosen': good_keyword, 
                    'rejected': bad_keyword,
                }
            )

    new_dataset_val = []
    dataset_val = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_val_with_negative_keywords.json', 'r'))
    for data in dataset_val:
        text = data['abstract']
        good_keywords = [keyword for keyword in data['keywords']]
        bad_keywords = [keyword for keyword in data['negative_keywords']]

        for gk, bk in zip(good_keywords, bad_keywords):
            good_keyword = f" {gk}"
            bad_keyword = f" {bk}"
            new_dataset_val.append(
                {
                    'prompt': f"{task_description_prompt}\nText:\n{text}\n### Response:", 
                    'chosen': good_keyword, 
                    'rejected': bad_keyword,
                }
            )

    return new_dataset_train, new_dataset_val

def get_dataset_baseline_single_keyword_sft():
    task_description_prompt = f"Select keywords for the following text."

    new_dataset_train = []
    dataset_train = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_train_with_negative_keywords.json', 'r'))
    for data in dataset_train:
        text = data['abstract']
        good_keywords = [keyword for keyword in data['keywords']]

        for gk in good_keywords:
            good_keyword = gk
            new_dataset_train.append(
                {
                    'prompt': f"{task_description_prompt}\nText:\n{text}\n### Response: {good_keyword}", 
                }
            )

    new_dataset_val = []
    dataset_val = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_val_with_negative_keywords.json', 'r'))
    for data in dataset_val:
        text = data['abstract']
        good_keywords = [keyword for keyword in data['keywords']]

        for gk in good_keywords:
            good_keyword = gk
            new_dataset_val.append(
                {
                    'prompt': f"{task_description_prompt}\nText:\n{text}\n### Response: {good_keyword}", 
                }
            )

    return new_dataset_train, new_dataset_val

def get_dataset_keywords_feedback_aware_orpo_v4():
    task_description_prompt = f"Iteratively select keywords for the following text."

    dataset_train = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_train_with_negative_keywords_feedback_aware_v4.json', 'r'))
    for data in dataset_train:
        prompt_sentences = data['abstract']
        previous_prompt_keywords = '\n'.join([f"{fp['type']} {{ {fp['keyword']} }}" for fp in data['previous_prompt_keywords']])
        chosen_keyword = f"\nGOOD {{ {data['chosen_keyword']['keyword']} }}"
        rejected_keyword = f"\nGOOD {{ {data['rejected_keyword']['keyword']} }}"
        
        prompt_words = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response:\n{previous_prompt_keywords}"
        chosen_words = chosen_keyword
        rejected_words = rejected_keyword

        data['prompt'] = prompt_words
        data['chosen'] = chosen_words
        data['rejected'] = rejected_words

    dataset_val = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_val_with_negative_keywords_feedback_aware_v4.json', 'r'))
    for data in dataset_val:
        prompt_sentences = data['abstract']
        previous_prompt_keywords = '\n'.join([f"{fp['type']} {{ {fp['keyword']} }}" for fp in data['previous_prompt_keywords']])
        chosen_keyword = f"\nGOOD {{ {data['chosen_keyword']['keyword']} }}"
        rejected_keyword = f"\nGOOD {{ {data['rejected_keyword']['keyword']} }}"

        prompt_words = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response:\n{previous_prompt_keywords}"
        chosen_words = chosen_keyword
        rejected_words = rejected_keyword

        data['prompt'] = prompt_words
        data['chosen'] = chosen_words
        data['rejected'] = rejected_words

    return dataset_train, dataset_val

def get_dataset_keywords_feedback_aware_sft_v4():
    task_description_prompt = f"Iteratively select keywords for the following text."

    dataset_train = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_train_with_negative_keywords_feedback_aware_v4.json', 'r'))
    for data in dataset_train:
        prompt_sentences = data['abstract']
        previous_prompt_keywords = '\n'.join([f"{fp['type']} {{ {fp['keyword']} }}" for fp in data['previous_prompt_keywords']])
        chosen_keyword = f"\nGOOD {{ {data['chosen_keyword']['keyword']} }}"
        
        prompt_words = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response:\n{previous_prompt_keywords}{chosen_keyword}"

        data['prompt'] = prompt_words
        

    dataset_val = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_val_with_negative_keywords_feedback_aware_v4.json', 'r'))
    for data in dataset_val:
        prompt_sentences = data['abstract']
        previous_prompt_keywords = '\n'.join([f"{fp['type']} {{ {fp['keyword']} }}" for fp in data['previous_prompt_keywords']])
        chosen_keyword = f"\nGOOD {{ {data['chosen_keyword']['keyword']} }}"

        prompt_words = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response:\n{previous_prompt_keywords}{chosen_keyword}"

        data['prompt'] = prompt_words

    return dataset_train, dataset_val