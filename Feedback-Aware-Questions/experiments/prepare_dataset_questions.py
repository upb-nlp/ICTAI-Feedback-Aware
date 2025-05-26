import json
import os
import random
random.seed(42)

def get_dataset_questions_baseline_single_answer_sft():
    task_description_prompt = f"Select a span from the following text that would serve as good answer for generating a question."

    new_dataset_train = []
    dataset_train = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_train_answers_questions_labeled_v2.json', 'r')) + json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_train_answers_questions_labeled_v2.json', 'r'))
    for data in dataset_train:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        for sent_answer in data['sampled_answers_questions']:
            if sent_answer['type'] == 'GOOD':
                prompt_generation = f"{sent_answer['answer']}"
                new_dataset_train.append({
                    'prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response: {prompt_generation}",
                    'generation_prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response:",
                    'answer': sent_answer['answer'],
                })
    
    new_dataset_val = []
    dataset_val = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_val_answers_questions_labeled_v2.json', 'r')) + json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_val_answers_questions_labeled_v2.json', 'r'))
    for data in dataset_val:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        for sent_answer in data['sampled_answers_questions']:
            if sent_answer['type'] == 'GOOD':
                prompt_generation = f"{sent_answer['answer']}"
                new_dataset_val.append({
                    'prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response: {prompt_generation}",
                    'generation_prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response:",
                    'answer': sent_answer['answer'],
                })

    return new_dataset_train, new_dataset_val

def get_dataset_questions_baseline_single_answer_orpo():
    new_dataset_train = []
    dataset_train = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_train_answers_questions_labeled_v2.json', 'r')) + json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_train_answers_questions_labeled_v2.json', 'r'))
    task_description_prompt = f"Select a span from the following text that would serve as good answer for generating a question."
    for data in dataset_train:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        aux_good_sent_pairs = [sent_pair for sent_pair in data['sampled_answers_questions'] if sent_pair['type'] == 'GOOD']
        aux_bad_sent_pairs = [sent_pair for sent_pair in data['sampled_answers_questions'] if sent_pair['type'] != 'GOOD']

        if len(aux_bad_sent_pairs) >= len(aux_good_sent_pairs):
            sampled_bad_sent_pairs = random.sample(aux_bad_sent_pairs, len(aux_good_sent_pairs))
        elif len(aux_bad_sent_pairs) > 0:
            percent = int(len(aux_good_sent_pairs) / len(aux_bad_sent_pairs)) + 1
            sampled_bad_sent_pairs = random.sample(aux_bad_sent_pairs * percent, len(aux_good_sent_pairs))
        else:
            continue

        for sent_pair_good, sent_pair_bad in zip(aux_good_sent_pairs, sampled_bad_sent_pairs):
            good_prompt_generation = f" {sent_pair_good['answer']}"
            bad_prompt_generation = f" {sent_pair_bad['answer']}"
            new_dataset_train.append(
                {
                    'prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response:", 
                    'chosen': good_prompt_generation, 
                    'rejected': bad_prompt_generation,
                    'context': prompt_sentences,
                }
            )

    new_dataset_val = []
    dataset_val = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_val_answers_questions_labeled_v2.json', 'r')) + json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_val_answers_questions_labeled_v2.json', 'r'))
    for data in dataset_val:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        aux_good_sent_pairs = [sent_pair for sent_pair in data['sampled_answers_questions'] if sent_pair['type'] == 'GOOD']
        aux_bad_sent_pairs = [sent_pair for sent_pair in data['sampled_answers_questions'] if sent_pair['type'] != 'GOOD']

        if len(aux_bad_sent_pairs) >= len(aux_good_sent_pairs):
            sampled_bad_sent_pairs = random.sample(aux_bad_sent_pairs, len(aux_good_sent_pairs))
        elif len(aux_bad_sent_pairs) > 0:
            percent = int(len(aux_good_sent_pairs) / len(aux_bad_sent_pairs)) + 1
            sampled_bad_sent_pairs = random.sample(aux_bad_sent_pairs * percent, len(aux_good_sent_pairs))
        else:
            continue

        for sent_pair_good, sent_pair_bad in zip(aux_good_sent_pairs, sampled_bad_sent_pairs):
            good_prompt_generation = f" {sent_pair_good['answer']}"
            bad_prompt_generation = f" {sent_pair_bad['answer']}"
            new_dataset_val.append(
                {
                    'prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response:", 
                    'chosen': good_prompt_generation, 
                    'rejected': bad_prompt_generation,
                    'context': prompt_sentences,
                }
            )

    return new_dataset_train, new_dataset_val


def get_dataset_questions_feedback_aware_orpo_v4():
    task_description_prompt = f"Iteratively select a span from the following text that would serve as good answer for generating a question."

    dataset_train = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_fairytaleqa_train_answers_questions_feedback_aware_v4.json', 'r'))
    for data in dataset_train:
        prompt_sentences = data['text']
        previous_prompt_answers = '\n'.join([f"{fp['type']} {{ {fp['answer']} }} {fp['generated_question']}" for fp in data['previous_prompt_answers']])
        chosen_answer = f"\nGOOD {{ {data['chosen_answer']['answer']} }} {data['chosen_answer']['generated_question']}"
        rejected_answer = f"\nGOOD {{ {data['rejected_answer']['answer']} }} {data['rejected_answer']['generated_question']}"
        
        prompt_words = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response:\n{previous_prompt_answers}"
        chosen_words = chosen_answer
        rejected_words = rejected_answer

        data['prompt'] = prompt_words
        data['chosen'] = chosen_words
        data['rejected'] = rejected_words

    dataset_val = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_fairytaleqa_val_answers_questions_feedback_aware_v4.json', 'r'))
    for data in dataset_val:
        prompt_sentences = data['text']
        previous_prompt_answers = '\n'.join([f"{fp['type']} {{ {fp['answer']} }} {fp['generated_question']}" for fp in data['previous_prompt_answers']])
        chosen_answer = f"\nGOOD {{ {data['chosen_answer']['answer']} }} {data['chosen_answer']['generated_question']}"
        rejected_answer = f"\nGOOD {{ {data['rejected_answer']['answer']} }} {data['rejected_answer']['generated_question']}"
        
        prompt_words = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response:\n{previous_prompt_answers}"
        chosen_words = chosen_answer
        rejected_words = rejected_answer

        data['prompt'] = prompt_words
        data['chosen'] = chosen_words
        data['rejected'] = rejected_words
    return dataset_train, dataset_val

def get_dataset_questions_feedback_aware_sft_v4():
    task_description_prompt = f"Iteratively select a span from the following text that would serve as good answer for generating a question."

    dataset_train = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_fairytaleqa_train_answers_questions_feedback_aware_v4.json', 'r'))
    for data in dataset_train:
        prompt_sentences = data['text']
        previous_prompt_answers = '\n'.join([f"{fp['type']} {{ {fp['answer']} }} {fp['generated_question']}" for fp in data['previous_prompt_answers']])
        chosen_answer = f"\nGOOD {{ {data['chosen_answer']['answer']} }} {data['chosen_answer']['generated_question']}"
        
        prompt_words = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response:\n{previous_prompt_answers}{chosen_answer}"

        data['prompt'] = prompt_words

    dataset_val = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_fairytaleqa_val_answers_questions_feedback_aware_v4.json', 'r'))
    for data in dataset_val:
        prompt_sentences = data['text']
        previous_prompt_answers = '\n'.join([f"{fp['type']} {{ {fp['answer']} }} {fp['generated_question']}" for fp in data['previous_prompt_answers']])
        chosen_answer = f"\nGOOD {{ {data['chosen_answer']['answer']} }} {data['chosen_answer']['generated_question']}"
        
        prompt_words = f"{task_description_prompt}\n\n### Text:\n{prompt_sentences}\n\n### Response:\n{previous_prompt_answers}{chosen_answer}"

        data['prompt'] = prompt_words
    return dataset_train, dataset_val

def get_dataset_questions_all_answers_sft():
    task_description_prompt = f"Select a span from the following text that would serve as good answers for generating a question."

    new_dataset_train = []
    dataset_train = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_train_answers_questions_labeled_v2.json', 'r')) + json.load(open('Failed-EMNLP/questions_dataset_preparation/fairytaleqa_train_answers_questions_labeled_v2.json', 'r'))
    for data in dataset_train:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        prompt_generation_list = []
        for sent_pair in data['sampled_answers_questions']:
            if sent_pair['type'] == 'GOOD':
                prompt_generation_list.append(f"{sent_pair['answer']}")
        prompt_generation = '\n'.join(prompt_generation_list)
        new_dataset_train.append({'prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response: \n{prompt_generation}"})

    new_dataset_val = []
    dataset_val = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_val_answers_questions_labeled_v2.json', 'r')) + json.load(open('Failed-EMNLP/questions_dataset_preparation/fairytaleqa_val_answers_questions_labeled_v2.json', 'r'))
    for data in dataset_val:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        prompt_generation_list = []
        for sent_pair in data['sampled_answers_questions']:
            if sent_pair['type'] == 'GOOD':
                prompt_generation_list.append(f"{sent_pair['answer']}")
        prompt_generation = '\n'.join(prompt_generation_list)
        new_dataset_val.append({'prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response: \n{prompt_generation}"})
    
    new_dataset_test = []
    dataset_test = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_test.json', 'r')) + json.load(open('Failed-EMNLP/questions_dataset_preparation/fairytaleqa_test.json', 'r'))
    for data in dataset_test:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        new_dataset_test.append({
            'generation_prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response:",
            'text': prompt_sentences,
        })

    return new_dataset_train, new_dataset_val, new_dataset_test

def get_dataset_questions_all_answers_with_aid_sft():
    task_description_prompt = f"Select a span from the following text that would serve as good answers for generating a question."

    new_dataset_train = []
    dataset_train = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_train_answers_questions_labeled_v2.json', 'r')) + json.load(open('Failed-EMNLP/questions_dataset_preparation/fairytaleqa_train_answers_questions_labeled_v2.json', 'r'))
    for data in dataset_train:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        prompt_generation_list = []
        for sent_pair in data['sampled_answers_questions']:
            if sent_pair['type'] == 'GOOD':
                prompt_generation_list.append(f"{{ {sent_pair['answer']} }} {sent_pair['generated_question']}")
        prompt_generation = '\n'.join(prompt_generation_list)
        new_dataset_train.append({'prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response: \n{prompt_generation}"})

    new_dataset_val = []
    dataset_val = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_val_answers_questions_labeled_v2.json', 'r')) + json.load(open('Failed-EMNLP/questions_dataset_preparation/fairytaleqa_val_answers_questions_labeled_v2.json', 'r'))
    for data in dataset_val:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        prompt_generation_list = []
        for sent_pair in data['sampled_answers_questions']:
            if sent_pair['type'] == 'GOOD':
                prompt_generation_list.append(f"{{ {sent_pair['answer']} }} {sent_pair['generated_question']}")
        prompt_generation = '\n'.join(prompt_generation_list)
        new_dataset_val.append({'prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response: \n{prompt_generation}"})
    
    new_dataset_test = []
    dataset_test = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_test.json', 'r')) + json.load(open('Failed-EMNLP/questions_dataset_preparation/fairytaleqa_test.json', 'r'))
    for data in dataset_test:
        prompt_sentences = data['text'] if 'text' in data else data['story_section']
        new_dataset_test.append({
            'generation_prompt': f"{task_description_prompt}\nText:\n{prompt_sentences}\n### Response:",
            'text': prompt_sentences,
        })

    return new_dataset_train, new_dataset_val, new_dataset_test
