import json
import os

def get_dataset_entailment_generation():
    base_dir = os.path.join('preprocessed_datasets', 'entailment_generation')
    files_train = [os.path.join(base_dir, filename) for filename in os.listdir(base_dir) if 'train' in filename]
    files_test = [os.path.join(base_dir, filename) for filename in os.listdir(base_dir) if 'test' in filename]

    dataset_train = []
    dataset_test = []

    for filename in files_train:
        dataset_train += json.load(open(filename, 'r'))
    for data in dataset_train:
        data['prompt'] = f"Generate an entailment based on the following sentences: {data['text']}\n\n### Response: {data['entailment_phrase']}"
        data['generation_prompt'] = f"Generate an entailment based on the following sentences: {data['text']}\n\n### Response: "
    
    for filename in files_test:
        dataset_test += json.load(open(filename, 'r'))
    for data in dataset_test:
        data['prompt'] = f"Generate an entailment based on the following sentences: {data['text']}\n\n### Response: {data['entailment_phrase']}"
        data['generation_prompt'] = f"Generate an entailment based on the following sentences: {data['text']}\n\n### Response: "
    return dataset_train, dataset_test

def get_dict_for_questions(data, task):
    new_data = {
        'context': data['context'],
        'question': data['question'],
        'answer': data['answer'],
    }
    if task == 'qa':
        new_data['prompt'] = f"Answer the following question based on the context.\nContext: {data['context']}\nQuestion: {data['question']}\n### Response: {data['answer']}"
        new_data['generation_prompt'] = f"Answer the following question based on the context.\nContext: {data['context']}\nQuestion: {data['question']}\n### Response:"
        new_data['task'] = 'qa'
        new_data['label'] = f"{data['answer']}"
    elif task == 'qg':
        new_data['prompt'] = f"Generate a question based on the context and the answer.\nContext: {data['context']}\nAnswer: {data['answer']}\n### Response: {data['question']}"
        new_data['generation_prompt'] = f"Generate a question based on the context and the answer.\nContext: {data['context']}\nAnswer: {data['answer']}\n### Response:"
        new_data['task'] = 'qg'
        new_data['label'] = f"{data['question']}"
    elif task == 'aq':
        new_data['prompt'] = f"Select an answer from the context that can be used to generate a question.\nContext: {data['context']}\n### Response: {data['answer']}"
        new_data['generation_prompt'] = f"Select an answer from the context that can be used to generate a question.\nContext: {data['context']}\n### Response:"
        new_data['task'] = 'aq'
        new_data['label'] = f"{data['answer']}"
    else:
        raise ValueError(f"Invalid task {task}")
    return new_data

def get_dataset_questions():
    files_train = [
        'reshaped_narrativeqa_train.json',
        'reshaped_hotpotqa_train.json',
        'reshaped_squad_train.json',
    ]

    data_train_raw = []
    for file in files_train:
        data_train_raw += json.load(open(file, 'r'))
    data_train = []
    for data in data_train_raw:
        data_train.append(get_dict_for_questions(data, 'qa'))
    for data in data_train_raw:
        data_train.append(get_dict_for_questions(data, 'qg'))
    for data in data_train_raw:
        data_train.append(get_dict_for_questions(data, 'aq'))

    files_test = [
        'reshaped_narrativeqa_test.json',
        'reshaped_hotpotqa_test.json',
        'reshaped_squad_test.json',
    ]
    data_test_raw = []
    for file in files_test:
        data_test_raw += json.load(open(file, 'r'))
    data_test = []
    for data in data_test_raw:
        data_test.append(get_dict_for_questions(data, 'qa'))
    for data in data_test_raw:
        data_test.append(get_dict_for_questions(data, 'qg'))
    for data in data_test_raw:
        data_test.append(get_dict_for_questions(data, 'aq'))

    return data_train, data_test
