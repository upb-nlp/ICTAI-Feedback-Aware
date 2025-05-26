import json
from collections import defaultdict

dataset = json.load(open("Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_test.json", "r"))
grouped_data = defaultdict(lambda: {'questions': [], 'answers': []})

for entry in dataset:
    section = entry['story_section']
    grouped_data[section]['questions'].append(entry['question'])
    grouped_data[section]['answers'].append(entry['answer1'])
grouped_data = dict(grouped_data)

json.dump(grouped_data, open("Feedback-Aware-Questions/questions_dataset_preparation/grouped_fairytaleqa_test.json", "w"), indent=4)