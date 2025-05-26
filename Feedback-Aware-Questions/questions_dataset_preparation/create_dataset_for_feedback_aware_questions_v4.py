import json
import random
import copy

random.seed(42)
PARTITION = 'val' # 'train' 'test' 'val'

dataset = json.load(open(f'Feedback-Aware-Questions/questions_dataset_preparation/tasa_{PARTITION}_answers_questions_labeled_v2.json'))

new_dataset = []

# For each text
for data in dataset:
    for iteration in range(3):
        good_answers = [answer for answer in data['sampled_answers_questions'] if answer['type'] == 'GOOD']
        bad_answers = [answer for answer in data['sampled_answers_questions'] if answer['type'] != 'GOOD']

        previous_prompt_answers = []

        for i in range(20):
            # Select if possible a random good and bad answer and remove them from the list
            if len(good_answers) > 0:
                sampled_good_answer = random.choice(good_answers)
                good_answers.remove(sampled_good_answer)
                
                if len(bad_answers) > 0 and len(previous_prompt_answers) > 0:
                    x = random.random()
                    if x < 0.8:
                        sampled_bad_answer = random.choice(bad_answers)
                        bad_answers.remove(sampled_bad_answer)
                    else:
                        sampled_bad_answer = copy.deepcopy(random.choice(previous_prompt_answers))
                        sampled_bad_answer['type'] = 'REPEATED'
                elif len(bad_answers) > 0:
                    sampled_bad_answer = random.choice(bad_answers)
                    bad_answers.remove(sampled_bad_answer)
                elif len(previous_prompt_answers) > 0:
                    sampled_bad_answer = copy.deepcopy(random.choice(previous_prompt_answers))
                    sampled_bad_answer['type'] = 'REPEATED'
            else:
                break

            # Create a new dataset entry
            new_data = {
                'text': data['text'],
                'dataset': 'TASA',
                'previous_prompt_answers': copy.deepcopy(previous_prompt_answers),
                'chosen_answer': sampled_good_answer,
                'rejected_answer': sampled_bad_answer,
            }
            new_dataset.append(new_data)
            
            if sampled_bad_answer['type'] != 'REPEATED':
                previous_prompt_answers.append(random.choice([sampled_good_answer, sampled_bad_answer]))
            elif sampled_bad_answer['type'] == 'REPEATED':
                sampled_bad_answer['type'] = 'BAD'
                previous_prompt_answers.append(sampled_good_answer)


dataset = json.load(open(f'Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_{PARTITION}_answers_questions_labeled_v2.json'))
# For each text

for data in dataset:
    for iteration in range(3):
        good_answers = [answer for answer in data['sampled_answers_questions'] if answer['type'] == 'GOOD']
        bad_answers = [answer for answer in data['sampled_answers_questions'] if answer['type'] != 'GOOD']

        previous_prompt_answers = []

        for i in range(20):
            # Select if possible a random good and bad answer and remove them from the list
            if len(good_answers) > 0:
                sampled_good_answer = random.choice(good_answers)
                good_answers.remove(sampled_good_answer)
                
                if len(bad_answers) > 0 and len(previous_prompt_answers) > 0:
                    x = random.random()
                    if x < 0.8:
                        sampled_bad_answer = random.choice(bad_answers)
                        bad_answers.remove(sampled_bad_answer)
                    else:
                        sampled_bad_answer = copy.deepcopy(random.choice(previous_prompt_answers))
                        sampled_bad_answer['type'] = 'REPEATED'
                elif len(bad_answers) > 0:
                    sampled_bad_answer = random.choice(bad_answers)
                    bad_answers.remove(sampled_bad_answer)
                elif len(previous_prompt_answers) > 0:
                    sampled_bad_answer = copy.deepcopy(random.choice(previous_prompt_answers))
                    sampled_bad_answer['type'] = 'REPEATED'
            else:
                break

            # Create a new dataset entry
            new_data = {
                'text': data['story_section'],
                'dataset': 'FairytaleQA',
                'previous_prompt_answers': copy.deepcopy(previous_prompt_answers),
                'chosen_answer': sampled_good_answer,
                'rejected_answer': sampled_bad_answer,
            }
            new_dataset.append(new_data)
            
            if sampled_bad_answer['type'] != 'REPEATED':
                previous_prompt_answers.append(random.choice([sampled_good_answer, sampled_bad_answer]))
            elif sampled_bad_answer['type'] == 'REPEATED':
                sampled_bad_answer['type'] = 'BAD'
                previous_prompt_answers.append(sampled_good_answer)

print(len(new_dataset))

for data in new_dataset:
    for answer in data['previous_prompt_answers']:
        if answer['type'] == 'REPEATED':
            answer['type'] = 'BAD'
    
json.dump(new_dataset, open(f'Feedback-Aware-Questions/questions_dataset_preparation/tasa_fairytaleqa_{PARTITION}_answers_questions_feedback_aware_v4.json', 'w'), indent=4)