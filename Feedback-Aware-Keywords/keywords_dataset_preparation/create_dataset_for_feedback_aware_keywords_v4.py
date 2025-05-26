import json
import random
import copy

random.seed(42)
PARTITION = 'val' # 'train' 'test' 'val'

dataset = json.load(open(f'Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_{PARTITION}_with_negative_keywords.json'))

new_dataset = []

# For each text
for data in dataset:
    for iteration in range(3):
        good_keywords = [{'keyword': k, 'type': 'GOOD'} for k in data['keywords']]
        bad_keywords = [{'keyword': k, 'type': 'BAD'} for k in data['negative_keywords']]

        previous_prompt_keywords = []

        for i in range(min(len(good_keywords), 20)):
            # Select if possible a random good and bad keyword and remove them from the list
            if len(good_keywords) > 0:
                sampled_good_keyword = random.choice(good_keywords)
                good_keywords.remove(sampled_good_keyword)
                
                if len(bad_keywords) > 0 and len(previous_prompt_keywords) > 0:
                    x = random.random()
                    if x < 0.8:
                        sampled_bad_keyword = random.choice(bad_keywords)
                        bad_keywords.remove(sampled_bad_keyword)
                    else:
                        sampled_bad_keyword = copy.deepcopy(random.choice(previous_prompt_keywords))
                        sampled_bad_keyword['type'] = 'REPEATED'
                elif len(bad_keywords) > 0:
                    sampled_bad_keyword = random.choice(bad_keywords)
                    bad_keywords.remove(sampled_bad_keyword)
                elif len(previous_prompt_keywords) > 0:
                    sampled_bad_keyword = copy.deepcopy(random.choice(previous_prompt_keywords))
                    sampled_bad_keyword['type'] = 'REPEATED'
            else:
                break

            # Create a new dataset entry
            new_data = {
                'abstract': data['abstract'],
                'previous_prompt_keywords': copy.deepcopy(previous_prompt_keywords),
                'chosen_keyword': sampled_good_keyword,
                'rejected_keyword': sampled_bad_keyword,
            }
            new_dataset.append(new_data)
            
            if sampled_bad_keyword['type'] != 'REPEATED':
                previous_prompt_keywords.append(random.choice([sampled_good_keyword, sampled_bad_keyword]))
            elif sampled_bad_keyword['type'] == 'REPEATED':
                sampled_bad_keyword['type'] = 'BAD'
                previous_prompt_keywords.append(sampled_good_keyword)


print(len(new_dataset))

for data in new_dataset:
    for keyword in data['previous_prompt_keywords']:
        if keyword['type'] == 'REPEATED':
            keyword['type'] = 'BAD'
    
json.dump(new_dataset, open(f'Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_{PARTITION}_with_negative_keywords_feedback_aware_v4.json', 'w'), indent=4)