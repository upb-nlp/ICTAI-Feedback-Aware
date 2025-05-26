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
        # Create a new dataset entry
        new_data = {
            'text': data['abstract'],
            'keywords': data['keywords'],
            'negative_keywords': data['negative_keywords'],
        }

        good_keywords = [keyword for keyword in data['keywords']]
        bad_keywords = [keyword for keyword in data['negative_keywords']]

        # Choose the number of good, bad and repeated keywords to sample
        total_num_keywords = min(25, len(data['keywords']) + len(data['negative_keywords']))

        if total_num_keywords < 2:
            continue

        num_repeated = random.choice(range(1, min(5, total_num_keywords)))
        max_num_good = min(len(good_keywords), total_num_keywords - num_repeated)

        if max_num_good == 0:
            continue

        num_good = random.choice(range(1, max_num_good + 1))
        num_bad = min(len(bad_keywords), total_num_keywords - num_repeated - num_good)

        # Sample the keywords
        sampled_good_keywords = random.sample(good_keywords, num_good)
        sampled_bad_keywords = random.sample(bad_keywords, num_bad)

        if len(sampled_good_keywords + sampled_bad_keywords) < num_repeated:
            continue
        sampled_repeated_keywords = random.sample(sampled_good_keywords + sampled_bad_keywords, num_repeated)

        new_data['good_keywords'] = [{'keyword': k} for k in sampled_good_keywords]
        new_data['bad_keywords'] = [{'keyword': k} for k in sampled_bad_keywords]
        new_data['repeated_keywords'] = [{'keyword': k} for k in sampled_repeated_keywords]

        # Deep copy the repeated keywords
        sampled_repeated_keywords = copy.deepcopy(sampled_repeated_keywords)

        # Create a list with all the good and bad sampled keywords, but add a key in each dict mentioning if it is good or bad
        for pair in new_data['good_keywords']:
            pair['type_training'] = 'GOOD'
        for pair in new_data['bad_keywords']:
            pair['type_training'] = 'BAD'
        for pair in new_data['repeated_keywords']:
            pair['type_training'] = 'REPEATED'

        all_keywords = new_data['good_keywords'] + new_data['bad_keywords']
        random.shuffle(all_keywords)

        # Find the positions of the keywords in the all_keywords list whose 'indexes' are the same as the 'indexes' of the keywords in the repeated sampled keywords
        for repeated_pair in new_data['repeated_keywords']:
            for i, pair in enumerate(all_keywords):
                if pair['keyword'] == repeated_pair['keyword']:
                    copy_index = i
                    break

            # Insert the repeated pair in a random position in the all_keywords list, after the copy_index
            pos = random.choice(range(copy_index + 1, len(all_keywords) + 1))
            all_keywords.insert(pos, repeated_pair)

        new_data['final_keywords'] = all_keywords

        new_dataset.append(new_data)

print(len(new_dataset))
json.dump(new_dataset, open(f'Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_{PARTITION}_with_negative_keywords_feedback_aware_v2.json', 'w'), indent=4)