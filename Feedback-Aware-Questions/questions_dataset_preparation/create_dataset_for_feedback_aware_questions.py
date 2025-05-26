import json
import random
import copy

random.seed(42)
PARTITION = 'val' # 'train' 'test' 'val'

dataset = json.load(open(f'Failed-EMNLP/questions_dataset_preparation/tasa_{PARTITION}_answers_questions_labeled_v2.json'))

new_dataset = []

# For each text
for data in dataset:
    for iteration in range(3):
        # Create a new dataset entry
        new_data = {
            'text': data['text'],
            'sampled_answers_questions': data['sampled_answers_questions'],
            'dataset': 'TASA',
        }

        good_pairs = [pair for pair in data['sampled_answers_questions'] if pair['type'] == 'GOOD']
        bad_pairs = [pair for pair in data['sampled_answers_questions'] if pair['type'] != 'GOOD']

        # Choose the number of good, bad and repeated pairs to sample
        total_num_pairs = min(20, len(data['sampled_answers_questions']))
        num_repeated = random.choice(range(1, 5))
        max_num_good = min(len(good_pairs), total_num_pairs - num_repeated)

        if max_num_good == 0:
            continue

        num_good = random.choice(range(1, max_num_good + 1))
        num_bad = min(len(bad_pairs), total_num_pairs - num_repeated - num_good)

        # Sample the pairs
        sampled_good_pairs = random.sample(good_pairs, num_good)
        sampled_bad_pairs = random.sample(bad_pairs, num_bad)

        if len(sampled_good_pairs + sampled_bad_pairs) < num_repeated:
            continue
        sampled_repeated_pairs = random.sample(sampled_good_pairs + sampled_bad_pairs, num_repeated)

        new_data['good_pairs'] = sampled_good_pairs
        new_data['bad_pairs'] = sampled_bad_pairs
        new_data['repeated_pairs'] = sampled_repeated_pairs

        # Deep copy the repeated pairs
        sampled_repeated_pairs = copy.deepcopy(sampled_repeated_pairs)

        # Create a list with all the good and bad sampled pairs, but add a key in each dict mentioning if it is good or bad
        for pair in sampled_good_pairs:
            pair['type_training'] = pair['type']
        for pair in sampled_bad_pairs:
            pair['type_training'] = pair['type']
        for pair in sampled_repeated_pairs:
            pair['type_training'] = 'REPEATED'

        all_pairs = sampled_good_pairs + sampled_bad_pairs
        random.shuffle(all_pairs)

        # Find the positions of the pairs in the all_pairs list whose 'indexes' are the same as the 'indexes' of the pairs in the repeated sampled pairs
        for repeated_pair in sampled_repeated_pairs:
            for i, pair in enumerate(all_pairs):
                if pair['answer'] == repeated_pair['answer']:
                    copy_index = i
                    break

            # Insert the repeated pair in a random position in the all_pairs list, after the copy_index
            pos = random.choice(range(copy_index + 1, len(all_pairs) + 1))
            all_pairs.insert(pos, repeated_pair)

        new_data['final_answers_questions'] = all_pairs

        new_dataset.append(new_data)

dataset = json.load(open(f'Failed-EMNLP/questions_dataset_preparation/fairytaleqa_{PARTITION}_answers_questions_labeled_v2.json'))
# For each text
for data in dataset:
    for iteration in range(3):
        # Create a new dataset entry
        new_data = {
            'text': data['story_section'],
            'sampled_answers_questions': data['sampled_answers_questions'],
            'dataset': 'FairytaleQA',
        }

        good_pairs = [pair for pair in data['sampled_answers_questions'] if pair['type'] == 'GOOD']
        bad_pairs = [pair for pair in data['sampled_answers_questions'] if pair['type'] != 'GOOD']

        # Choose the number of good, bad and repeated pairs to sample
        total_num_pairs = min(20, len(data['sampled_answers_questions']))
        num_repeated = random.choice(range(1, 5))
        max_num_good = min(len(good_pairs), total_num_pairs - num_repeated)

        if max_num_good <= 0:
            continue

        num_good = random.choice(range(1, max_num_good + 1))
        num_bad = min(len(bad_pairs), total_num_pairs - num_repeated - num_good)

        # Sample the pairs
        sampled_good_pairs = random.sample(good_pairs, num_good)
        sampled_bad_pairs = random.sample(bad_pairs, num_bad)

        if len(sampled_good_pairs + sampled_bad_pairs) < num_repeated:
            continue
        sampled_repeated_pairs = random.sample(sampled_good_pairs + sampled_bad_pairs, num_repeated)

        new_data['good_pairs'] = sampled_good_pairs
        new_data['bad_pairs'] = sampled_bad_pairs
        new_data['repeated_pairs'] = sampled_repeated_pairs

        # Deep copy the repeated pairs
        sampled_repeated_pairs = copy.deepcopy(sampled_repeated_pairs)

        # Create a list with all the good and bad sampled pairs, but add a key in each dict mentioning if it is good or bad
        for pair in sampled_good_pairs:
            pair['type_training'] = pair['type']
        for pair in sampled_bad_pairs:
            pair['type_training'] = pair['type']
        for pair in sampled_repeated_pairs:
            pair['type_training'] = 'REPEATED'

        all_pairs = sampled_good_pairs + sampled_bad_pairs
        random.shuffle(all_pairs)

        # Find the positions of the pairs in the all_pairs list whose 'indexes' are the same as the 'indexes' of the pairs in the repeated sampled pairs
        for repeated_pair in sampled_repeated_pairs:
            for i, pair in enumerate(all_pairs):
                if pair['answer'] == repeated_pair['answer']:
                    copy_index = i
                    break

            # Insert the repeated pair in a random position in the all_pairs list, after the copy_index
            pos = random.choice(range(copy_index + 1, len(all_pairs) + 1))
            all_pairs.insert(pos, repeated_pair)

        new_data['final_answers_questions'] = all_pairs

        new_dataset.append(new_data)

print(len(new_dataset))
json.dump(new_dataset, open(f'Failed-EMNLP/questions_dataset_preparation/tasa_fairytaleqa_{PARTITION}_answers_questions_feedback_aware_v2.json', 'w'), indent=4)