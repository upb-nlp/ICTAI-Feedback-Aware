import json
import numpy as np

def fail_qa_prob(qs):
    if qs['qa_prob'] < QA_PROB_TRESHOLD:
        return True
    return False

def fail_prob_rapport(qs):
    if qs['prob_rapport'] > PROB_RAPPORT_TRESHOLD:
        return True
    return False

FUNCTION_DICT = {
    'qa_prob': fail_qa_prob,
    'prob_rapport': fail_prob_rapport,
}

def respects_tresholds(qs, order):    
    for key in order:
        if FUNCTION_DICT[key](qs):
            return key
    return 'GOOD'

dataset_t_1 = json.load(open('Failed-EMNLP/tasa_dataset_preparation/tasa_train_answers_questions_quality_scores_sum_no_context.json'))
dataset_f_1 = json.load(open('Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_train_answers_questions_quality_scores_sum_no_context.json'))

dataset_t = json.load(open('Failed-EMNLP/tasa_dataset_preparation/tasa_train_answers_questions_quality_scores_sum.json'))
dataset_f = json.load(open('Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_train_answers_questions_quality_scores_sum.json'))

def prob(loss):
    return np.exp(-loss)

for data, data_1 in zip(dataset_t + dataset_f, dataset_t_1 + dataset_f_1):
    for l, l1 in zip(data['sampled_answers_questions'], data_1['sampled_answers_questions']):
        l['quality_scores']['prob_rapport'] = prob(l1['quality_scores']['qa_loss']) / prob(l['quality_scores']['qa_loss'])
        l['quality_scores']['qa_prob'] = prob(l['quality_scores']['qa_loss'])

while True:
    nums = input("Enter two float numbers separated by spaces (qa_prob) (prob_rapport): ").split()
    nums = [float(num) for num in nums]
            
    if len(nums) != 2:
        print("Please enter exactly three numbers.")
        continue
            
    QA_PROB_TRESHOLD = nums[0]
    PROB_RAPPORT_TRESHOLD = nums[1]

    print(f"QA_PROB_TRESHOLD: {QA_PROB_TRESHOLD}")
    print(f"PROB_RAPPORT_TRESHOLD: {PROB_RAPPORT_TRESHOLD}")

    ORDER = input("Enter the order of the tresholds (qa_prob prob_rapport): ").split()

    count_dict = {
        'qa_prob': 0,
        'prob_rapport': 0,
        'GOOD': 0,
    }
    count_empty_texts = 0
    for data in dataset_t + dataset_f:
        has_one_good_question = False
        for sentence_pair in data['sampled_answers_questions']:
            qs = sentence_pair['quality_scores']
            label = respects_tresholds(qs, ORDER)
            count_dict[label] += 1

            if label == 'GOOD':
                has_one_good_question = True

        if not has_one_good_question:
            count_empty_texts += 1
    
    sum_dict = sum(count_dict.values()) + count_empty_texts

    for key, value in count_dict.items():
        print(f"Percentage of {key} quality scores: {value/sum_dict*100}%")
    print(f"Percentage of empty texts: {count_empty_texts/len(dataset_t + dataset_f)*100}%")
