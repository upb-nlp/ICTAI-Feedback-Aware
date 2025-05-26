import json
import numpy as np

def prob(loss):
    return np.exp(-loss)


train_t = json.load(open('Failed-EMNLP/tasa_dataset_preparation/tasa_train_answers_questions_quality_scores_sum.json'))
train_t_no_context = json.load(open('Failed-EMNLP/tasa_dataset_preparation/tasa_train_answers_questions_quality_scores_sum_no_context.json'))
train_f = json.load(open('Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_train_answers_questions_quality_scores_sum.json'))
train_f_no_context = json.load(open('Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_train_answers_questions_quality_scores_sum_no_context.json'))
val_t = json.load(open('Failed-EMNLP/tasa_dataset_preparation/tasa_val_answers_questions_quality_scores_sum.json'))
val_t_no_context = json.load(open('Failed-EMNLP/tasa_dataset_preparation/tasa_val_answers_questions_quality_scores_sum_no_context.json'))
val_f = json.load(open('Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_val_answers_questions_quality_scores_sum.json'))
val_f_no_context = json.load(open('Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_val_answers_questions_quality_scores_sum_no_context.json'))

for data, data_no_context in zip(train_t + train_f + val_t + val_f, train_t_no_context + train_f_no_context + val_t_no_context + val_f_no_context):
    for l, l_no_context in zip(data['sampled_answers_questions'], data_no_context['sampled_answers_questions']):
        l['quality_scores']['prob_rapport'] = prob(l_no_context['quality_scores']['qa_loss']) / prob(l['quality_scores']['qa_loss'])
        l['quality_scores']['qa_prob'] = prob(l['quality_scores']['qa_loss'])

QA_PROB_TRESHOLD = 0.03
PROB_RAPPORT_TRESHOLD = 1.0

for dataset in [train_t, train_f, val_t, val_f]:
    for data in dataset:
        for l in data['sampled_answers_questions']:
            if l['quality_scores']['qa_prob'] < QA_PROB_TRESHOLD or l['quality_scores']['prob_rapport'] > PROB_RAPPORT_TRESHOLD:
                l['type'] = 'BAD'
            else:
                l['type'] = 'GOOD'

            del l['quality_scores']

json.dump(train_t, open('Failed-EMNLP/tasa_dataset_preparation/tasa_train_answers_questions_labeled_v2.json', 'w'), indent=4)
json.dump(train_f, open('Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_train_answers_questions_labeled_v2.json', 'w'), indent=4)
json.dump(val_t, open('Failed-EMNLP/tasa_dataset_preparation/tasa_val_answers_questions_labeled_v2.json', 'w'), indent=4)
json.dump(val_f, open('Failed-EMNLP/fairytaleqa_dataset_preparation_question/fairytaleqa_val_answers_questions_labeled_v2.json', 'w'), indent=4)