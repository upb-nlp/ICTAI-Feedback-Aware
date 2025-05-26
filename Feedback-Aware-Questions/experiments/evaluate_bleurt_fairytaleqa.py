import json
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from tqdm import tqdm
import torch
import networkx as nx

device='mps'

model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
model.eval()
tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

def get_bleurt_scores(candidates, references):
    scores = []
    batch_size = 64
    for i in range(0, len(references), batch_size):
        end_idx = min(i + batch_size, len(references))
        ref = references[i:end_idx]
        cand = candidates[i:end_idx]
        with torch.no_grad():
            inputs = tokenizer(ref, cand, padding='longest', return_tensors='pt').to(device)
            res = model(**inputs).logits.flatten().tolist()
            scores += res
    return scores

def get_max_bipartite_matching(matrix):
    G = nx.Graph()
    N, M = len(matrix), len(matrix[0])
    G.add_weighted_edges_from((i, N + j, matrix[i][j]) for i in range(N) for j in range(M))

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    mean_cost = sum(G[u][v]['weight'] for u, v in matching) / len(matching)
    return mean_cost

grouped_fairytaleqa = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/grouped_fairytaleqa_test.json', 'r'))
dataset1 = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_test.json', 'r'))


my_files_dict = {
    'Feedback Aware SFT': 'Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_feedback_aware_sft_smart_duplicates.json',
    'Feedback Aware SFT Ablation': 'Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_feedback_aware_sft_ablation_smart_duplicates.json',
    'Single Answer SFT': 'Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_single_answer_sft_smart_duplicates.json',
    'All Answers SFT': 'Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_all_answers_sft_smart_duplicates.json',
    'All Answers SFT with Aid': 'Feedback-Aware-Questions/experiments/tasa_fairytaleqa_test_questions_baseline_all_answers_with_aid_sft_smart_duplicates.json',
}


for name, filename in my_files_dict.items():
    print(name)
    dataset2 = json.load(open(filename, 'r'))[1000:]

    my_dataset = []
    for i in range(len(dataset1)):
        my_dataset.append({
            'context': dataset1[i]['story_section'],
            'generations': dataset2[i],
        })


    used_contexts = set()
    bipartite_scores = []
    for data in tqdm(my_dataset):
        context = data['context']
        if context in used_contexts:
            continue
        used_contexts.add(context)

        good_questions = [gen['question'] + " " + gen['answer'] for gen in data['generations']][:10]
        reference_questions = [q + " " + a for q, a in zip(grouped_fairytaleqa[context]['questions'], grouped_fairytaleqa[context]['answers'])]

        if len(good_questions) == 0 or len(reference_questions) == 0:
            continue

        all_scores = []
        for ref_q in reference_questions:
            scores = get_bleurt_scores(good_questions, [ref_q] * len(good_questions))
            all_scores.append(scores)

        bipartite_score = get_max_bipartite_matching(all_scores)
        bipartite_scores.append(bipartite_score)

    print(sum(bipartite_scores) / len(bipartite_scores))
    print()