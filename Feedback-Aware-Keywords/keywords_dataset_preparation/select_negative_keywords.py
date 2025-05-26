import hnswlib
import numpy as np
import random
import json
from collections import Counter
from tqdm import tqdm
random.seed(42)

def levenshtein_distance(s1, s2):
    dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1
                )
    
    return dp[-1][-1]

PARTITION = 'val' # 'train' or 'val'
keywords_with_embeddings = json.load(open(f"Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_train_keywords_with_embeddings.json", "r"))
dataset_train = json.load(open(f"Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_train_with_abstract_embeddings.json", "r"))
dataset = json.load(open(f"Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_{PARTITION}_with_abstract_embeddings.json", "r"))

counter_keywords = Counter()
for data in dataset_train:
    for keyword in data['keywords']:
        if keyword.strip():
            counter_keywords[keyword.strip()] += 1

frequent_keywords_with_embeddings = [keyword for keyword in keywords_with_embeddings if counter_keywords[keyword['keyword']] >= 10]

data = np.array([keyword['embedding'] for keyword in frequent_keywords_with_embeddings])
ids = np.array(range(len(frequent_keywords_with_embeddings)))
dim = len(data[0])

index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=len(data), ef_construction=200, M=64)
index.add_items(data, ids)


for data in tqdm(dataset):
    index.set_ef(200)
    labels, distances = index.knn_query(np.array(data['abstract_embedding']), k=max(len(data['keywords']) * 10, 20))
    labels = labels[0]
    distances = distances[0]

    selected_keywords = []
    for i in labels:
        keyword = frequent_keywords_with_embeddings[i]['keyword']
        ok = True
        data["keywords"] = [keyword.strip() for keyword in data["keywords"] if keyword.strip()]
        for keyword_data in data['keywords']:
            words_a = set(keyword.split())
            words_b = set(keyword_data.split())
            if words_a <= words_b or words_b <= words_a or levenshtein_distance(keyword, keyword_data) < 5:
                ok = False
                break
        if ok:
            selected_keywords.append(keyword)
    try:
        data['negative_keywords'] = random.sample(selected_keywords, len(data['keywords']))
    except:
        print(data['keywords'])
    del data['abstract_embedding']

json.dump(dataset, open(f"Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_{PARTITION}_with_negative_keywords.json", "w"), indent=4)

