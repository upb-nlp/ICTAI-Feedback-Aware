import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to('mps')

batch_size = 16

for PARTITION in ['train', 'val']:
    dataset = json.load(open(f"Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_{PARTITION}.json", "r"))
    abstracts = [data['abstract'] for data in dataset]

    for i in tqdm(range(0, len(abstracts), batch_size)):
        end_index = min(i + batch_size, len(abstracts))
        batch_abstracts = abstracts[i:end_index]
        encoded_input = tokenizer(batch_abstracts, padding=True, truncation=True, return_tensors='pt').to('mps')
        with torch.no_grad():
            model_output = model(**encoded_input)
        abstracts_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        abstracts_embeddings = F.normalize(abstracts_embeddings, p=2, dim=1)

        for abs_emb, data in zip(abstracts_embeddings, dataset[i:end_index]):
            data['abstract_embedding'] = abs_emb.tolist()

    json.dump(dataset, open(f"Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_{PARTITION}_with_abstract_embeddings.json", "w"), indent=4)

    keywords_set = set()
    for data in dataset:
        for keyword in data['keywords']:
            keywords_set.add(keyword)
    keywords = list(keywords_set)
    keywords_with_embeddings = []

    for i in tqdm(range(0, len(keywords), batch_size)):
        end_index = min(i + batch_size, len(keywords))
        batch_keywords = keywords[i:end_index]
        encoded_input = tokenizer(batch_keywords, padding=True, truncation=True, return_tensors='pt').to('mps')
        with torch.no_grad():
            model_output = model(**encoded_input)
        keywords_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        keywords_embeddings = F.normalize(keywords_embeddings, p=2, dim=1)

        for keyword_emb, keyword in zip(keywords_embeddings, batch_keywords):
            keywords_with_embeddings.append({
                'keyword': keyword,
                'embedding': keyword_emb.tolist()
            })

    json.dump(keywords_with_embeddings, open(f"Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_{PARTITION}_keywords_with_embeddings.json", "w"), indent=4)
