import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import json
from tqdm import tqdm
device='mps'

model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

dataset = json.load(open("Feedback-Aware-Questions/prerequisites_models/generated_answers_fairytaleqa_deterministic.json", "r"))

task = 'answer'
#task = 'question'

references = [data[task] for data in dataset]
candidates = [data[f"generated_{task}"] for data in dataset]

scores = []
batch_size = 8
model.eval()
for i in tqdm(range(0, len(references), batch_size)):
    end_idx = min(i + batch_size, len(references))
    ref = references[i:end_idx]
    cand = candidates[i:end_idx]
    with torch.no_grad():
        inputs = tokenizer(ref, cand, padding='longest', return_tensors='pt').to(device)
        res = model(**inputs).logits.flatten().tolist()
        scores += res

print(sum(scores) / len(scores))

