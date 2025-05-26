import json
import random
random.seed(42)

for partition in ["train", "test", "val"]:
    dataset = json.load(open(f"Feedback-Aware-Keywords/keywords_dataset_preparation/clean_{partition}.json", "r"))

    dataset = random.sample(dataset, int(len(dataset) * 0.1))

    json.dump(dataset, open(f"Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_{partition}.json", "w"), indent=4)