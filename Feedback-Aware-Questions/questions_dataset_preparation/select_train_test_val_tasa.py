import json
from sklearn.model_selection import train_test_split

dataset = json.load(open('tasa_all.json', 'r'))

# Count the number of examples in each category data['subject']
categories = {}
for data in dataset:
    if data['subject'] not in categories:
        categories[data['subject']] = 0
    categories[data['subject']] += 1

print(categories)

no_train = 10000
no_test = 1000
no_val = 1000


# Split the dataset into train, test and validation using scikit-learn, stratify by data['subject']
train, test_val = train_test_split(dataset, train_size=no_train, test_size=no_test+no_val, stratify=[data['subject'] for data in dataset], random_state=42)
test, val = train_test_split(test_val, test_size=no_val, stratify=[data['subject'] for data in test_val], random_state=42)

print(len(train))
print(len(test))
print(len(val))

# Save the datasets
json.dump(train, open('Failed-EMNLP/tasa_train.json', 'w'))
json.dump(test, open('Failed-EMNLP/tasa_test.json', 'w'))
json.dump(val, open('Failed-EMNLP/tasa_val.json', 'w'))