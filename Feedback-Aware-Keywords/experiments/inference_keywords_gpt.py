import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import numpy as np
import random
import spacy
from openai import OpenAI
client = OpenAI(
    api_key="TOKEN_KEY",
)
# Set the seed to 42 for random and torch and transformers
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

device = 'mps' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_test = json.load(open('Feedback-Aware-Keywords/keywords_dataset_preparation/sampled_test.json', 'r'))

def generate_response(model_inputs):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": model_inputs}
        ],
        seed=42,
    )

    response = completion.choices[0].message.content
    all_responses = response.split('\n')[:15]
    all_responses = [r.strip() for r in all_responses]

    return all_responses

task_description_prompt = """Generate 15 keywords for the following abstract. Write your selected answers on separate lines. Don't add any additional characters or numbering. Take into consideration the following example:

### Example
Abstract:
The Department of Computer Science at the University of Colorado has recently developed a graduate curriculum in Business-Oriented Computing. The program was developed in recognition of the increasing demand for individuals who are trained in both business methodology and computer science. The 30 semester hour program is designed to produce masters level computer scientists capable of integrating the needs of the business community with the technology of computer science. Prerequisites for the program include a Bachelor's degree in Business (or the equivalent), ten semester hours of computing, and nine semester hours of upper division mathematics. The prerequisite computing courses are: Introduction to Computer Science for Business Majors (CS 202), a four hour course in COBOL programming; Business Data Processing Methods (CS 312), a three hour course in FORTRAN emphasizing business applications; and Assembly Language and System Software (CS 400). The mathematics courses are typically in the areas of statistics, probability theory, mathematical programming, computability, and linear algebra. At the University of Colorado, the programming and mathematics courses can be taken in the undergraduate Computer Based Information Systems option of the Business School curriculum.

### Response:
computer science
statistics
systems
technologies
data processing
methodology
recognition
language
computation
linear algebra
probability
assemblers
information system
mathematical programming
curriculum"""

test_generated_list = []
for data in tqdm(data_test):
    try:
        gen_list = []
        text = data['abstract']

        prompt = f"{task_description_prompt}\nAbstract:\n{text}\n### Response:"

        generated_answers = generate_response(prompt)

        for answer in generated_answers:
            gen_list.append({
                'keyword': answer,
            })
        test_generated_list.append(gen_list)
    except Exception as e:
        test_generated_list.append([])
        print(f"Error at {len(test_generated_list)}")

json.dump(test_generated_list, open('Feedback-Aware-Keywords/experiments/gpt4o_all_one_shot.json', 'w'), indent=4)
