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

nlp = spacy.load("en_core_web_md")
def extract_lemmas(text):
    doc = nlp(text)
    lemmatized_words = []
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADV', 'ADJ', 'AUX']:
            lemmatized_words.append(token.lemma_)

    bag_of_words = set(lemmatized_words)
    return bag_of_words

def is_response_repeated(response, gen_list_responses):
    lemma_response = extract_lemmas(response)
    for gen_response in gen_list_responses:
        lemma_gen_response = extract_lemmas(gen_response)
        if response == gen_response or len(lemma_response.intersection(lemma_gen_response)) > 0:
            return True
    return False

# Set the seed to 42 for random and torch and transformers
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

device = 'mps' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_test = json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/tasa_test.json', 'r')) + json.load(open('Feedback-Aware-Questions/questions_dataset_preparation/fairytaleqa_test.json', 'r'))

data_test = data_test

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

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
    all_responses = response.split('\n')[:25]
    all_responses = [r.split('->') for r in all_responses]
    all_responses = [[r[0].strip(), r[1].strip()] if len(r) == 2 else [r[0].strip(), ''] for r in all_responses]

    return all_responses

task_description_prompt = """Select 25 spans from the following text that would serve as good answers for generating questions. Write your selected answers together with the corresponding question on separate lines, in the following format:\n<answer> -> <question>. Don't add any additional characters or numbering. Take into consideration the following example:

### Example
Text:
Freud's theory of play comes directly from his theory of psycho-sexual maturation. From the moment you were born, Freud said your body began producing libidinal energy. To Freud, high energy levels always mean pain and discomfort, while low energy levels mean pleasure. Anything that increases the amount of libidinal energy inside you leads to tension and unhappiness, while a decrease of energy leads to a feeling of contentment and well-being. When environmental situations cause an increase in tension, you attempt to relive the experience in fantasy. This "acting out" reduces the excitement and hence is rewarding to you. By going over the situation again and again, as a youngster so often does when it plays, you learned to master your frustrations and to release your repressed energy in socially-approved ways. To Freud, every behavior was motivated, that is goal-oriented or purposeful. Play is tension-reducing, therefore it is rewarding and is engaged in by all children. Freud was a genius. His theory of psycho-analysis was in its own way, as bold and creative a step forward as was Darwin's theory of evolution. Freud forced psychologists to consider the symbolic, intra-psychic aspects of much human behavior, and to look at childhood experiences for explanation of the beginnings of many adult behavioral problems. More than that, Freud focused in clearly on the unconscious aspects of cognitive development. Piaget and many other theorists talk almost exclusively about "what goes on in the child's mind," which is to say, these theorists speak primarily of the mental development of the child's left (or dominant) hemisphere. But half a century before the "split brain" research was first published, Sigmund Freud knew that the non-conscious (right hemisphere) aspects of cognitive growth were just as important as were language learning and abstract reasoning. Freud developed his theory the way that you or i would, by reading books, by listening to lectures, and by observing the actions of the people around him as carefully as he could. Most of the data that Freud reported were quite accurate: Children (in our society) do show preferences for the parent of the opposite sex at about age four or five, they do engage in masturbatory play when 3 or 4 and stop doing so (to some extent) at five or six; they do develop fantasies related to their daily frustrations, and their psychological maturation is retarded by parental deprivation and excessive punishment.

### Response:
psycho-analysis ->  What theory did Freud create?
cognitive ->  What type of development did Freud focus on?
reading books ->  What is one of the ways that Freud developed his theory?
our ->  What society does Freud report data for?
excessive punishment ->  What can retard a child's psychological maturation?
libidinal ->  What type of energy did Freud say children's bodies produce?
almost exclusively ->  Do Piaget and other theorists talk about what goes on in a child's mind?
as carefully as he could ->  How did Freud observe the actions of the people around him?
an increase in tension ->  What is caused by an environmental situation?
psychologists ->  Who did Freud force to look at childhood experiences?
environmental situations ->  What causes an increase in tension?
Sigmund ->  What was Freud's first name?
look at childhood experiences for explanation of the beginnings of many adult behavioral problems ->  What did Freud want psychologists to do?
directly ->  How does Freud's theory of play relate to his theory of psycho-sexual maturation?
Most ->  How much of Freud's data was accurate?
focused ->  What did Freud do with the unconscious aspects of cognitive development?
psychological ->  What kind of maturation is retarded by parental deprivation?
behavioral ->  What kind of problems do adults have that can be traced back to childhood?
by reading books, by listening to lectures, and by observing the actions of the people around him as carefully as he could ->  How did Freud develop his theory?
the parent of the opposite sex ->  At about age four or five, children show preferences for what?
his theory of psycho-sexual maturation ->  What is Freud's theory of play based on?
pain and discomfort ->  What does high energy levels mean to Freud?
in fantasy ->  Where do children go to relieve tension?
Darwin ->  Who's theory of evolution was similar to Freud's?
low energy levels ->  What is pleasurable to Freud?"""

test_generated_list = []
for data in tqdm(data_test):
    try:
        gen_list = []
        text = data['text'] if 'text' in data else data['story_section']

        prompt = f"{task_description_prompt}\nText:\n{text}\n### Response:"

        generated_answers = generate_response(prompt)

        for answer in generated_answers:
            gen_list.append({
                'answer': answer[0],
                'question': answer[1],
            })
        test_generated_list.append(gen_list)
    except Exception as e:
        test_generated_list.append([])
        print(f"Error at {len(test_generated_list)}")

json.dump(test_generated_list, open('Feedback-Aware-Questions/experiments/gpt4o_all_one_shot.json', 'w'), indent=4)
