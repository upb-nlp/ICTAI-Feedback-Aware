# Prompts

This section contains the prompts used for different models and tasks.
In **bold** we denoted the expected generated text by the model.

## Answer Selection for Question Generation

### Prerequisites: Labeling Models

#### QGen
> Generate a question based on the context and the answer.
>
> Context: {{Context}}
>
> Answer: {{Answer}}
>
> Question: **{{Generated question}}**

#### QAns
> Answer the following question based on the context.
>
> Context: {{Context}}
>
> Question: {{Question}}
>
> Answer: **{{Generated answer}}**


### Feedback-Aware Model
> Iteratively select a span from the following text that would serve as a
good answer for generating a question.
> 
> \### Text: {{Text}}
> 
> \### Response:
> 
> GOOD: {{Previous selected answer}} - {{Previous resulting question}}
> 
> BAD: {{Previous selected answer}} - {{Previous resulting question}}
> 
> ...
> 
> GOOD: {{Previous selected answer}} - {{Previous resulting question}}
> 
> GOOD: **{{Selected answer}}**


### Single Sequence Generation
> Select a span from the following text that would serve as a good answer
for generating a question.
> 
> \### Text: {{Text}}
> 
> \### Response:
> 
> **{{Selected answer}}**


### All Sequences Generation
> Iteratively select a span from the following text that would serve as a
good answer for generating a question.
> 
> \### Text: {{Text}}
> 
> \### Response:
> 
> **{{Selected answer}}**
> 
> **{{Selected answer}}**
> 
> **...**
> 
> **{{Selected answer}}**


### All Sequences Generation with Resulting Content
> Iteratively select a span from the following text that would serve as a
good answer for generating a question.
> 
> \### Text: {{Text}}
> 
> \### Response:
> 
> {{Previous selected answer}} - {{Previous resulting question}}
> 
> {{Previous selected answer}} - {{Previous resulting question}}
> 
> ...
> 
> {{Previous selected answer}} - {{Previous resulting question}}
> 
> **{{Selected answer}}**


### GPT-4o
> Select 25 spans from the following text that would serve as good answers
for generating questions. Write your selected answers together with the
corresponding question on separate lines, in the following format:
<answer> -> <question>
> 
> Don't add any additional characters or numbering. Take into
consideration the following example:
> 
> {{Example text and response}}
> 
> \### Text: {{Text}}
> 
> **{{Selected answer}} - {{Generated question}}**
> 
> **{{Selected answer}} - {{Generated question}}**
> 
> **...**
> 
> **{{Selected answer}} - {{Generated question}}**


## Keyword Generation

### Feedback-Aware Model
> Iteratively select keywords for the following text.
> 
> \### Text: {{Text}}
> 
> \### Response:
> 
> GOOD: {{Previous selected keyword}}
> 
> BAD: {{Previous selected keyword}}
> 
> ...
> 
> GOOD: {{Previous selected keyword}}
> 
> GOOD: **{{Selected keyword}}**


### Single Sequence Generation
> Select keywords for the following text.
> 
> \### Text: {{Text}}
> 
> \### Response:
> 
> **{{Selected keyword}}**


### All Sequences Generation
> Select keywords for the following text.
> 
> \### Text: {{Text}}
> 
> \### Response:
> 
> **{{Selected keyword}}**
> 
> **{{Selected keyword}}**
> 
> **...**
> 
> **{{Selected keyword}}**


### GPT-4o
> Generate 15 keywords for the following abstract. Write your selected
answers on separate lines. Don't add any additional characters or
numbering. Take into consideration the following example:
> 
> {{Example text and response}}
> 
> \### Text: {{Text}}
> 
> **{{Selected keyword}}**
> 
> **{{Selected keyword}}**
> 
> **...**
> 
> **{{Selected keyword}}**

