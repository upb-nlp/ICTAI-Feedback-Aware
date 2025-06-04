# _QAns_ and _QGen_ Models

As stated in the article, prerequisite models for the labeling system and
content generation are required to evaluate and train both our approach
and baselines. In our case, $QGen$ is a language model fine-tuned for
question generation and $QAns$ is fine-tuned for question answering.

We fine-tune these models on three established datasets: a) SQuAD [^1] - one of the most widely used resources for
question answering and generation, SQuAD consists of over 100K
question-answer pairs derived from a pool of 5K Wikipedia articles; b)
HotpotQA [^2] - designed to test a model's ability to
answer questions that require reasoning across multiple paragraphs,
HotpotQA contains questions that should be answered by bridging
information from two different Wikipedia articles; and c) NarrativeQA
[^3] - designed to assess reading comprehension,
particularly for lengthy texts, NarrativeQA consists of stories, along
with corresponding questions and answers.

The models were independently fine-tuned. The loss is computed only on the `{{Generated question}}` and `{{Generated answer}}` tokens, respectively. More details about the prompts can be found in `Prompts.md`.

These models are not proposed as state-of-the-art for these tasks but
rather as plug-and-play modules independent of our proposed method or
baselines. This means that any model can be a prerequisite as the
proposed method is not dependent on the choice.

In order to assess the performance of these models and validate their
usage, we computed the BLEURT score [^4] with the
ground-truth for the test partition of the SQuAD, HotpotQA, and
NarrativeQA, and for the test partition of FairytaleQA
[^5] (a dataset used for our method and baselines, but
on which we did not train $QGen$ and $QAns$).

The table below showcases the BLEURT scores
for the prerequisites models. Performance is acceptable as the models
are capable of answering and generating questions with accuracy,
considering the context. Moreover, the models generalize well on an
unseen dataset (FairytaleQA), highlighting their capability to serve as
suitable prerequisites for our tasks with a diverse range of texts from
different domains.

|               | **QGen** | **QAns** |
|---------------|---------:|---------:|
| **SQuAD**     |    0.58  |    0.75  |
| **HotpotQA**  |    0.50  |    0.50  |
| **NarrativeQA** |  0.54  |    0.60  |
| **FairytaleQA** |  0.51  |    0.45  |

### References

[^1]: P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang,
“SQuAD: 100,000+ questions for machine comprehension of text,” in Proceedings of the 2016 Conference
on Empirical Methods in Natural Language Processing,
J. Su, K. Duh, and X. Carreras, Eds. Austin, Texas:
Association for Computational Linguistics, Nov. 2016,
pp. 2383–2392.

[^2]: Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen,
R. Salakhutdinov, and C. D. Manning, “HotpotQA: A
dataset for diverse, explainable multi-hop question answering,” in Proceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing,
E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii,
Eds. Brussels, Belgium: Association for Computational
Linguistics, Oct.-Nov. 2018, pp. 2369–2380

[^3]: T. Kocisky, J. Schwarz, P. Blunsom, C. Dyer, K. M. Hermann, G. Melis, and E. Grefenstette, “The NarrativeQA
reading comprehension challenge,” Transactions of the
Association for Computational Linguistics, vol. 6, pp.
317–328, 2018.

[^4]: T. Sellam, D. Das, and A. Parikh, “BLEURT: Learning
robust metrics for text generation,” in Proceedings of the
58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and
J. Tetreault, Eds. Online: Association for Computational
Linguistics, Jul. 2020, pp. 7881–7892.

[^5]: Y. Xu, D. Wang, M. Yu, D. Ritchie, B. Yao, T. Wu,
Z. Zhang, T. Li, N. Bradford, B. Sun, T. Hoang,
Y. Sang, Y. Hou, X. Ma, D. Yang, N. Peng, Z. Yu,
and M. Warschauer, “Fantastic questions and where to
find them: FairytaleQA – an authentic dataset for narrative comprehension,” in Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), S. Muresan, P. Nakov, and
A. Villavicencio, Eds. Dublin, Ireland: Association for
Computational Linguistics, May 2022, pp. 447–460.


