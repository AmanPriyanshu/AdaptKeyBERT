<img src="https://github.com/AmanPriyanshu/AdaptKeyBERT/blob/master/images/keybert_logo.png" width="35%" height="35%" align="right" />

# AdaptKeyBERT

**TLDR;** *Keyword/keyphrase extraction with zero-shot and few-shot semi-supervised domain adaptation.*

KeyBERT is a minimal and easy-to-use keyword extraction technique that leverages BERT embeddings to create keywords and keyphrases that are most similar to a document

AdaptKeyBERT expands the aforementioned library by integrating semi-supervised attention for creating a few-shot domain adaptation technique for keyphrase extraction. Also extended the work by allowing zero-shot word seeding, allowing better performance on topic relevant documents

# Our Aim:

* We reconsider downstream training keyword extractors on varied domains by integrating pre-trained LLMs with Few-Shot and Zero-Shot paradigms for domain accommodation. We incorporate regularized attention based embedding reconstruction for domain attentive keyword extraction.

* Demonstrate two experimental setting with the objectives of achieving high performance for **Few-Shot Domain Adaptation** & **Zero-Shot Domain Adaptation**. The experimental results on both objective benchmarks demonstrate that our framework outperforms the base/naive approaches.

* We open source a python library (AdaptKeyBERT) for the construction of FSL/ZSL for keyword extraction models that employ LLMs directly integrated with the KeyBERT API. Allowing easy training, validation, and deployment of said models.


# Our Pipeline:

<img src="https://github.com/AmanPriyanshu/AdaptKeyBERT/blob/master/images/flowcharts.png" align="center" />

# Results: 

Datasets Used: 

* fao780 dataset (Food and Agriculture Organization) - 780 documents from the food and agriculture industry.
* CERN-290 dataset - 290 high energy physics documents.

## Results: 

Table: AdaptKeyBERT performance on FAO-780 dataset with p%=10%.

| Model                | Precision | Recall | F-Score |
|----------------------|--------------------|-----------------|------------------|
| Benchmark            | 36.74              | 33.67           | 35.138           |
| Zero-Shot            | 37.25              | 38.59           | 37.908           |
| Few-Shot             | 40.03              | 39.1            | 39.559           |
| Zero-Shot & Few-Shot | 40.02              | 39.86           | 39.938           |
|                      |                    |                 |                  |


Table: AdaptKeyBERT performance on CERN-290 dataset with p%=10%.

| Model                | Precision | Recall | F-Score |
|----------------------|-----------|--------|------------------|
| Benchmark            | 24.74     | 26.58  | 25.627           |
| Zero-Shot            | 27.35     | 25.9   | 26.605           |
| Few-Shot             | 29        | 27.4   | 28.177           |
| Zero-Shot & Few-Shot | 29.11     | 28.67  | 28.883           |


## Installation:

`pip install adaptkeybert`

## Basic Use:

Take a look at `runner.py`

```py
from adaptkeybert import KeyBERT

doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias). But then what about supervision and unsupervision, what happens to unsupervised learning.
      """
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(doc, top_n=10)
print(keywords)


kw_model = KeyBERT(domain_adapt=True)
kw_model.pre_train([doc], [['supervised', 'unsupervised']], lr=1e-3)
keywords = kw_model.extract_keywords(doc, top_n=10)
print(keywords)


kw_model = KeyBERT(zero_adapt=True)
kw_model.zeroshot_pre_train(['supervised', 'unsupervised'], adaptive_thr=0.15)
keywords = kw_model.extract_keywords(doc, top_n=10)
print(keywords)


kw_model = KeyBERT(domain_adapt=True, zero_adapt=True)
kw_model.pre_train([doc], [['supervised', 'unsupervised']], lr=1e-3)
kw_model.zeroshot_pre_train(['supervised', 'unsupervised'], adaptive_thr=0.15)
keywords = kw_model.extract_keywords(doc, top_n=10)
print(keywords)

```
