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
