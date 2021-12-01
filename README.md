# Topic Noise Discriminator (TND) and Noiseless LDA (NLDA)
Implementations of Topic Noise Discriminator and NLDA, along with our evaluation metrics, can be found here.

### Requirements and Setup
to install relevant Python requirements:
> pip install -r requirements.txt

You must have the Java JDK installed on your computer to run TND. It can be downloaded [here](https://www.oracle.com/java/technologies/javase-downloads.html).  We originally built this with JDK 11, but have tested with JDK 8 and 16.

### Using TND and NLDA
All of the code in this section, with the exception of pseudocode, is included in the script `readme_script.py`.  There is another script called `run_models.py` that can be used to test models at scale using sets of parameters.

**Loading and Preparing Data for Modeling.**
Data sets should be loaded as a list of documents, where each document is a list of words.  We have a built-in function to load data.  We also convert the data set to a gensim corpus for use in our models.
```python
# dataset = [['this', 'is', 'doc', '1'], ['this', 'is', 'doc', '2']]
from gensim import corpora
from settings.common import load_flat_dataset

dataset = load_flat_dataset('data/sample_tweets.csv', delimiter=' ')
dictionary = corpora.Dictionary(dataset)
dictionary.filter_extremes()
corpus = [dictionary.doc2bow(doc) for doc in dataset]
```

**Training Embeddings.**
TND is designed to use an embedding space to get a more complete noise distribution for topics.  We used FastText to train an embedding space on our data set.  Any embedding space can hypothetically be used, but we found that training on the data itself provided the best vectors for the purposes of identifying noise.  Below we show how to train FastText vectors on our data set.  Notice the `min_count` parameter.  It refers to the minimum frequency of a word in the data set in order for its vector to be computed.  Make sure it is set to something reasonable for the size of the data set being used.
```python
from gensim.models import FastText
from gensim.models.fasttext import save_facebook_model
from settings.common import load_flat_dataset

dataset_name = 'sample_tweets'
dataset = load_flat_dataset('data/{}.csv'.format(dataset_name))
ft = FastText(sentences=dataset, vector_size=100, min_count=50)
save_facebook_model(ft, 'local_{}_ft.bin'.format(dataset_name))
```

**Parameters.**
Lines 230-249 of `run_models.py` contain example parameter settings for each of our models.
Here, we explain the parameters of TND and then of NLDA.

**TND Parameters.**
* k: the number of topics to approximate
* alpha: hyper-parameter tuning number of topics per document
* beta (Beta_0 in the paper): hyper-parameter tuning number of topics per word
* skew (Beta_1 in the paper): hyper-parameter tuning topic probability of a given word compared to its noise probability. Higher means more weight is given to the topic (words are less likely to be noise).
* noise_words_max: the number of noise words to save to a noise words file for use as a context noise list (words with highest probability of being noise)
* iterations: the number of iterations to run inference of topics

**TND Parameters for Embedding Space Incorporation.**
The version of TND that employs embedding spaces is split out in the Java implementation (based on the Mallet LDA implementation) to simplify the code base and keep the non-embedding version as fast as possible.
* embedding_path: the path to the trained embedding space
* closest_x_words (mu in the paper): the number of words to be sampled from the embedding space for each observed word (based on distance in the embedding space)
* tau: the number of iterations before using the embedding space (aka burnin period)

**NLDA Parameters.**
NLDA comprises TND and LDA (or whichever generative method one wants).  As such, it contains the parameters of its component models, as well as:
* nlda_phi: similar to skew from TND, this is a hyper-parameter that tunes the topic probability of a given word compared to its noise probability during the ensemble phase.
* nlda_topic_depth: the number of top-probability topics words to consider when removing noise.  This limits the computation time of the ensemble and ensures that we will still filter noise from the topic words we are going to actually look at.

**Running TND and eTND.**
We used the old Gensim wrapper for Mallet LDA to create wrappers for TND.
The `workers` parameter is the number of threads to dedicate to running the model.  We have found that four is sufficient for many mid-sized data sets on our servers.
They can be used as follows (assuming we've already loaded and prepped our data):
```python
from tm_pipeline.tndmallet import TndMallet
from tm_pipeline.etndmallet import eTndMallet

tnd_path = 'mallet-tnd/bin/mallet'
etnd_path = 'mallet-etnd/bin/mallet'

model1 = TndMallet(tnd_path, corpus, num_topics=30, id2word=dictionary, workers=4,
                   alpha=50, beta=0.01, skew=25, noise_words_max=200, iterations=1000)

model2 = eTndMallet(etnd_path, corpus, num_topics=30, id2word=dictionary, workers=4,
                   alpha=50, beta=0.01, skew=25, noise_words_max=200,
                   tau=200, embedding_path='local_sample_tweets_ft.bin',
                   closest_x_words=3, iterations=1000)

topics = model1.show_topics(num_topics=k, num_words=20, formatted=False)
noise = model1.load_noise_dist()
noise_list = sorted([(x, noise[x]) for x in noise.keys()], key=lambda x: x[1], reverse=True)
```

**Running NLDA and eNLDA.**
We can run NLDA and eNLDA in one go using their classes in `tm_pipeline`.
```python
from tm_pipeline.nlda import NLDA

mallet_path = 'mallet-2.0.8/bin/mallet'

model = NLDA(dataset=dataset, tnd_k=30, tnd_alpha=50, tnd_beta0=0.01, tnd_beta1=25, tnd_noise_words_max=200,
                 tnd_iterations=1000, lda_iterations=1000, lda_k=30, nlda_phi=10, nlda_topic_depth=100, top_words=20,
                 save_path='results/', mallet_tnd_path=tnd_path, mallet_lda_path=mallet_path, random_seed=1824, run=True)
```

Setting `run=True` here (the default) will result in NLDA being run through on initialization.  Setting it to false allows one to go through the model one step at a time, like so:
```python 
    model.prepare_data()
    model.compute_tnd()
    model.compute_lda()
    model.compute_nlda()
```

We can also pass through a pre-computed TND or LDA model, or both.
```python
from tm_pipeline.ldamallet import LdaMallet
from tm_pipeline.tndmallet import TndMallet

lda_model = LdaMallet(dataset, other_parameters)
tnd_model = TndMallet(dataset, other_parameters)

nlda = NLDA(dataset, tnd_noise_distribution=tnd_model.load_noise_dist(), 
            lda_tw_dist=lda_model.load_word_topics(), nlda_phi=10, 
            nlda_topic_depth=100, save_path='results/', run=True)
```


**Testing many configurations at once.** 
In performing research on topic models, we often want to run a bunch of model parameter settings at once.  In the `run_models.py` file, we have wrappers for Mallet LDA, TND, NLDA, and their embedded versions (eNLDA, eTND).  These are essentially deconstructed versions of the NLDA and eNLDA classes that allow for easier customization and for a lot of repeated experiments.

### Referencing TND and NLDA
```
Churchill, Rob and Singh, Lisa. 2021. Topic-Noise Models: Modeling Topic and Noise Distributions in Social Media Post Collections. International Conference on Data Mining (ICDM).
```

```bibtex 
@inproceedings{churchill2021tnd,
author = {Churchill, Rob and Singh, Lisa},
title = {Topic-Noise Models: Modeling Topic and Noise Distributions in Social Media Post Collections},
booktitle = {ICDM 2021},
year = {2021},
}
```

### Citations
```
A. K. McCallum, “Mallet: A machine learning for language toolkit.”
2002.
```

```
P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, "Enriching Word Vectors with Subword Information." 2016.
```

```
R. Rehurek, P. Sojka, "Gensim–python framework for vector space modelling." 2011.
```
