# dataset = [['this', 'is', 'doc', '1'], ['this', 'is', 'doc', '2']]
from gensim import corpora
from settings.common import load_flat_dataset

dataset = load_flat_dataset('data/sample_tweets.csv', delimiter=' ')
dictionary = corpora.Dictionary(dataset)
dictionary.filter_extremes()
corpus = [dictionary.doc2bow(doc) for doc in dataset]

from gensim.models import FastText
from gensim.models.fasttext import save_facebook_model
from settings.common import load_flat_dataset

dataset_name = 'sample_tweets'
dataset = load_flat_dataset('data/{}.csv'.format(dataset_name))
ft = FastText(sentences=dataset, vector_size=100, min_count=50)
save_facebook_model(ft, 'local_{}_ft.bin'.format(dataset_name))

from tm_pipeline.tndmallet import TndMallet
from tm_pipeline.etndmallet import eTndMallet

tnd_path = 'mallet-tnd/bin/mallet'
etnd_path = 'mallet-etnd/bin/mallet'
mallet_path = 'mallet-2.0.8/bin/mallet'

model1 = TndMallet(tnd_path, corpus, num_topics=30, id2word=dictionary, workers=4,
                   alpha=50, beta=0.01, skew=25, noise_words_max=200, iterations=1000)

model2 = eTndMallet(etnd_path, corpus, num_topics=30, id2word=dictionary, workers=4,
                   alpha=50, beta=0.01, skew=25, noise_words_max=200,
                   tau=200, embedding_path='local_sample_tweets_ft.bin',
                   closest_x_words=3, iterations=1000)

topics = model1.show_topics(num_topics=10, num_words=20, formatted=False)
noise = model1.load_noise_dist()
noise_list = sorted([(x, noise[x]) for x in noise.keys()], key=lambda x: x[1], reverse=True)

from tm_pipeline.nlda import NLDA

model = NLDA(dataset=dataset, tnd_k=30, tnd_alpha=50, tnd_beta0=0.01, tnd_beta1=25, tnd_noise_words_max=200,
                 tnd_iterations=1000, lda_iterations=1000, lda_k=30, nlda_phi=10, nlda_topic_depth=100, top_words=20,
                 save_path='results/', mallet_tnd_path=tnd_path, mallet_lda_path=mallet_path, random_seed=1824, run=True)
