import os
import math
import random
from gensim import corpora
from gensim.models import FastText
from gensim.models.fasttext import save_facebook_model
from settings.common import save_topics, load_flat_dataset
from tm_pipeline.tndmallet import TndMallet
from tm_pipeline.ldamallet import LdaMallet
from tm_pipeline.etndmallet import eTndMallet


def run_LDA(dataset, dataset_name, mallet_path, param_combos, results_path):
    '''
    Run an instance of LDA for each parameter setting
    :param dataset:
    :param dataset_name:
    :param mallet_path:
    :param param_combos:
    :return:
    '''
    if not os.path.exists('{}/{}/lda/'.format(results_path, dataset_name)):
        os.makedirs('{}/{}/lda/'.format(results_path, dataset_name))
    dictionary = corpora.Dictionary(dataset)
    dictionary.filter_extremes()
    corpus = [dictionary.doc2bow(doc) for doc in dataset]
    for param_combo in param_combos:
        k = param_combo[0]
        model = LdaMallet(mallet_path, corpus, num_topics=k, id2word=dictionary)
        topics = model.show_topics(num_topics=k, num_words=20, formatted=False)
        topic_words = []
        for topic in topics:
            t = [w for (w, _) in topic[1]]
            topic_words.append(t)
        save_topics(topic_words, '{}/{}/lda/topics_{}.csv'.format(results_path, dataset_name, k))


def run_TND_MALLET(dataset, dataset_name, mallet_path, param_combos, results_path):
    if not os.path.exists('{}/{}/tnd/'.format(results_path, dataset_name)):
        os.makedirs('{}/{}/tnd/'.format(results_path, dataset_name))
    dictionary = corpora.Dictionary(dataset)
    dictionary.filter_extremes()
    corpus = [dictionary.doc2bow(doc) for doc in dataset]
    model = None
    for param_combo in param_combos:
        k = param_combo[0]
        alpha = param_combo[1]
        beta = param_combo[2]
        skew = param_combo[3]
        nwm = param_combo[4]
        iterations = param_combo[5]

        model = TndMallet(mallet_path, corpus, num_topics=k, id2word=dictionary, workers=4,
                          alpha=alpha, beta=beta, skew=skew, noise_words_max=nwm, iterations=iterations)
        topics = model.show_topics(num_topics=k, num_words=20, formatted=False)
        noise = model.load_noise_dist()
        noise_list = sorted([(x, noise[x]) for x in noise.keys()], key=lambda x: x[1], reverse=True)
        topic_words = []
        for topic in topics:
            t = [w for (w, _) in topic[1]]
            topic_words.append(t)
        save_topics(topic_words, '{}/{}/tnd/topics_{}_{}.csv'.format(results_path, dataset_name, k, skew))

        with open('{}/{}/tnd/noise_{}_{}.csv'.format(results_path, dataset_name, k, skew), 'w') as f:
            for pair in noise_list:
                f.write('{},{}\n'.format(pair[0], pair[1]))
    return model


def run_ETND_MALLET(dataset, dataset_name, mallet_path, param_combos, results_path):
    if not os.path.exists('{}/{}/etnd/'.format(results_path, dataset_name)):
        os.makedirs('{}/{}/etnd/'.format(results_path, dataset_name))
    dictionary = corpora.Dictionary(dataset)
    dictionary.filter_extremes()
    corpus = [dictionary.doc2bow(doc) for doc in dataset]
    model = None
    for param_combo in param_combos:
        k = param_combo[0]
        alpha = param_combo[1]
        beta = param_combo[2]
        skew = param_combo[3]
        nwm = param_combo[4]
        embedding_path = param_combo[5].format(dataset_name)
        closest_x_words = param_combo[6]
        tau = param_combo[7]
        iterations = param_combo[8]

        model = eTndMallet(mallet_path, corpus, num_topics=k, id2word=dictionary, workers=4,
                          alpha=alpha, beta=beta, skew=skew, noise_words_max=nwm, tau=tau, embedding_path=embedding_path,
                           closest_x_words=closest_x_words, iterations=iterations)
        topics = model.show_topics(num_topics=k, num_words=20, formatted=False)
        noise = model.load_noise_dist()
        noise_list = sorted([(x, noise[x]) for x in noise.keys()], key=lambda x: x[1], reverse=True)
        topic_words = []
        for topic in topics:
            t = [w for (w, _) in topic[1]]
            topic_words.append(t)
        save_topics(topic_words, '{}/{}/etnd/topics_{}_{}_{}.csv'.format(results_path, dataset_name, k, skew, closest_x_words))

        with open('{}/{}/etnd/noise_{}_{}_{}.csv'.format(results_path, dataset_name, k, skew, closest_x_words), 'w') as f:
            for pair in noise_list:
                f.write('{},{}\n'.format(pair[0], pair[1]))
    return model


def run_NLDA(dataset, dataset_name, mallet_path, nft_mallet_path, param_combos, results_path, noise_dist=None):
    '''

    :param dataset:
    :param dataset_name:
    :param mallet_path:
    :param param_combos:
    :return:
    '''
    if not os.path.exists('{}/{}/nlda/'.format(results_path, dataset_name)):
        os.makedirs('{}/{}/nlda/'.format(results_path, dataset_name))
    dictionary = corpora.Dictionary(dataset)
    dictionary.filter_extremes()
    corpus = [dictionary.doc2bow(doc) for doc in dataset]
    model_tuple = [noise_dist, None]
    for param_combo in param_combos:
        noise_params = param_combo[1]
        lda_param_combos = param_combo[0]
        topic_weights = param_combo[2]

        if noise_dist is None:
            model = run_TND_MALLET(dataset, dataset_name, nft_mallet_path, [noise_params], results_path=results_path)
            noise_dist = model.load_noise_dist()
            model_tuple[0] = noise_dist
        for lda_params in lda_param_combos:
            lda_k = lda_params[0]
            model = LdaMallet(mallet_path, corpus, num_topics=lda_k, id2word=dictionary)
            model_tuple[1] = model
            topic_word_distribution = model.load_word_topics()
            topics = model.show_topics(num_topics=lda_k, num_words=100, formatted=False)
            for topic_weight in topic_weights:
                final_topics = []
                for i in range(0, len(topics)):
                    topic = [w for (w, _) in topics[i][1]]
                    final_topic = []
                    j = 0
                    while len(final_topic) < 20 and j < 100  and j < len(topic):
                        w = topic[j]
                        id = dictionary.token2id[w]
                        beta = 2
                        if w in noise_dist:
                            beta += noise_dist[w]
                        beta = max(2, beta * (topic_weight / lda_k))
                        alpha = 2 + topic_word_distribution[i, id]
                        roll = random.betavariate(alpha=math.sqrt(alpha), beta=math.sqrt(beta))
                        if roll >= 0.5:
                            final_topic.append(w)
                            if not w in noise_dist:
                                noise_dist[w] = 0
                            noise_dist[w] += (alpha - 2)
                        j += 1
                    final_topics.append(final_topic)
                param_string = '_'.join([str(x).replace('.', '-')
                                         for x in lda_params])
                save_topics(final_topics, '{}/{}/nlda/topics_{}_{}.csv'
                            .format(results_path, dataset_name, param_string, topic_weight))
    return model_tuple


def run_eNLDA(dataset, dataset_name, mallet_path, nft_mallet_path, param_combos, results_path, noise_dist=None):
    '''

    :param dataset:
    :param dataset_name:
    :param mallet_path:
    :param param_combos:
    :return:
    '''
    if not os.path.exists('{}/{}/enlda/'.format(results_path, dataset_name)):
        os.makedirs('{}/{}/enlda/'.format(results_path, dataset_name))
    dictionary = corpora.Dictionary(dataset)
    dictionary.filter_extremes()
    corpus = [dictionary.doc2bow(doc) for doc in dataset]
    model_tuple = [noise_dist, None]
    for param_combo in param_combos:
        noise_params = param_combo[1]
        lda_param_combos = param_combo[0]
        topic_weights = param_combo[2] # tuple of topic weights to try

        if noise_dist is None:
            model = run_ETND_MALLET(dataset, dataset_name, nft_mallet_path, [noise_params], results_path=results_path)
            noise_dist = model.load_noise_dist()
            model_tuple[0] = noise_dist
        for lda_params in lda_param_combos:
            lda_k = lda_params[0]
            model = LdaMallet(mallet_path, corpus, num_topics=lda_k, id2word=dictionary)
            model_tuple[1] = model
            topic_word_distribution = model.load_word_topics()
            topics = model.show_topics(num_topics=lda_k, num_words=100, formatted=False)
            for topic_weight in topic_weights:
                final_topics = []
                for i in range(0, len(topics)):
                    topic = [w for (w, _) in topics[i][1]]
                    final_topic = []
                    j = 0
                    while len(final_topic) < 20 and j < 100  and j < len(topic):
                        w = topic[j]
                        id = dictionary.token2id[w]
                        beta = 2
                        if w in noise_dist:
                            beta += noise_dist[w]
                        beta = max(2, beta * (topic_weight / lda_k))
                        alpha = 2 + topic_word_distribution[i, id]
                        roll = random.betavariate(alpha=math.sqrt(alpha), beta=math.sqrt(beta))
                        if roll >= 0.5:
                            final_topic.append(w)
                            if not w in noise_dist:
                                noise_dist[w] = 0
                            noise_dist[w] += (alpha - 2)
                        j += 1
                    final_topics.append(final_topic)
                param_string = '_'.join([str(x).replace('.', '-')
                                         for x in lda_params])
                save_topics(final_topics, '{}/{}/enlda/topics_{}_{}_{}.csv'
                            .format(results_path, dataset_name, param_string, topic_weight, noise_params[6])) # noise_params[6] = closest x words
    return model_tuple


def main():
    mallet_path = 'mallet-2.0.8/bin/mallet'
    tnd_path = 'mallet-tnd/bin/mallet'
    etnd_path = 'mallet-etnd/bin/mallet'

    results_path = 'results'
    dataset_names = ['sample_tweets']

    lda_params = [
        # k
        (30,)
    ]

    tnd_params = [
        # k, alpha, beta, skew, noise_words_max, iterations
        (30, 50, 0.01, 25, 200, 1000),
    ]
    etnd_params = [
        # k, alpha, beta, skew, noise_words_max, embedding_path, closest_x_words, tau (number of iterations before embedding activation), iterations
        # (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 3, 200, 1000),
        (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 5, 200, 1000),
        # (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 10, 200, 1000),
    ]
    nlda_params = [
        # lda_k, nft_k, alpha, beta, skew, noise_words_max, iterations, phi values tuple
        (((10,), (20,), (30,)), (30, 0.1, 0.01, 25, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
    ]
    enlda_params = [
        # lda_k, nft_k, alpha, beta, skew, noise_words_max, embedding_path, closest_x_words, tau (number of iterations before embedding activation), iterations, phi values tuple
        # (((30,),), (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 3, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
        (((30,),), (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 5, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
        # (((30,),), (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 10, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
    ]

    for dataset_name in dataset_names:
        dataset = load_flat_dataset('data/{}.csv'.format(dataset_name))

        # train fasttext vectors if needed
        ft = FastText(sentences=dataset, vector_size=100, min_count=50)
        save_facebook_model(ft, 'local_{}_ft.bin'.format(dataset_name))

        # run LDA
        model0 = run_LDA(dataset, dataset_name, mallet_path, lda_params, results_path)

        # run TND without embeddings
        model1 = run_TND_MALLET(dataset, dataset_name, tnd_path, tnd_params, results_path)

        # run TND with embeddings
        model2 = run_ETND_MALLET(dataset, dataset_name, etnd_path, etnd_params, results_path)

        # this will compute TND and LDA from scratch
        model3 = run_NLDA(dataset, dataset_name, mallet_path, tnd_path, nlda_params, results_path)

        # this will compute LDA from scratch, but use the noise distribution calculated in model2 to save computation time
        model4 = run_eNLDA(dataset, dataset_name, mallet_path, etnd_path, enlda_params, results_path,
                           noise_dist=model2.load_noise_dist())


if __name__ == '__main__':
    main()
