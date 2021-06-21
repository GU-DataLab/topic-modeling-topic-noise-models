from settings.common import (load_flat_dataset, load_topics, load_noise_words,
                             word_co_frequency, word_frequency)
from .evaluation_metrics import (topic_coherence, topic_diversity,
                                 noise_penetration)


def save_metric_results(path, label, scores):
    for i in range(0, len(scores)):
        scores[i] = str(round(scores[i], 3))
    with open(path, 'a') as f:
        f.write('{},{}\n'.format(label, ','.join(scores)))


def analyze_dataset(dataset_path, noise_path=None):
    dataset = load_flat_dataset(dataset_path)
    freqs = {}
    freqs = word_frequency(freqs, dataset)
    cofreqs = {}
    cofreqs = word_co_frequency(cofreqs, dataset)
    noise_words = []
    if noise_path:
        noise_words = load_noise_words(noise_path)
    return freqs, cofreqs, noise_words


def compute_metrics(topics, freqs, cofreqs, noise_words, k=30):
    coherence_score = topic_coherence(topics, freqs, cofreqs, k)
    diversity_score = topic_diversity(topics, k)
    noise_score = 0
    if noise_words and len(noise_words) > 0:
        noise_score = noise_penetration(topics, noise_words, k)
    return [coherence_score, diversity_score, noise_score]


def compute_all_results(topicset_paths, results_path, dataset_path, noise_path=None, k=30):
    '''

    :param results_paths: Dict of key = topicset label, value = topicset path
    :param dataset_path:
    :param noise_path:
    :param k:
    :return:
    '''
    freqs, cofreqs, noise_words = analyze_dataset(dataset_path, noise_path)
    for topicset_label in topicset_paths.keys():
        topicset_path = topicset_paths[topicset_label]
        topics = load_topics(topicset_path)
        topics = [x for x in topics if len(x) > 0]
        scores = compute_metrics(topics, freqs, cofreqs, noise_words, k)
        save_metric_results(results_path, topicset_label, scores)