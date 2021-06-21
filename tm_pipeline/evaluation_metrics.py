import math
from statistics import mean


def split_ngrams(topics):
    split_topics = []
    for topic in topics:
        split_topic = []
        for x in topic:
            l = x.split('$')
            l.append(x)
            split_topic.extend(l)
        split_topic = list(set(split_topic))
        split_topics.append(split_topic)
    return split_topics


def words_in_topic(test, gt, topn=10):
    words = 0
    for word in test[:topn]:
        if word in gt:
            words += 1
    return words


def match_topic(topic, gt_topics, gt_names, topn=10, delimiter=None):
    matched_topics = []
    for i in range(0, len(gt_topics)):
        gt_topic = gt_topics[i]
        gt_name = gt_names[i]
        wint = words_in_topic(topic, gt_topic, topn)
        if wint > 0:
            matched_topics.append((gt_name, wint))
    sorted_topics = sorted(matched_topics, reverse=True, key=lambda x: x[1])
    if not delimiter:
        return sorted_topics
    else:
        return [delimiter.join((x[0], format(x[1], "1.3f"))) for x in sorted_topics]


def match_topicset(topics, gt_topics, gt_names, topn=10, delimiter=None):
    matched_topicset = []
    for topic in topics:
        matched_topicset.append(match_topic(topic, gt_topics, gt_names, topn, delimiter))
    return matched_topicset


def recall(ts, gt_topics):
    matched_topics = set()
    for topic in ts:
        for x in topic:
            matched_topics.add(x[0])
    matched_topics = list(matched_topics)
    return len(matched_topics)/len(gt_topics)


def mean_cof(topic, token, cofrequencies):
    if len(topic) < 2:
        return 0
    cof_count = 0
    for w in topic:
        if token != w:
            word_tup = tuple(sorted([token, w]))
            if word_tup in cofrequencies:
                cof_count += cofrequencies[word_tup]
    if token in topic:
        return cof_count / (len(topic) - 1)
    return cof_count / len(topic)


def silhouette(T, topic, token, cofrequencies):
    '''
    Maximizing mean cofrequency instead of minimizing distance.  Silhouette value of a given token from the given topic
    :param T: topic set
    :param topic: home topic of queried token
    :param token: queried token we wish to get silhouette value for
    :param cofrequencies: dictionary of cofrequencies in data set
    :return:
    '''
    a = mean_cof(topic, token, cofrequencies)
    b = 0
    for i in range(0, len(T)):
        t_i = T[i]
        if t_i != topic:
            topic_score = mean_cof(t_i, token, cofrequencies)
            if topic_score > b:
                b = topic_score
    if a == b:
        return 0
    return (a - b) / max(a, b)


def topic_silhouette(T, topic, cofrequencies):
    silhouettes = []
    for w in topic:
        silhouettes.append(silhouette(T, topic, w, cofrequencies))
    return silhouettes


def topicset_silhouettes(T, cofrequencies):
    silhouettes = []
    for topic in T:
        s = topic_silhouette(T, topic, cofrequencies)
        silhouettes.append(s)
    return silhouettes


def npmi(topic, frequencies, cofrequencies):
    v = 0
    x = max(2, len(topic))
    for i in range(0, len(topic)):
        w_i = topic[i]
        p_i = 0
        if w_i in frequencies:
            p_i = frequencies[w_i]
        for j in range(i+1, len(topic)):
            w_j = topic[j]
            p_j = 0
            if w_j in frequencies:
                p_j = frequencies[w_j]
            word_tup = tuple(sorted([w_i, w_j]))
            p_ij = 0
            if word_tup in cofrequencies:
                p_ij = cofrequencies[word_tup]
            if p_ij < 2:
                v -= 1
            else:
                pmi = math.log(p_ij / (p_i * p_j), 2)
                denominator = -1 * math.log(p_ij, 2)
                v += (pmi / denominator)
    return (2*v) / (x*(x-1))


def topic_npmis(T, frequencies, cofrequencies, k=20):
    npmis = []
    for topic in T:
        n = npmi(topic[:k], frequencies, cofrequencies)
        npmis.append(n)
    return npmis


def topic_coherence(T, frequencies, cofrequencies, k=20):
    '''
    Computes the coherence of a topic set (average NPMI of topics)
    :param T:
    :param frequencies:
    :param cofrequencies:
    :param k: top-k words per topic to consider
    :return:
    '''
    npmis = topic_npmis(T, frequencies, cofrequencies, k)
    if len(npmis) > 0:
        return mean(npmis)
    return 0


def topic_word_overlap(T, k=20, q=1):
    '''
    fraction of top-k words in topic vocab that appear in more than q topics
    :param T: topic set
    :param k:
    :return:
    '''
    topics_per_word = dict()
    for topic in T:
        for w in topic[:k]:
            if not w in topics_per_word:
                topics_per_word[w] = 0
            topics_per_word[w] += 1

    len_V = len(topics_per_word.keys())
    overlapping_words = 0
    for w in (topics_per_word.keys()):
        if topics_per_word[w] > q:
            overlapping_words += 1

    return overlapping_words / len_V


def overlap_one(T):
    return topic_word_overlap(T, k=1)


def overlap_frac(T, frac=1):
    return topic_word_overlap(T, k=int(len(T)*frac))


def topic_coverage(all_topics, T_labels):
    '''
    fraction of labelled topics recovered by a topic set
    :param T:
    :param labels: hand labelled topic labels
    :return:
    '''
    unique_topic_labels = []
    [unique_topic_labels.extend(x) for x in T_labels]
    unique_topic_labels = list(set(unique_topic_labels))
    return len(unique_topic_labels) / len(all_topics)


def concept_topic_overlap(T_labels, k=1):
    '''
    fraction of topics that are recovered by more than k approximated topics
    :param T:
    :param T_labels:
    :param k:
    :return:
    '''
    topics_per_approx = dict()
    for labels in T_labels:
        for topic in labels:
            if not topic in topics_per_approx:
                topics_per_approx[topic] = 0
            topics_per_approx[topic] += 1

    len_V = len(topics_per_approx.keys())
    overlapping_words = 0
    for topic in (topics_per_approx.keys()):
        if topics_per_approx[topic] > k:
            overlapping_words += 1

    return overlapping_words / len_V


def topic_relevance(T, T_relevant_words):
    '''
    average fraction of words per topic that are relevant
    :param T:
    :param T_relevant_words: hand labelled words
    :return:
    '''
    relevance = []
    for i in range(0, len(T)):
        topic = T[i]
        relevant_words = T_relevant_words[i]
        relevance.append(len(relevant_words) / len(topic))
    return sum(relevance)/len(relevance)


def topic_diversity(T, k):
    '''
    fraction of words in top-k words of each topic that are unique
    :param T:
    :param k: top k words per topic
    :return:
    '''
    top_words = []
    for topic in T:
        top_words.extend(topic[:k])
    unique_words = set(top_words)
    if len(top_words) > 0:
        return len(unique_words)/len(top_words)
    return 0


def noise_penetration(T, noise_words, k):
    '''
    fraction of words in top-k words of each topic that are noise words
    :param T:
    :param noise_words: set of noise words
    :param k: top-k words of each topic
    :return:
    '''
    noise_count = 0
    for topic in T:
        for w in topic[:k]:
            if w in noise_words:
                noise_count += 1
    if len(T) > 0:
        return noise_count / (len(T) * k)
    return 1