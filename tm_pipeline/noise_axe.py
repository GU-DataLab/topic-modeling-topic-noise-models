from settings.common import load_topics, save_topics, load_noise_words


def chop_noise(topics, noise_list, c):
    '''
    Remove top-c words from noise_list from the topics
    :param topics: list of topics
    :param noise_list: list of all noise words ranked by frequency
    :param c: top-c words to consider noise from noise-list
    :return:
    '''

    final_topics = []
    for topic in topics:
        final_topic = []
        for w in topic:
            if not w in noise_list[:c]:
                final_topic.append(w)
        final_topics.append(final_topic)
    return final_topics


def load_chop_save_topics(topics_path, noise_path, chopped_topics_path=None, c=100):
    if chopped_topics_path is None:
        chopped_topics_path = topics_path
    topics = load_topics(topics_path)
    noise_list = load_noise_words(noise_path)
    chopped_topics = chop_noise(topics, noise_list, c)
    save_topics(chopped_topics, chopped_topics_path)
    return chopped_topics
