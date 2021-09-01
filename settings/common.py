def save_flat_list(lizt, file):
    with open(file, 'w') as f:
        f.write('\n'.join(lizt))


def load_flat_dataset(path, delimiter=' '):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(delimiter))
    return dataset


def get_vocabulary(docs):
    '''
    This version of get_vocabulary takes 0.08 seconds on 100,000 documents whereas the old version took forever.
    '''
    vocab = []
    for i in range(0, len(docs)):
        vocab.extend(docs[i])
    return list(set(vocab))


def word_frequency(frequency, docs):
    '''
    :param frequency: passed explicitly so that you can increment existing frequencies if using in online mode
    :param docs:
    :return: updated frequency

    '''
    for doc in docs:
        for word in doc:
            if word in frequency:
                frequency[word] += 1
            else:
                frequency[word] = 1
    return frequency


def word_co_frequency(frequency, docs):
    for doc in docs:
        for i in range(0, len(doc) - 1):
            w1 = doc[i]
            for j in range(i + 1, len(doc)):
                w2 = doc[j]
                word_list = sorted([w1, w2])
                word_tup = tuple(word_list)
                if not word_tup in frequency:
                    frequency[word_tup] = 0
                frequency[word_tup] += 1
    return frequency


def word_tf_df(frequency, docs):
    '''
    :param frequency: passed explicitly so that you can increment existing frequencies if using in online mode
    :param docs:
    :return: updated frequency freq[0] = df, freq[1] = tf

    '''
    for doc in docs:
        doc_word = []
        for word in doc:
            if word not in frequency:
                frequency[word] = [0, 0]
            frequency[word][1] += 1
            if word not in doc_word:
                frequency[word][0] += 1
                doc_word.append(word)
    return frequency


def normalize_frequencies(frequencies, k):
    nf = {}
    for key in frequencies.keys():
        nf[key] = frequencies[key] / k
    return nf


def save_topics(topics, path):
    with open(path, 'w') as f:
        for topic in topics:
            f.write('{}\n'.format(','.join(topic)))


def save_noise_dist(noise, path):
    with open(path, 'w') as f:
        for word, freq in noise:
            f.write('{},{}\n'.format(word, freq))


def load_topics(path):
    topics = []
    with open(path, 'r') as f:
        for line in f:
            words = line.strip().split(',')
            for i in range(0, len(words)):
                words[i] = words[i].strip().replace(' ', '$')
            words = [w for w in words if len(w) > 0]
            topics.append(words)
    return topics


def load_noise_words(path):
    noise_words = []
    with open(path, 'r') as f:
        for line in f:
            word = line.strip()
            noise_words.append(word)
    return noise_words
