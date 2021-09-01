from tm_pipeline.evaluate_topic_set import compute_all_results


def get_param_string(param):
    return '_'.join([str(x).replace('.', '-') for x in param])


def main():
    path = 'results/{}/{}/'

    dataset_names = ['sample_tweets']
    models = ['tnd', 'nlda', 'etnd', 'enlda']
    model_params = {
        "tnd": [
            # k, alpha, beta, skew, noise_words_max, iterations
            # (30, 50, 0.01, 25, 200, 1000),
            (30, 50, 0.01, 25, 200, 1000),
            # (30, 50, 0.01, 25, 200, 1000),
        ],
        "nlda": [
            # lda_k, nft_k, alpha, beta, skew, noise_words_max, iterations, topic_weights tuple
            # (((30,),), (30, 50, 0.01, 25, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
            (((30,),), (30, 50, 0.01, 25, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
            # (((30,),), (30, 50, 0.01, 25, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
        ],
        "etnd": [
            # k, alpha, beta, skew, noise_words_max, embedding_path, closest_x_words, tau (number of iterations before embedding activation), iterations
            # (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 3, 200, 1000),
            (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 5, 200, 1000),
            # (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 10, 200, 1000),
        ],
        "enlda": [
            # lda_k, nft_k, alpha, beta, skew, noise_words_max, embedding_path, closest_x_words, tau (number of iterations before embedding activation), iterations, topic_weights tuple
            # (((30,),), (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 3, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
            (((30,),), (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 5, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
            # (((30,),), (30, 50, 0.01, 25, 200, 'local_{}_ft.bin', 10, 200, 1000), (1, 5, 10, 15, 20, 25, 30)),
        ],
    }


    for dataset_name in dataset_names:
        print(dataset_name)
        topicset_noise_paths = {}
        results_path = 'results/{}_metrics.csv'.format(dataset_name)
        dataset_path = 'data/{}.csv'.format(dataset_name)

        for model in models:
            ds_model_path = path.format(dataset_name, model)
            model_p = model
            params = model_params[model_p]
            for param_set in params:
                if model == 'tnd':
                    nft_k = param_set[0]
                    skew = param_set[3]
                    noise_path = ds_model_path + 'noise_{}_{}_{}.csv'.format(nft_k, skew, 0)
                    topic_path = ds_model_path + 'topics_{}_{}_{}.csv'.format(nft_k, skew, 0)
                    label = 'tnd {} {} {}'.format(nft_k, skew, 0)
                    topicset_noise_paths[label] = (topic_path, noise_path, nft_k)
                elif model == 'etnd':
                    nft_k = param_set[0]
                    skew = param_set[3]
                    closest_x_words = param_set[6]
                    noise_path = ds_model_path + 'noise_{}_{}_{}.csv'.format(nft_k, skew, closest_x_words)
                    topic_path = ds_model_path + 'topics_{}_{}_{}.csv'.format(nft_k, skew, closest_x_words)
                    label = 'etnd {} {} {}'.format(nft_k, skew, closest_x_words)
                    topicset_noise_paths[label] = (topic_path, noise_path, nft_k)
                elif model == 'enlda':
                    lda_param_set = param_set[0]
                    nft_param_set = param_set[1]
                    topic_weights = param_set[2]
                    nft_k = nft_param_set[0]
                    skew = nft_param_set[3]
                    closest_x_words = nft_param_set[6]
                    noise_path = path.format(dataset_name, 'etnd') + 'noise_{}_{}_{}.csv'.format(nft_k, skew, closest_x_words)
                    for i in range(0, len(topic_weights)):
                        for lda_param in lda_param_set:
                            nlda_path = ds_model_path + 'topics_{}_{}_{}.csv'.format(lda_param[0], topic_weights[i], closest_x_words)
                            label = 'enlda {} {} {}'.format(lda_param[0], topic_weights[i], closest_x_words)
                            topicset_noise_paths[label] = (nlda_path, noise_path, lda_param[0])
                elif model == 'nlda':
                    lda_param_set = param_set[0]
                    nft_param_set = param_set[1]
                    topic_weights = param_set[2]
                    nft_k = nft_param_set[0]
                    skew = nft_param_set[3]
                    noise_path = path.format(dataset_name, 'tnd') + 'noise_{}_{}_{}.csv'.format(nft_k, skew, 0)
                    for i in range(0, len(topic_weights)):
                        for lda_param in lda_param_set:
                            nlda_path = ds_model_path + 'topics_{}_{}_{}.csv'.format(lda_param[0], topic_weights[i], 0)
                            label = 'nlda {} {} {}'.format(lda_param[0], topic_weights[i], 0)
                            topicset_noise_paths[label] = (nlda_path, noise_path, lda_param[0])


        compute_all_results(topicset_noise_paths, results_path, dataset_path)

if __name__ == '__main__':
    main()