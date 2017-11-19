"""
Reimplementation of:
'A Class of Submodular Functions for Document Summarization' paper by Hui Lin and Jeff Blimes
@author: Hardy
"""
import os
import math
import pickle
from reader.DUCReader import DUCReader
from eval.calc_rouge_n import calc_ROUGE
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer



def _clustering(tf_idf, n):
    kmeans = KMeans(init='k-means++', verbose=1, n_clusters=int(n*tf_idf['tf_idf'].shape[0]), random_state=1, n_init=10)
    # svd = TruncatedSVD(30, random_state=1)
    # normalizer = Normalizer(copy=False)
    # lsa = make_pipeline(svd, normalizer)
    # X = lsa.fit_transform(tf_idf['tf_idf'])
    # kmeans.fit_transform(X)
    kmeans.fit_transform(tf_idf['tf_idf'])
    cluster_idxs = kmeans.labels_
    return cluster_idxs

def _arg_max_greedy(docs, params):

    def __coverage(S):
        if params['a'] == 1:
            return math.fsum(
                [math.fsum([docs['sim_matrix'][i, j] for i in S]) for j in X]
            )
        return math.fsum(
            [
                min(
                    (
                        math.fsum([docs['sim_matrix'][i, j] for j in S]),
                        params['a'] / len(docs['precompute']) * docs['precompute'][i]
                    )
                ) for i in X

                # min((
                #     math.fsum([docs['sim_matrix'][i, j] for i in X]),
                #     params['a']/len(docs['precompute']) * docs['precompute'][j]))
                # for j in S
            ]
        )

    def __cost(S):
        return sum([len(sents[s]) for s in S])

    def __diversity(S):
        return math.fsum(
            [
                math.sqrt(math.fsum(
                    [1/len(X) * docs['precompute'][j] for j in S if docs['cluster_idxs'][j] == kc]
                )
            ) for kc in range(int(params['k'] * docs['tf_idf'].shape[0]))]
        )

    def __F(S):
        result = 0.0
        if len(S) == 0:
            return result
        if params['L']:
            result += params['ld'] * __coverage(S)
        if params['R']:
            result += (1 - params['ld']) * __diversity(S)
        return result

    sents = docs['sents']
    X = list(range(len(docs['sents'])))
    G = []
    U = X[:]

    while len(U) > 0:
        max_k = float('-inf')
        k = None
        idx = None
        for l in range(len(U)):
            temp = (__F(G + [U[l]]) - __F(G)) / math.pow(__cost([U[l]]), params['r'])
            if temp >= max_k:
                max_k = temp
                k = U[l]
                idx = l
        if __cost(G + [k]) <= params['b'] and max_k > 0:
            G = G + [k]
        del U[idx]
        smallest = float('inf')
        for l in range(len(U)):
            small = __cost([U[l]])
            if small < smallest:
                smallest = small
        if __cost(G) + smallest > params['b']:
            break
    max_v = float('-inf')
    v_star = None
    for v in range(len(X)):
        if __cost([X[v]]) <= params['b']:
            temp = __F([X[v]])
            if temp >= max_v:
                max_v = temp
                v_star = X[v]
    if __F([v_star]) > __F(G):
        return [v_star]
    else:
        return G


def init_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs', help='Gold standard documents.')
    parser.add_argument('--summaries', help='Gold standard summaries.')
    parser.add_argument('--corpus', help='Corpus')
    parser.add_argument('--k', help='Percentage of size as cluster', default=0.2, type=float)
    parser.add_argument('--a', help='Alpha', default=6.0, type=float)
    parser.add_argument('--ld', help='Lambda', default=0.15, type=float)
    parser.add_argument('--L', help='Enable coverage function', action='store_true')
    parser.add_argument('--R', help='Enable diversity function', action='store_true')
    parser.add_argument('--A', help='Set alpha=1', action='store_true')
    parser.add_argument('--r', help='Scaling factor', default=0.1, type=float)
    parser.add_argument('--b', help='Max length of document in byte', default=665, type=int)
    parser.add_argument('--stop_word', help='Enable stop words', action='store_true')
    args = parser.parse_args()
    if not args.docs:
        raise Exception('No document is specified.')
    if not args.summaries:
        raise Exception('No summary is specified.')
    if not args.corpus:
        raise Exception('No corpus is specified.')
    return args


def summarize(corpus, params):
    all_summary = {}
    count = 1
    for corpora_name, corpora in corpus.items():
        print(count)
        docs = corpora['docs']
        if params['k'] != 0.2:
            docs['cluster_idxs'] = _clustering(corpora['docs'], params['k'])
        S = _arg_max_greedy(docs, params)
        summaries = []
        for i in S:
            summaries.append(docs['sents'][i])
        all_summary[corpora_name] = summaries
        count += 1
    return all_summary


def save_summary(all_summary, params_str):
    save_path = os.path.join(args.corpus, 'summaries')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    output_file = open(os.path.join(save_path, 'summaries' + params_str + '.pickle'), 'wb')
    pickle.dump(all_summary, output_file, -1)


def load_summary(params_str):
    infile = open(os.path.join(os.path.join(args.corpus, 'summaries'), 'summaries' + params_str + '.pickle'), 'rb')
    return pickle.load(infile)


def eval(all_summary, params_str, params):

    folder = 'summaries' + params_str
    folder_path = os.path.join(args.corpus, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for summary_name, summary in all_summary.items():
        file_path = os.path.join(folder_path, summary_name[:-1].upper()+'.M.100.T.S')
        text_file = open(file_path, 'w')
        text_file.write(''.join(summary))
        text_file.close()
    calc_ROUGE(model_dir=args.summaries, summ_dir=folder_path,
               corpus_path=args.corpus, params_str=params_str, params=params)


def main():
    params = dict()
    params['ld'] = args.ld
    params['a'] = args.a if not args.A else 1
    params['R'] = args.R
    params['L'] = args.L
    params['k'] = args.k
    params['b'] = args.b
    params['r'] = args.r
    params['stop'] = args.stop_word
    duc_reader = DUCReader(args.docs, args.summaries, args.corpus, overwrite_save=False, stop_word=params['stop'])
    params_str = '_ld_' + str(params['ld']) + \
                 '_a_' + str(params['a']) + \
                 '_R_' + str(params['R']) + \
                 '_L_' + str(params['L']) + \
                 '_k_' + str(params['k']) + \
                 '_b_' + str(params['b']) + \
                 '_r_' + str(params['r']) + \
                 '_stop_' + str(params['stop'])
    corpus = duc_reader.load_data()
    # all_summary = summarize(corpus, params)
    # save_summary(all_summary, params_str)
    all_summary = load_summary(params_str)
    eval(all_summary, params_str, params)
    print()


if __name__ == '__main__':
    args = init_args()
    main()

