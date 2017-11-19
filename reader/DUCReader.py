"""
Reader for DUC corpus
@author: Hardy
"""
import os
import pickle
import math
import string
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


class DUCReader:

    def __init__(self, docs_path: str, summaries_path: str, corpus_path: str,
                 overwrite_save=False, stop_word=False):

        stemmer = PorterStemmer()
        self.stop_word = stop_word
        self.docs_path = docs_path
        self.summaries_path = summaries_path
        self.corpus = dict()
        self.corpus_path = corpus_path
        self.transformer = TfidfTransformer(sublinear_tf=True)
        translator = str.maketrans('', '', string.punctuation)
        tokenizer = lambda text: [stemmer.stem(token.lower()) for token in tokenize.word_tokenize(text.translate(translator))]
        if self.stop_word:
            self.vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenizer, stop_words='english')
        else:
            self.vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenizer)
        self.overwrite_save = overwrite_save

    def read_dir(self) -> dict:
        """
        Read directory from docs_path and summaries_path
        :return: A corpus
        """
        def extract_text(path: str)-> str:
            """
            Extract the TEXT content from input document
            :param path: A path of input document
            :return: the TEXT's string
            """
            is_read = False
            result = ''
            for line in open(path):
                if line == '<TEXT>\n' or line == '</TEXT>\n':
                    is_read = not is_read
                    continue
                if is_read:
                    result += line
            return tokenize.sent_tokenize(result.replace('\n', ' '))

        def get_sents_and_tfidf(docs_dir: str):
            """
            Given a multi-document, get all the sentences as a set. Each multi-document is represented as tf-idf matrix
            :param docs_dir: the directory of the multi-document
            :return: a tuple of all the sentences and the tf idf matrix
            """
            all_sents = []
            for doc_name in os.listdir(docs_dir):
                sentences = extract_text(os.path.join(docs_dir, doc_name))
                sentences = [sentence + '\n' for sentence in sentences]
                all_sents.extend(sentences)
            return (all_sents, self.transformer.fit_transform(self.vectorizer.fit_transform(all_sents)))

        def get_summaries(docs_name: str):
            """
            Given a docs name, retrieve all the summaries
            :param docs_name: the name of the docs
            :return: all summaries
            """
            summaries = dict()
            for summary_file in os.listdir(self.summaries_path):
                if not summary_file.endswith('xml') and summary_file.startswith(docs_name[:-1].upper()):
                    summary = [line for line in open(os.path.join(self.summaries_path, summary_file))]
                    summaries[summary_file[-1]] = summary
            return summaries

        def get_cosine_sim_matrix(tf_idf):
            def fCosine(u, v):
                uData, vData = u.data, v.data
                denominator = math.sqrt(np.sum(uData ** 2) * np.sum(vData ** 2))
                uCol, vCol = u.indices, v.indices
                uI = uData[np.in1d(uCol, vCol)]
                vI = vData[np.in1d(vCol, uCol)]
                if denominator == 0:
                    return 0
                return np.dot(uI, vI) / denominator

            n = tf_idf.shape[0]
            arr = np.zeros((n,n), dtype=np.float64)

            for i in range(n):
                for j in range(n):
                    arr[i, j] = 1 if i == j else fCosine(tf_idf[i, :], tf_idf[j, :])
                print(i)
            return arr


        n = 0

        # Each document is grouped in a cluster
        for docs_name in os.listdir(self.docs_path):
            print('Read ' + docs_name + ': ' + str(n))
            n = n + 1
            sub_corpus = dict()
            docs = dict()
            sents, tf_idf = get_sents_and_tfidf(os.path.join(self.docs_path, docs_name))
            docs['sim_matrix'] = get_cosine_sim_matrix(tf_idf)
            docs['tf_idf'] = tf_idf
            docs['sents'] = sents
            kmeans = KMeans(init='k-means++', verbose=1, n_jobs=-1,
                            n_clusters=int(0.2 * docs['tf_idf'].shape[0]), random_state=1, n_init=10)
            kmeans.fit_transform(docs['tf_idf'])
            docs['cluster_idxs'] = kmeans.labels_
            precompute = dict()
            X = list(range(len(docs['sents'])))
            for j in X:
                precompute[j] = math.fsum([docs['sim_matrix'][i, j] for i in X])
            docs['precompute'] = precompute
            sub_corpus['docs'] = docs
            sub_corpus['summaries'] = get_summaries(docs_name)
            # Both sub_corpus are joined as one corpus
            self.corpus[docs_name] = sub_corpus
        self.save_data()
        return self.corpus

    def save_data(self):
        """
        Dumping the corpus into pickle file
        """
        if self.stop_word:
            output_file = open(os.path.join(self.corpus_path, 'corpus_sw.pickle'), 'wb')
        else:
            output_file = open(os.path.join(self.corpus_path, 'corpus.pickle'), 'wb')
        pickle.dump(self.corpus, output_file, -1)

    def load_data(self):
        """
        Loading the corpus from pickle file
        :return the corpus
        """
        if not self.is_file_exist() or self.overwrite_save:
            self.read_dir()
            self.save_data()
            return self.corpus
        if self.stop_word:
            infile = open(os.path.join(self.corpus_path, 'corpus_sw.pickle'), 'rb')
        else:
            infile = open(os.path.join(self.corpus_path, 'corpus.pickle'), 'rb')
        self.corpus = pickle.load(infile)
        return self.corpus

    def is_file_exist(self):
        """
        Checking whether the corpus is exist or not
        :return: boolean of file existence
        """
        return os.path.isfile(os.path.join(self.corpus_path, 'corpus.pickle'))

    def write_sent_to_text_file(self):
        infile = open(os.path.join(self.corpus_path, 'sent.txt'), 'w')
        for corpora_name, corpora in self.corpus.items():
            docs = corpora['docs']
            for sent in docs['sents']:
                infile.write(sent)
        infile.close()

if __name__ == '__main__':
    duc_reader = DUCReader('/home/acp16hh/Projects/Research/Exp_8_Multi_Docs_Submodular_AMR/dataset/input/docs',
                           '/home/acp16hh/Projects/Research/Exp_8_Multi_Docs_Submodular_AMR/dataset/input/summaries',
                           '/home/acp16hh/Projects/Research/Exp_8_Multi_Docs_Submodular_AMR/dataset/output',
                           overwrite_save=False)
    corpus = duc_reader.load_data()
    duc_reader.write_sent_to_text_file()
    print()
