import argparse
import os
import numpy as np
import datetime
from time import sleep


def gen_header():
    header = '#!/usr/bin/env bash\n' \
             + '#$ -l rmem=1G\n' \
             + '#$ -l h_rt=4:00:00\n' \
             + '#$ -M hhardy2@sheffield.ac.uk\n' \
             + '#$ -m easb\n' \
             + '#$ -a ' + (n + datetime.timedelta(hours=delta_a)).strftime('%m%d%H%M') + '\n' \
             + 'module load apps/python/anaconda3-4.2.0\n' \
             + 'source activate nlp\n' \
             + 'export PATH=/usr/bin/perl:$PATH\n'
    return header


if __name__ == '__main__':
    n = datetime.datetime.now()
    summaries = '/home/acp16hh/Projects/Research/Exp_8_Multi_Docs_Submodular_AMR/dataset/input/summaries/'
    docs = '/home/acp16hh/Projects/Research/Exp_8_Multi_Docs_Submodular_AMR/dataset/input/docs/'
    corpus = '/home/acp16hh/Projects/Research/Exp_8_Multi_Docs_Submodular_AMR/dataset/output'
    k = 0.2
    a = 6.0
    ld = 0.15
    L = True
    R = True
    A = False
    r = 0.1
    b = 665
    L = False
    if not os.path.exists(os.path.join(corpus, 'scripts')):
        os.makedirs(os.path.join(corpus, 'scripts'))

    count = 0
    delta_a = 0
    for a in np.arange(7.0, 8.0, 1.0):
        for ld in np.arange(0.15, 0.2, 0.05):
            for r in np.arange(0.15, 0.2, 0.05):
                for k in np.arange(0.1, 0.2, 0.1):
                    body = 'python /home/acp16hh/Projects/Research/Exp_8_Multi_Docs_Submodular_AMR/src/main.py --summaries ' + summaries + ' --docs ' + docs + ' --corpus ' + corpus \
                        + ' --k ' + str(k) \
                        + ' --a ' + str(a) \
                        + ' --ld ' + str(ld) \
                        + ' --R ' \
                        + ' --r ' + str(r) \
                        + ' --b ' + str(b)
                    file_path = os.path.join(os.path.join(corpus, 'scripts'), 'script_' + str(count))
                    file = open(file_path, 'w')
                    count += 1
                    file.write(gen_header() + body)
                    file.close()
                    os.system('qsub ' + file_path)
                    if count % 10 == 0:
                        sleep(1)
                    if count % 50 == 0:
                        delta_a += 1

    # for a in np.arange(7.0, 8.0, 1.0):
    #     for ld in np.arange(0.15, 0.20, 0.05):
    #         for r in np.arange(0.15, 0.20, 0.05):
    #             for k in np.arange(0.1, 0.2, 0.1):
    #                 body = 'python /home/acp16hh/Projects/Research/Exp_8_Multi_Docs_Submodular_AMR/src/main.py --summaries ' + summaries + ' --docs ' + docs + ' --corpus ' + corpus \
    #                     + ' --k ' + str(k) \
    #                     + ' --a ' + str(a) \
    #                     + ' --ld ' + str(ld) \
    #                     + ' --L --A ' \
    #                     + ' --r ' + str(r) \
    #                     + ' --b ' + str(b)
    #                 file_path = os.path.join(os.path.join(corpus, 'scripts'), 'script_' + str(count))
    #                 file = open(file_path, 'w')
    #                 count += 1
    #                 file.write(gen_header() + body)
    #                 file.close()
    #                 os.system('qsub ' + file_path)
    #                 if count % 10 == 0:
    #                     sleep(1)
    #                 if count % 50 == 0:
    #                     delta_a += 1

