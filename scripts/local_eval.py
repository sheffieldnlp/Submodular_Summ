import numpy as np
import os
if __name__ == '__main__':
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
    for a in np.arange(7.0, 8.0, 1.0):
        for ld in np.arange(0.15, 0.20, 0.05):
            for r in np.arange(0.15, 0.20, 0.05):
                for k in np.arange(0.1, 0.2, 0.1):
                    body = 'python /home/acp16hh/Projects/Research/Exp_8_Multi_Docs_Submodular_AMR/src/main.py --summaries ' + summaries + ' --docs ' + docs + ' --corpus ' + corpus \
                           + ' --k ' + str(k) \
                           + ' --a ' + str(a) \
                           + ' --ld ' + str(ld) \
                           + ' --R ' \
                           + ' --r ' + str(r) \
                           + ' --b ' + str(b)
                    os.system(body)
