"""
Calculating ROUGE
@author: Hardy
"""
import os
from pyrouge.Rouge155 import Rouge155


def calc_ROUGE(model_dir, summ_dir, corpus_path, params_str, params):
    rouge_dir = '/home/acp16hh/Projects/perl/rouge'
    rouge_args = \
        '-e /home/acp16hh/Projects/perl/rouge/data -v -d -x -a -c 95 -b 665 -m -n 1'
    rouge = Rouge155(rouge_dir, rouge_args)
    rouge.model_dir = model_dir
    rouge.model_filename_pattern = 'D#ID#.M.100.T.[A-Z]$'

    # system results
    rouge.summ_dir = summ_dir
    rouge.system_dir = summ_dir
    rouge.system_filename_pattern = 'D([0-9]{5}).M.100.T.[A-Z]'
    rouge_output = rouge.convert_and_evaluate(split_sentences=True)
    print(rouge_output)
    output_dict = rouge.output_to_dict(rouge_output)

    result = '%.1f%%\t%.1f%%\t%.1f%%' % (
    output_dict['rouge_1_precision'] * 100, output_dict['rouge_1_recall'] * 100, output_dict['rouge_1_f_score'] * 100)
    result_path = os.path.join(corpus_path, 'result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_path = os.path.join(result_path, 'rouge_result' + params_str + '.tsv')
    text_file = open(file_path, 'w')
    text_file.write(result)
    text_file.close()
    output_path = os.path.join(corpus_path, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = os.path.join(output_path, 'rouge_output' + params_str + '.txt')
    text_file = open(file_path, 'w')
    text_file.write(rouge_output)
    text_file.close()



