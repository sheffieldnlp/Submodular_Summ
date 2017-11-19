import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='Result directory')
    arg = parser.parse_args()
    result = {}
    for file in os.listdir(arg.output):
        if file == 'merge_result.tsv':
            continue
        infile = open(os.path.join(arg.output, file))
        line = infile.readline()
        result[file] = line
    result_file = open(os.path.join(arg.output, 'merge_result.tsv'), 'w')
    result_file.write('Name\tP\tR\tF1\n')
    for name, line in result.items():
        result_file.write(name + '\t' + line + '\n')

