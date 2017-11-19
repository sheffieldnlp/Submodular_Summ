"""
Read propbank frameset
@author Hardy
"""
import os
from os import listdir
from xml.dom import minidom
import pickle


class PropBankReader:
    def __init__(self, path, output_path):
        self.propbank = {}
        self.path = path
        self.output_path = os.path.join(output_path, 'data')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def build_data(self):
        """
        Reading the propbank from the propbank folder that comes with LDC2017T10.
        """
        for file in listdir(self.path):
            if file == 'frameset.dtd':
                continue
            single_file = minidom.parse(os.path.join(self.path, file))
            for roleset in single_file.getElementsByTagName('roleset'):
                roleset_id = roleset.attributes['id'].value
                self.propbank[roleset_id] = roleset
        return self.propbank

    def is_file_exist(self):
        """
        Checking whether the file exist or not
        """
        return os.path.isfile(os.path.join(self.output_path, 'propbank.pickle'))

    def save_data(self):
        """
        Dumping the propbank to pickle file
        """
        output_file = open(os.path.join(self.output_path, 'propbank.pickle'), 'wb')
        pickle.dump(self.propbank, output_file, -1)

    def load_data(self):
        """
        Loading the propbank from pickle file
        """
        infile = open(os.path.join(self.output_path, 'propbank.pickle'), 'rb')
        self.propbank = pickle.load(infile)
        return self.propbank


if __name__ == '__main__':
    propbank_reader = PropBankReader('/home/acp16hh/Projects/Research/Exp_7_Improve_CMap/dataset/Input/propbank-frames/frames', '/home/acp16hh/Projects/Research/Exp_7_Improve_CMap/dataset/output')
    propbank_reader.build_data()
    propbank_reader.save_data()
