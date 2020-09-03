from __future__ import division, print_function, unicode_literals
from io import open
import unicodedata
import string
import os
import glob

class preProcess:
    def __init__(self, path):
        self.path = path
        self.all_letters = string.ascii_letters + " .,;'-"

    def findFiles(self):
        return glob.glob(self.path)

    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def readWords(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [self.unicode_to_ascii(line.strip()) for line in f.readlines()]


    def process(self):
        category_words = {}
        all_categories = []
        for filename in self.findFiles():
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            words = self.readWords(filename)
            category_words[category] = words

        if len(all_categories) == 0:
            raise RuntimeError("""Data not found. Make sure that you downloaded data from 
            https://download.pytorch.org/tutorial/data.zip and extract it to the current directory.""")
        #for the convenience of testing
        #print('# categories:', len(all_categories), all_categories)
        return category_words, all_categories

if __name__=='__main__':
    path = r'D:\\Pycharm\\workspcae\\NLP-playing\\data_2\\names\\*.txt'
    testing = preProcess(path)
    category_words, all_categories = testing.process()
    print(category_words['Chinese'])