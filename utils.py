#in the program, 'print' is the for the convenience of debugging
import torch
import string
import random

class Utils:
    def __init__(self, category_words, all_categories):
        """
        :param category_words: it is the dict type {}
        :param all_categories: it is the list type []
        """
        self.category_words = category_words
        self.all_categories = all_categories
        self.all_letters = string.ascii_letters + " .,;'-"
        self.n_letters = len(self.all_letters) + 1  # Plus EOS marker
        #print(self.all_letters)

    def categoryTensor(self, category):
        idx = self.all_categories.index(category)
        tensor = torch.zeros(1, len(self.all_categories))
        tensor[0][idx] = 1
        return tensor

    def inputTensor(self, word):
        tensor = torch.zeros(len(word), 1, self.n_letters)
        for i in range(len(word)):
            letter = word[i]
            tensor[i][0][self.all_letters.find(letter)] = 1
        return tensor

    def targetTensor(self, word):
        letter_idxes = [self.all_letters.find(word[i]) for i in range(1, len(word))]
        letter_idxes.append(self.n_letters - 1)
        #return torch.Tensor(letter_idxes)
        return torch.LongTensor(letter_idxes)

    #random choose a item from a list
    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    #random choose a instance of (category, word)
    def randomTrainPair(self):
        category = self.randomChoice(self.all_categories)
        word = self.randomChoice(self.category_words[category])
        return category, word

    def randomTrainExample(self):
        category, word = self.randomTrainPair()
        #print(category)
        #print(word)
        category_tensor = self.categoryTensor(category)
        word_tensor = self.inputTensor(word)
        target_tensor = self.targetTensor(word)
        return category_tensor, word_tensor, target_tensor

if __name__=='__main__':
    from preprocess import preProcess
    path = r'D:\\Pycharm\\workspcae\\NLP-playing\\data_2\\names\\*.txt'
    testing = preProcess(path)
    category_words, all_categories = testing.process()
    #print(category_words['Chinese'])
    utils = Utils(category_words, all_categories)
    category_tensor, word_tensor, target_tensor = utils.randomTrainExample()
    print(category_tensor)
    print(word_tensor)
    print(target_tensor)