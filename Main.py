import sys
import string
import math
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from preprocess import preProcess
from Net import RNN
from utils import Utils


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class Train:
    def __init__(self, model):
        self.criterion = nn.NLLLoss()
        self.lr = 0.0005
        self.model = model#the instance of NN

    def train(self, category_tensor, word_tensor, target_tensor):
        target_tensor.unsqueeze_(-1)
        hidden = self.model.init_hidden()

        self.model.zero_grad()

        loss = 0
        for i in range(word_tensor.size(0)):
            output, hidden = self.model(category_tensor, word_tensor[i], hidden)
            loss += self.criterion(output, target_tensor[i])
        loss.backward()

        for p in self.model.parameters():
            p.data.add_(p.grad.data, alpha=-self.lr)

        return output, loss.item() / word_tensor.size(0)


if __name__=='__main__':

    path = r'D:\\Pycharm\\workspcae\\NLP-playing\\data_2\\names\\*.txt'
    pp = preProcess(path)
    category_words, all_categories = pp.process()
    # print(category_words['Chinese'])
    utils = Utils(category_words, all_categories)
    #category_tensor, word_tensor, target_tensor = utils.randomTrainExample()

    all_letters = string.ascii_letters + " .,;'-"
    input_size = len(all_letters) + 1
    hidden_size = 128
    model = RNN(len(all_categories), input_size, hidden_size, input_size)

    train = Train(model)

    n_iters = 100000
    print_every = 5000

    start = time.time()
    for iter in range(1, n_iters+1):
        output, loss = train.train(*utils.randomTrainExample())
        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))


    #for testing the generation of name
    def generate_name(max_length, category, start_letters):
        for start_letter in start_letters:
            with torch.no_grad():
                category_tensor = utils.categoryTensor(category)
                input = utils.inputTensor(start_letter)
                hidden = model.init_hidden()

                output_name = start_letter
                for i in range(max_length):
                    output, hidden = model(category_tensor, input[0], hidden)
                    topv, topi = output.topk(1)
                    topi = topi[0][0]
                    #print(topi.item())
                    if topi >= len(all_letters)-1:
                        break
                    else:
                        output_name += all_letters[topi]
                    input = utils.inputTensor(all_letters[topi])

                print(output_name)

    generate_name(10, 'Chinese', 'CHI')
    generate_name(20, 'German', 'GER')
    print('..............Finished....................')
    sys.exit()
