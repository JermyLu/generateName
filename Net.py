import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        #define that the dimensions of input tensor and output tensor in the every layer
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

if __name__=='__main__':
    from preprocess import preProcess
    from utils import Utils
    path = r'D:\\Pycharm\\workspcae\\NLP-playing\\data_2\\names\\*.txt'
    pp = preProcess(path)
    category_words, all_categories = pp.process()
    utils = Utils(category_words, all_categories)
    category_tensor, word_tensor, target_tensor = utils.randomTrainExample()























