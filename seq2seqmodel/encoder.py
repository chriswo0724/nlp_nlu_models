import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#一个最基础的encoderRNN 每个神经元使用GRU

'''
input is words for example: I like you !
vocab:
0: I
1: like
2: you
3: !


words list put into embedding module: 
input:one-hot    embedding   output:new_embedding
1 0 0 0                
0 1 0 0
0 0 1 0
0 0 0 1

each word generates the vector of it through embedding layer
for each vector of word:
    input: vector of current word + previous word hidden
    through GRU
    output: output + current word hidden
'''
class EncoderRNN(nn.Module):
    #初始化yigeencoderRNN 隐藏状态的维度
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #初始化embedding 模块
        self.embedding = nn.Embedding(input_size, hidden_size)
        #初始化GRU单元
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #one-hot vector of this word as input and output is new_embedding of this word
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        #GRU new_embedding and previous hidden as inpput 
        # and output and current hidden as output
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        #初始化Hidden vector and value are 0
        return torch.zeros(1, 1, self.hidden_size, device=device)