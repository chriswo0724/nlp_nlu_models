from evaluation import evaluateRandomly
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
import time
import torch
from torch import optim
from prepareTrainingData import tensorFromPair
import random
from dataprocesser import prepareDate
from torch import nn


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

from train_one_iter import train
from helper import timeSince

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_data, output_data, pairs = prepareDate('eng', 'fra', False)
print(random.choice(pairs))

hidden_size = 256
encoder = EncoderRNN(input_data.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_data.n_words, dropout_p=0.1).to(device)

def showPlot(points,name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("./seq2seqwithattention_model/%s.png" % name)

def trainingprocess(encoder, decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0


    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

   
    training_pairs = [tensorFromPair(input_data, output_data, random.choice(pairs)) \
        for i in range(n_iters)]

       
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter -1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        

        print_loss_total += loss
        plot_loss_total += loss


        if iter % print_every == 0:
           print_loss_avg = print_loss_total / print_every
           print_loss_total = 0
           

           print('%s (%d %d%%) %.4f' % (timeSince(start, iter/n_iters),iter, \
               iter/n_iters * 100, print_loss_avg)) 

          
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        

    showPlot(plot_losses,'object_function_NLLLoss')

    return encoder, decoder, plot_losses


if __name__ == '__main__':

    #training step
    encoder, decoder, plot_losses = trainingprocess(encoder, decoder, 75000, print_every = 5000)
    
    #save model
    torch.save(encoder, './seq2seqwithattention_model/model/encodermodel.pkl')
    torch.save(decoder, './seq2seqwithattention_model/model/attndecodermodel.pkl')

    #evaluate model and plot attention_matrix
    evaluateRandomly(input_data, output_data, pairs, encoder, decoder, n=10)
