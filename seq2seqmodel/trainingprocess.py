from evaluation import evaluateRandomly
from encoder import EncoderRNN
from decoder import DecoderRNN
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
decoder = DecoderRNN(hidden_size, output_data.n_words).to(device)

encoder1 = EncoderRNN(input_data.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_data.n_words).to(device)

def showPlot(points,name):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("NLU_model/seq2seq_model/%s.png" % name)

def trainingprocess(encoder, decoder, n_iters, print_every = 1000, plot_every = 100, learning_rate = 0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    plot_losses1 = []
    print_loss_total1 = 0
    plot_loss_total1 = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

    encoder_optimizer1 = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer1 = optim.SGD(decoder.parameters(), lr = learning_rate)

    #training_pairs = [random.choice(pairs) for i in range(n_iters)]

    #for i in range(n_iters):
    #    print(random.choice(pairs))
   
    training_pairs = [tensorFromPair(input_data, output_data, random.choice(pairs)) \
        for i in range(n_iters)]

    
    
    criterion = nn.NLLLoss()
    criterion1 = nn.PoissonNLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter -1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        loss1 = train(input_tensor, target_tensor, encoder1, decoder1, encoder_optimizer1, decoder_optimizer1, criterion1)


        print_loss_avg = print_loss_total / print_every
        print_loss_total += loss
        plot_loss_total += loss

        print_loss_avg1 = print_loss_total1 / print_every
        print_loss_total1 += loss1
        plot_loss_total1 += loss1

        if iter % print_every == 0:
           print_loss_avg = print_loss_total / print_every
           print_loss_total = 0
           
           print_loss_avg1 = print_loss_total1 / print_every
           print_loss_total1 = 0

           #print('%s (%d %d%%) %.4f' % (timeSince(start, iter/n_iters),iter, \
           #    iter/n_iters * 100, print_loss_avg)) 

           print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, iter/n_iters),iter, \
               iter/n_iters * 100, print_loss_avg, print_loss_avg1)) 

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            plot_loss_avg1 = plot_loss_total1 / plot_every
            plot_losses1.append(plot_loss_avg1)
            plot_loss_total1 = 0            

    showPlot(plot_losses,'object_function_NLLLoss')
    showPlot(plot_losses1,'object_function_BCEWithLogitsLoss')

    return encoder, decoder, plot_losses, plot_losses1


if __name__ == '__main__':
  
    encoder, decoder, plot_losses, plot_losses1 = trainingprocess(encoder, decoder, 75000, print_every = 5000)
    torch.save(encoder, 'NLU_model/seq2seq_model/model/encodermodel.pkl')
    torch.save(decoder, 'NLU_model/seq2seq_model/model/decodermodel.pkl')