import torch
from prepareTrainingData import tensorFromSentence
from dataprocesser import MAX_LENGTH
from DataStructure import EOS_token, SOS_token
from encoder import EncoderRNN, device
from decoder import DecoderRNN
from dataprocesser import prepareDate
import random

input_data, output_data, pairs = prepareDate('eng', 'fra', False)

def evaluate(input_data, output_data, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_data, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_data.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly(input_data, output_data, pairs, encoder, decoder, n=100):
    for i in range(n):
        pair = random.choice(pairs)
        print('')
        '-' * 20 + 'sample ' + str(i) + '-' * 20
        print('lang 1 >', pair[0])
        print('standard lang 2 =', pair[1])
        output_words = evaluate(input_data, output_data, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('predict lang 2<', output_sentence)
        print('')

if __name__ == '__main__':
  
    encoder = torch.load('NLU_model/seq2seq_model/model/encodermodel.pkl')
    decoder = torch.load('NLU_model/seq2seq_model/model/decodermodel.pkl')
    evaluateRandomly(input_data, output_data, pairs, encoder,decoder)