from DataStructure import DataStructurer, EOS_token
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indexesFromSentence(input_data, sentence):
    return [input_data.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(input_data, sentence):
    indexes = indexesFromSentence(input_data, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype = torch.long, device=device).view(-1,1)


def tensorFromPair(input_data, output_data, pair):
    input_tensor = tensorFromSentence(input_data, pair[0])
    target_tensor = tensorFromSentence(output_data,pair[1])
    return (input_tensor, target_tensor)
