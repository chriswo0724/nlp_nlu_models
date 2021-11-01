import re 
import unicodedata
from DataStructure import DataStructurer
import random

MAX_LENGTH =  10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "   
)

#将读取的unicode编码转换成ascii编码
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readdatafile(langtype1, langtype2, reverse=False):
    print("Reading lines ...")

    #read the file and split into lines
    lines = open(('NLU_model/seq2seq_model/data/%s-%s.txt') % (langtype1, langtype2), encoding = 'utf-8'). \
        read().strip().split('\n')
    
    #split every line into pairs and normalize 
    #two-layers-for: each sentence in each l in lines 
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    
    #pairs_1 = [] 
    #for l in lines[0:10]:
    #    pairs_1.append([normalizeString(s) for s in l.split('\t')])

    '''
    print(type(pairs))
    print(type(pairs_1))
    print(pairs)
    print(pairs_1)
    '''

    #reverse pairs, make lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_data = DataStructurer(langtype2)
        output_data = DataStructurer(langtype1)
    else:
        input_data = DataStructurer(langtype1)
        output_data = DataStructurer(langtype2)       

    return input_data, output_data, pairs

#选择一对语句中简单的少于10个词的语句，方便快速训练
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

#过滤所有的lang1-lang2 pairs
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareDate(langtype1, langtype2, reverse = False):
    input_data, output_data, pairs =  readdatafile(langtype1, langtype2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    #filter pairs
    #pairs = filterPairs(pairs)
    #print("Trimed to %s sentence pairs" % len(pairs))
    #print("Counting words ...")

    for pair in pairs:
        input_data.addSentence(pair[0])
        output_data.addSentence(pair[1])

    print("counted words:")
    print(input_data.name, input_data.n_words)
    print(output_data.name, output_data.n_words)

    return input_data, output_data, pairs

if __name__ == '__main__':
    
    #第三个字段为是否reverse 任务互换 eng->fra 换成 fra->eng
    input_data, output_data, pairs = prepareDate('eng', 'fra', False)
    print(random.choice(pairs))