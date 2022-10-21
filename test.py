import torch
from train import get_bert_encoding
from pretrian_task import BERT_Model
net= BERT_Model()
net.load_state_dict(torch.load('./model/bert_wiki.pth'))

tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
print(encoded_text)