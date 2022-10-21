import torch
import math
from torch import nn
from model import BERTEncoder
# 定义MLM任务类
class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)

        # resume batch_size = 2, num_pred_positions = 3; then batch_idx is np.array([0,0,0,1,1,1])
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

# 定义Next Sentence Prediction
class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)
    def forward(self, X):
        # X.shape (batch_size, num_hiddens)
        return self.output(X)

# BERT Model
class BERT_Model(nn.Module):
    def __init__(self, vocab_size=20256, num_hiddens=128, norm_shape=[128], ffn_num_input=128, ffn_num_hiddens=256,
                 num_heads=2, num_layers=2, dropout=0.2, max_len=1000, key_size=128, query_size=128, value_size=128,
                 hid_in_features=128, mlm_in_features=128, nsp_in_features=128):
        super(BERT_Model, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len=max_len, key_size=key_size, query_size=query_size,
                                   value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)
    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


if __name__=='__main__':
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 128, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                          num_heads, num_layers, dropout)

    # create tokens and segments
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])

    encoded_X = encoder(tokens, segments, None)


    mlm = MaskLM(vocab_size, num_hiddens)
    mlm_positions = torch.tensor([[1,5,2],[6,1,5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)


    mlm_Y = torch.tensor([[7,8,9], [10,20,30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))


    encoded_X = torch.flatten(encoded_X, start_dim=1)
    nsp = NextSentencePred(encoded_X.shape[-1])
    nsp_Y_hat = nsp(encoded_X)

    nsp_y = torch.tensor([0,1])
    nsp_l = loss(nsp_Y_hat, nsp_y)








