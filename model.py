import torch
import math

from torch import nn

# 将句子加入固定的开始与结束标记
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """
    :param tokens_a: 第一条句子的tokens
    :param tokens_b: 第二条句子的tokens， 默认只有一条句子。
    :return: tokens 与 segments
    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1代表A，B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b +['<sep>']
        segments += [1] * (len(tokens_b) +1)
    return tokens, segments

# BERT编码器
class BERTEncoder(nn.Module):
    #
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,\
                 num_heads, num_layers, dropout, max_len,\
                 key_size, query_size, value_size, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blocks = nn.Sequential()
        for i in range(num_layers):
            self.blocks.add_module(f"{i}", EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                                        norm_shape, ffn_num_input, ffn_num_hiddens,
                                                        num_heads, dropout, True))
        # bert中 position embedding是可学习的
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for bls in self.blocks:
            X = bls(X, valid_lens)
        return X





'''
define
'''

class EncoderBlock(nn.Module):
    #
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.add_norm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.add_norm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y0 = self.attention(X, X, X, valid_lens)
        Y = self.add_norm1(X, Y0)
        Y1 = self.ffn(Y)
        return self.add_norm2(Y, Y1)


class MultiHeadAttention(nn.Module):
    '''
    Defined in :numref: 'sec_multihead-attention'
    '''
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        :param queries:
        :param keys:
        :param values:
        :param valid_lens:
        :return:
        """

        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values= transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在dim 0， 将第一个复制num_heads次。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )
        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    # (batch_size, num, num_hiddens) -> (batch_size, num, num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # (batch_size, num, num_heads, num_hiddens/num_heads) -> (batch_size, num_heads, num, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # (batch_size, num_heads, num, num_hiddens/num_heads) -> (batch_size*num_heads, num, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    # 残差结构
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2) / math.sqrt(d))
        self.attention_weights = masked_softmax(scores, valid_lens)
        weights = self.dropout(self.attention_weights)
        return torch.bmm(weights , values)

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)

def sequence_mask(X, valid_lens, value=0):
    max_len = X.size(1)
    mask = torch.arange((max_len), dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
    X[~mask] = value
    return X


if __name__=='__main__':
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                          num_heads, num_layers, dropout)

    # create tokens and segments
    tokens = torch.randint(0, vocab_size, (2,8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])

    encoded_X = encoder(tokens, segments, None)
    encoded_X.shape



















