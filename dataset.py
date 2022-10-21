import os
import random
import torch
import collections
import hashlib
import zipfile
import requests
import tarfile
from torch.utils import data
from model import get_tokens_and_segments

def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 大写字母转小写
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

# NSP预测的任务数据（二分类任务）
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        """paragraphs 是三重列表的嵌套"""
        next_sentence = random.choice(random.choice(paragraphs))  # 从所有段落中选择一条句子作为next_sentence, 同时把is_next设置为False
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph)-1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        """考虑一个'<cls>和两个'<sep>'"""
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

# MLM的任务数据(二分类任务)
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    """
    :param tokens: BERT输入的tokens list
    :param candidate_pred_positions: 不包括特殊token的bert输入序列token index的list（特殊token在mlm中不参与）
    :param num_mlm_preds: 需要遮蔽进行预测的token数量
    :param vocab:
    :return:
    """
    """遮蔽语言模型的输入创建新的token，输入包含替换的<mask>或随机token"""
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    """shuffle后用于在mlm中获取15%的随机tokens"""
    random.shuffle(candidate_pred_positions)
    for mlm_pred_positions in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        """80%:将词替换为'<mask>'token"""
        if random.random() < 0.8:
            masked_token = '<mask>'

        else:
            # 10% 不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_positions]
            else:
                """10%: 用随机词替换"""
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_positions] = masked_token
        pred_positions_and_labels.append((mlm_pred_positions, tokens[mlm_pred_positions]))
        """
        mlm_input_tokens代表替换了<mask>后的tokens
        pred_positions_and_labels代表替换词的位置和原始的token
        """

    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_posstions = []
    # tokens 是一个字符串列表
    for i, token in enumerate(tokens):
        # 在mlm中不预测的special token
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_posstions.append(i)
    # mlm 中预测15% 的随机token
    num_mlm_preds = max(1, round(len(tokens)) * 0.15)   # 需要masked token的位置范围
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_posstions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

# 定一个一个Dataset类，将<mask> token加入
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len*0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))

        # valid_lens 不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32)) # 记录每个sentence中，没有padding的个数
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充token的预测将通过乘0权重在损失中过滤掉
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)

# 将用与生成两个预训练任务的训练样本的辅助函数和其用于填充输入的辅助函数放一起，定义数据集。
# 原始的bert是使用词表大小为30000的wordpiece。我们在tokenize时，少于5次的不频繁token将被过滤掉。

class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        """
        :param paragraphs[i]:代表句子的字符串list ,最后输出的是tokenize后的句子列表
        :param max_len:
        """
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs] # 一个段落的句子，可能包含多个句子。3D
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph] # 2D, 句子
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        """获取NSP任务的数据(输出两个句子的token, segments, is_next)"""
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
        """获取MLM任务的数据(利用上一个去得到MLM的结果,token的vocab中的位置, mask在句子中的index, mask位置上对应token在vocab的位置, segments, is_next )"""
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]

        """填充输入()"""
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)
    def __getitem__(self, index):
        return (self.all_token_ids[index], self.all_segments[index], self.valid_lens[index],
                self.all_pred_positions[index], self.all_mlm_weights[index], self.all_mlm_labels[index],
                self.nsp_labels[index])
    def __len__(self):
        return len(self.all_token_ids)

def tokenize(lines, token='word'):
    '''将文本行拆分为单词或字词元'''
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Error: unknown token' + token)

class Vocab:
    """text token"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        """按出现的频率"""
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        """未知token索引"""
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    @property
    def unk(self):
        """未知token索引为0"""
        return 0
    @property
    def token_ferqs(self):
        return self._token_freqs

# 统计token频率
def count_corpus(tokens):
    """1D list or 2D list"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def get_dataloader_workers():
    """使用四个进程来读取数据"""
    return 4

def load_data_wiki(batch_size, max_len):
    """load WikiText-2 dataset"""
    num_workers = get_dataloader_workers()
    data_dir = download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, train_set.vocab

def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'only can extact zip/tar file'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

DATA_HUB = dict()
DATA_URL = 'http://d2d-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB 中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} not exist {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname,'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'download {fname} from {url} ...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def load_data_wiki(batch_size, max_len):
    """load WikiText-2 dataset"""
    num_workers = get_dataloader_workers()
    # data_dir = download_extract('wikitext-2', 'wikitext-2')
    data_dir = '.'
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, train_set.vocab

if __name__=='__main__':
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    for (tokens_X, segments_X, valid_lens_X, pred_position_X, mlm_weights_X, mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_X.shape, pred_position_X.shape,
              mlm_weights_X.shape, mlm_Y.shape, nsp_y.shape)
        break

























