import time
import sys
from tqdm import tqdm
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from pretrian_task import BERT_Model
from model import get_tokens_and_segments
from dataset import load_data_wiki
d2l = sys.modules[__name__]

def try_all_gpus():
    """返回所有可用的GPU， 如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                         pred_positions_x, mlm_weights_X, mlm_Y, nsp_y):
    # forward
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_x.reshape(-1), pred_positions_x)

    # mlm loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1,1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)

    # nsp loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

# 训练策略
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps, epochs, out_dir):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, Timer()
    animator = Animator(xlabel='step', ylabel='loss', xlim=[1, num_steps], legend=['mlm', 'nsp'])
    ttt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    """MLM损失的和， NSP损失的和， 句子对的数量， 计数"""
    metric = Accumulator(4)
    num_steps_reached = False
    best_loss = 10000000
    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        while step < num_steps and not num_steps_reached:
            for tokens_X, segments_X, valid_lens_X, pred_positions_X, \
                mlm_weights_X, mlm_Y, nsp_Y, in train_iter:
                tokens_X = tokens_X.to(devices[0])
                segments_X = segments_X.to(devices[0])
                valid_lens_X = valid_lens_X.to(devices[0])
                pred_positions_X = pred_positions_X.to(devices[0])
                mlm_weights_X = mlm_weights_X.to(devices[0])
                mlm_Y, nsp_Y = mlm_Y.to(devices[0]), nsp_Y.to(devices[0])

                trainer.zero_grad()
                timer.start()
                mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_X,
                             pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y)
                total_loss += l.data.item()
                l.backward()
                trainer.step()
                metric.add(mlm_l ,nsp_l, tokens_X.shape[0], l)

                timer.stop()
                animator.add(step+1, (metric[0] / metric[3], metric[1] / metric[3]))
                step += 1

                if step == num_steps:
                    num_steps_reached = True
                    break
        print(f'MLM loss {metric[0] / metric[3]: .3f}, '
              f'NSP loss {metric[1] / metric[3]: .3f}')
        print(f'{metric[2] / timer.sum(): .1f} sentence pairs/ sec on '
              f'{str(devices)}')
        if total_loss < best_loss:  # 保存最优模型（loss最小的模型）,注意为了方便后续的进一步使用，不要使用torch.save保存，而是使用transfomers提供的内置保存方法
            print('save best model to {}!!!'.format(out_dir))
            best_loss = total_loss
            torch.save(net.module.state_dict(), out_dir)
            # net.save_pretrained('out_dir_{}'.format(ttt))

# 预训练完bert，用bert表示单个文本，返回tokens_a, tokens_b的词元的表示
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    tokens_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(tokens_ids, segments, valid_len)
    return encoded_X



# 计算训练时间
class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动器"""
        self.tik = time.time()
    def stop(self):
        """停止记录"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        """平均时长"""
        return sum(self.times) / len(self.times)
    def sum(self):
        """总时长"""
        return sum(self.times)
    def cumsum(self):
        """累计时长"""
        return np.array(self.times).cumsum().tolist()
# 在动画中绘制数据
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图
    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小
    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点
    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__=='__main__':
    batch_size, max_len = 32, 64    # 原始bert的 max_len = 512

    train_iter, vocab = load_data_wiki(batch_size, max_len)
    num_steps = len(train_iter)
    # load dataset
    bert = BERT_Model(len(vocab), num_hiddens=128, norm_shape=[128], ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                      num_layers=2, dropout=0.2, key_size=128, query_size=128, value_size=128, hid_in_features=128,\
                      mlm_in_features=128, nsp_in_features=128)
    devices = try_all_gpus()
    loss = nn.CrossEntropyLoss()
    train_bert(train_iter, bert, loss, len(vocab), devices, 50, 3, './model/bert_wiki.pth')
    # train_bert(train_iter, bert, loss, len(vocab), devices, num_steps)
    #
    tokens_a = ['a', 'crane', 'is', 'flying']
    encoded_text = get_bert_encoding(bert, tokens_a)
    print(encoded_text.shape)
    # # 词元：'<cls>','a','crane','is','flying','<sep>'
    # encoded_text_cls = encoded_text[:, 0, :]
    # encoded_text_crane = encoded_text[:, 2, :]
    #
    # tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
    # encoded_pair = get_bert_encoding(bert, tokens_a, tokens_b)
    # # 词元：'<cls>','a','crane','driver','came','<sep>','he','just',
    # # 'left','<sep>'
    # encoded_pair_cls = encoded_pair[:, 0, :]
    # encoded_pair_crane = encoded_pair[:, 2, :]
    # encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
