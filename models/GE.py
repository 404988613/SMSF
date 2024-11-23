import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import BertModel


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, filter_num, filter_size, dropout_rate):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # vocab_size：当前要训练的文本中不重复单词的个数
        self.cnn_list = nn.ModuleList()
        for size in filter_size:
            self.cnn_list.append(nn.Conv1d(embed_size, filter_num, size, padding = 'same'))
        self.relu = nn.ReLU()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        # self.output_layer = nn.Linear(filter_num * len(filter_size), class_num)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        """
		:param x:(N,L)，N为batch_size，L为句子长度
		:return: (N,class_num) class_num是分类数，文本隐写分析最终分为stego和cover两类
		"""
        x = x.long()
        _ = self.embedding(x)  # 词嵌入，（N,L,embed_size）
        # _ = x
        _ = _.permute(0, 2, 1)  # 卷积是在最后一维进行，因此要交换embed_size维和L维，在句子上进行卷积操作
        result = []  # 定义一个列表，存放每次卷积、池化后的特征值，最终列表元素个数=卷积核size的个数
        for self.cnn in self.cnn_list:
            __ = self.cnn(_)  # 卷积操作
            __ = self.relu(__)
            result.append(__)  # 判断第2维的维度是否为1，若是则去掉.因为池化后的第2维是1，因此这里是去掉第2维，结果是（batch_size,filter_num）

        _ = torch.cat(result, dim = 1)  # 将result列表中的元素在行上进行拼接
        _ = self.dropout(_)
        _ = self.softmax(_)
        return _


class TextRNN(nn.Module):
    """RNN"""

    def __init__(self, cell, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate):
        """

        :param cell: 隐藏单元类型'rnn','bi-rnn','gru','bi-gru','lstm','bi-lstm'
        :param vocab_size: 词表大小
        :param embed_size: 词嵌入维度
        :param hidden_dim: 隐藏神经元数量
        :param num_layers: 隐藏层数
        :param class_num:
        :param dropout_rate:
        """
        super(TextRNN, self).__init__()
        self._cell = cell

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = None
        if cell == 'rnn':
            self.rnn = nn.RNN(embed_size, hidden_dim, num_layers, batch_first = True, dropout = dropout_rate)
            out_hidden_dim = hidden_dim * num_layers
        elif cell == 'bi-rnn':
            self.rnn = nn.RNN(
                embed_size, hidden_dim, num_layers, batch_first = True, dropout = dropout_rate, bidirectional = True
                )
            out_hidden_dim = 2 * hidden_dim * num_layers
        elif cell == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_dim, num_layers, batch_first = True, dropout = dropout_rate)
            out_hidden_dim = hidden_dim * num_layers
        elif cell == 'bi-gru':
            self.rnn = nn.GRU(
                embed_size, hidden_dim, num_layers, batch_first = True, dropout = dropout_rate, bidirectional = True
                )
            out_hidden_dim = 2 * hidden_dim * num_layers
        elif cell == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first = True, dropout = dropout_rate)
            out_hidden_dim = hidden_dim * num_layers
        elif cell == 'bi-lstm':
            self.rnn = nn.LSTM(
                embed_size, hidden_dim, num_layers, batch_first = True, dropout = dropout_rate, bidirectional = True
                )
            out_hidden_dim = 2 * hidden_dim * num_layers
        else:
            raise Exception("no such rnn cell")

        # self.output_layer = nn.Linear(out_hidden_dim, k)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        """
		:param x:(B，L）B：batch_size, L：sentence_length
		:return: (B, hidden_dim*2, L） bi-gru
		"""
        x = x.long()
        _ = self.embedding(x)  # (B, L, E)
        # _ = x
        __, h_out = self.rnn(_)  # Bi-gru:(B, L, H*2), (2*n_l, B, H); gru:(B, L, H), (1, B, H)
        # h_out = h_out.reshape(1, -1, h_out.shape[0] * h_out.shape[2]) # (1, B, 2*n_l*H)
        __ = __.permute(0, 2, 1)
        _ = self.softmax(__)
        return _


class GroupEnhance(nn.Module):
    def __init__(self, groups=5):
        super(GroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化，获取每个group的代表特征向量
        self.weight = Parameter(torch.zeros(1, groups, 1))
        self.bias = Parameter(torch.ones(1, groups, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):  # (b, f, l)=(2,20,2)=80 group = 4
        """
        :param x: 上层输出的特征map (b, f, l)
        :return: 增强后的特征map，(b, f, l)
        """
        b, f, l = x.size()  # b=l=2,f=30
        x = x.view(b * self.groups, -1, l)  # (2*4,5,2)
        dot = x * self.avg_pool(x)  # 平均值与每个位置进行点积(8,5,2)*(8*5*1)=(8,5,2)
        dot = dot.sum(dim = 1, keepdim = True)  # 获得每个group的系数map (8,1,2)
        norm = dot.view(b * self.groups, -1)  # (8,2)
        norm = norm - norm.mean(dim = 1, keepdim = True)  # 除了batch * groups外，计算均值
        std = norm.std(dim = 1, keepdim = True) + 1e-5  # 标准差(2,1)
        norm = norm / std  # 标准化，是数据在0左右分布 (8,2)
        norm = norm.view(b, self.groups, l)  # (2,4,2)
        norm = norm * self.weight + self.bias
        norm = norm.view(b * self.groups, 1, l)
        x = x * self.sig(norm)  # (b*group,f/group,l)(2*4,5,2)
        x = x.view(b, f, l)
        return x


class ge(nn.Module):
    def __init__(self, cell, vocab_size, embed_size, filter_num, filter_size, hidden_dim, num_layers, class_num,
                 dropout_rate, g, k, model_name_or_path="/home/xuzh/code/bert-base-uncased/"):
        super(ge, self).__init__()
        self.cnn = TextCNN(vocab_size, embed_size, filter_num, filter_size, dropout_rate)
        self.rnn = TextRNN(cell, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate)
        self.g_e = GroupEnhance(g)
        self.dropout = nn.Dropout(dropout_rate)
        self.k_pool = nn.AdaptiveMaxPool1d(k)  # k-max-pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # max_pooling

        self.conv1 = nn.Conv1d(in_channels = 4 * hidden_dim, out_channels = 20, kernel_size = 3, padding = 'same')

        self.output_layer = nn.Linear(filter_num * 4, class_num)  # 不含GE模块
        self.output_layer1 = nn.Linear(20 * k, class_num)  # 卷积后
        self.softmax = nn.Softmax(dim = 1)

        self.bert_model_path = model_name_or_path
        self.bert = BertModel.from_pretrained(self.bert_model_path)
        self.embedding_linear = nn.Linear(768, 300)

    def forward(self, inputs=None, tokenized_inputs=None, attn_mask=None, model_type=None):
        """
        :param x:文本
        :return: 二分类结果
        """
        # outputs = self.bert(
        #     input_ids = tokenized_inputs,
        #     attention_mask = attn_mask,
        #     # token_type_ids = token_type_ids
        #     )  # [batch_size, sequence_length, hidden_size]
        # embedding = outputs[0]  # [batch_size, node,hidden_size]
        # # 768->128
        # embedding = self.embedding_linear(embedding)
        # x = embedding
        x = tokenized_inputs
        cnn_res = self.cnn(x)  # (B,filter_num * length(filter_size),L)
        rnn_res = self.rnn(x)  # (B, H*2, L） H：hidden_dim，NL：num_layers
        # 特征增强
        cge = self.g_e(cnn_res)  # 增强后的局部特征:(B,f,L),f=filter_num * length(filter_size)=hidden_dim*2
        rge = self.g_e(rnn_res)  # 增强后的全局特征:(B,f,L)
        """不增强/K增强，卷积后，输入output_layer1"""
        # _cat = torch.cat((cnn_res, rnn_res), dim=1) # (B, H*4, L） # 不增强+串联
        _cat = torch.cat((self.k_pool(cge), self.k_pool(rge)), dim = 1)  # K增强+串联(B, H*4, k）,k=10
        _ = F.relu(self.k_pool(self.dropout(self.conv1(_cat))))  # 串联卷积(B,out_channels,k),out_channels=20,k=10
        _ = _.view(_.shape[0], -1)  # (B,out_channels*k) # 展平
        _ = self.output_layer1(_)
        logits = self.softmax(_)
        return logits, _

# if __name__ == "__main__":
#     x = torch.rand((2, 10))
#     ge = ge("bi-gru",20,20,4,[3],2,1,2,0.5,1,1)
#     res = ge(x)
#     # print(x)
#     print(res)
