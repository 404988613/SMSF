# -*- coding: utf-8 -*-
# @Time : 2024/3/15 9:09
# @File : TDA-FSMS.py
# @Software : PyCharm

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn import Parameter
from transformers import AutoModel, AutoConfig
import torch
from torch import nn
import torch.nn.functional as F

class TFIDF(nn.Module):
    def __init__(self,data = None, z_dim = 768, n_dim = 128):
        super(TFIDF, self).__init__()

        # 函数说明:创建数据样本
        dataset = data["sentence"]
        labels = data["label"]
        self.max_features = 9000
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, dtype=np.float32)

        # 在训练集上进行向量化和训练
        train_vectors = self.vectorizer.fit_transform(dataset,y=labels)

        self.n_dim = n_dim
        self.z_dim = z_dim
        self.linear = nn.Linear(in_features=self.max_features, out_features=self.z_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        # 将文本转为词频矩阵并计算tf-idf
        # 在测试集上进行向量化和预测
        tf_idf = self.vectorizer.transform(text)
        # tf_idf = self.tf_idf_transformer.fit_transform(self.vectorizer.fit_transform(text))
        # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        tf_idf = torch.tensor(tf_idf.toarray()).cuda()
        tf_idf = self.linear(tf_idf)
        tf_idf = self.dropout(tf_idf)
        return tf_idf

class DAN(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.alpha = Parameter(torch.zeros(1))


    def forward(self, input_features):
        Q = self.query(input_features)
        K = self.key(input_features)
        V = self.value(input_features)
        Q1 = Q.transpose(-2,-1)
        S = torch.softmax(torch.matmul(Q1, K),dim=1)
        S1 = self.alpha * torch.matmul(V,S)
        output_features = input_features + S1

        return output_features

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

# model : use DAN map Statistical and Multi-layer Semantic features into the same space and then add fusion
class ETAFF(nn.Module):
    def __init__(self, bert_config='/home/xuzh/code/ernie-2.0-base-en/',
                 data = None, hidden_size=768, label_size=2, num_class=2, z_dim=768, n_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        # tfidf
        self.TFIDF = TFIDF(data=data, z_dim=self.z_dim, n_dim=self.n_dim)
        self.tfidf_linear = nn.Linear(self.z_dim, self.z_dim)

        self.config = AutoConfig.from_pretrained(bert_config, num_labels=2)
        self.config.output_hidden_states = True#需要设置为true才输出
        self.bert = AutoModel.from_pretrained(bert_config, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(13)
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size // 12) for _ in range(13)
        ])
        self.ernie_linear = nn.Linear(hidden_size, hidden_size)

        self.aff = AFF()
        self.aligner1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.dan1 = DAN()
        self.aligner2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )
        self.dan2 = DAN()
        self.label = nn.Linear(self.hidden_size, label_size)  # output logits

    def forward(self, inputs=None, tokenized_inputs=None, attn_mask=None, model_type=None):

        tfidf = self.TFIDF(inputs)
        tfidf = self.dropout(self.tfidf_linear(tfidf))

        outputs = self.bert(input_ids=tokenized_inputs, attention_mask=attn_mask)
        last_hidden_states = outputs['last_hidden_state'] #(batch_size, seq_len, 768)
        pool = outputs['pooler_output'] #(batch_size, 768)
        all_hidden_states = outputs['hidden_states'] # tuple:13
        batch_size = tokenized_inputs.shape[0]

        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768) # 取13个的(batch_size,0,768)得到(batch_size * 13, 1, 768)
        # ht_cls = ht_cls[1:]
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2]) #(batch_size, 768)
        hs = []
        for i, fc in enumerate(self.fcs):
            if i == 0:
                continue
            h = fc(self.dropout(all_hidden_states[i][:, 0, :]))
            hs.append(h)
        hs = torch.cat(hs, dim=1)
        ernie = self.ernie_linear(hs)
        # hs = self.fc(self.dropout(hs))


        # DAN
        ernie = self.aligner1(ernie)
        ernie = self.dan1(ernie)
        tfidf = self.aligner2(tfidf)
        tfidf = self.dan2(tfidf)
        h = self.label(self.dropout(ernie + tfidf))

        # ablution : AFF direct attention
        # h = self.aff(ernie, tfidf)

        supcon_fea_cls_logits = h
        supcon_fea_cls = F.normalize(pool, dim=1)
        return supcon_fea_cls_logits, supcon_fea_cls


# ablation study

# only TF-IDF
class T(nn.Module):
    def __init__(self, bert_config='/home/xuzh/code/ernie-2.0-base-en/',
                 data=None, hidden_size=768, label_size=2, num_class=2, z_dim=768, n_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        # tfidf
        self.TFIDF = TFIDF(data=data, z_dim=self.z_dim, n_dim=self.n_dim)
        self.tfidf_linear = nn.Linear(self.z_dim, self.z_dim)

        self.dropout = nn.Dropout(0.1)
        self.label = nn.Linear(self.hidden_size, label_size)  # output logits

    def forward(self, inputs=None, tokenized_inputs=None, attn_mask=None, model_type=None):

        tfidf = self.TFIDF(inputs)
        tfidf = self.dropout(self.tfidf_linear(tfidf))

        h = self.label(self.dropout(tfidf))

        supcon_fea_cls_logits = h
        supcon_fea_cls = F.normalize(tfidf, dim=1)
        return supcon_fea_cls_logits, supcon_fea_cls
# only ERNIE
class ernie(nn.Module):
    def __init__(self, bert_config='/home/xuzh/code/ernie-2.0-base-en/', hidden_size=768, label_size=2):
        super().__init__()

        # self.tokenizer = XLNetTokenizer.from_pretrained('/home/xuzh/code/ernie-2.0-base-en/', do_lower_case=True)
        self.bert = AutoModel.from_pretrained(bert_config)
        # self.bert.encoder.config.gradient_checkpointing = True
        for param in self.bert.parameters():
            param.requires_grad = True

        self.label = nn.Linear(hidden_size, label_size)  # output logits

    def forward(self, inputs=None, tokenized_inputs=None, attn_mask=None, model_type=None):

        outputs = self.bert(input_ids=tokenized_inputs, attention_mask=attn_mask)
        # outputs = outputs['pooler_output']
        outputs = outputs['last_hidden_state'][:,0,:]
        supcon_fea_cls_logits = self.label(outputs)
        supcon_fea_cls = F.normalize(outputs, dim=1)

        return supcon_fea_cls_logits, supcon_fea_cls
# multi-layer ERNIE
class ernie_mlayers(nn.Module):
    def __init__(self, bert_config='/home/xuzh/code/ernie-2.0-base-en/', hidden_size=768, label_size=2, num_class=2):
        super().__init__()

        self.config = AutoConfig.from_pretrained(bert_config, num_labels=2)
        self.config.output_hidden_states = True#需要设置为true才输出
        self.bert = AutoModel.from_pretrained(bert_config, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(13)
        ])
        self.dropout = nn.Dropout(0.1)
        # self.fc13 = nn.Linear(hidden_size * 13, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size // 12) for _ in range(13)
        ])
        # self.fcs = nn.ModuleList([
        #     IntraAttention(d_model=768) for _ in range(13)
        # ])
        self.label = nn.Linear(hidden_size, label_size)  # output logits

    def forward(self, inputs=None, tokenized_inputs=None, attn_mask=None, model_type=None):

        outputs = self.bert(input_ids=tokenized_inputs, attention_mask=attn_mask)
        last_hidden_states = outputs['last_hidden_state'] #(batch_size, seq_len, 768)
        pool = outputs['pooler_output'] #(batch_size, 768)
        all_hidden_states = outputs['hidden_states'] # tuple:13
        batch_size = tokenized_inputs.shape[0]

        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768) # 取13个的(batch_size,0,768)得到(batch_size * 13, 1, 768)
        # ht_cls = ht_cls[1:]
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2]) #(batch_size, 768)
        hs = []
        for i, fc in enumerate(self.fcs):
            if i == 0:
                continue
            h = fc(self.dropout(all_hidden_states[i][:, 0, :]))
            hs.append(h)
        hs = torch.cat(hs, dim=1)
        # hs = self.fc(self.dropout(hs))
        h = self.label(self.dropout(hs))

        supcon_fea_cls_logits = h
        supcon_fea_cls = F.normalize(pool, dim=1)
        return supcon_fea_cls_logits, supcon_fea_cls
# direct add Statistical and Multi-layer Semantic features without attention fusion module
class ET(nn.Module):
    def __init__(self, bert_config='/home/xuzh/code/ernie-2.0-base-en/',
                 data = None, hidden_size=768, label_size=2, num_class=2, z_dim=768, n_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.hidden_size = hidden_size
        # tfidf
        self.TFIDF = TFIDF(data=data, z_dim=self.z_dim, n_dim=self.n_dim)
        self.tfidf_linear = nn.Linear(self.z_dim, self.z_dim)

        self.config = AutoConfig.from_pretrained(bert_config, num_labels=2)
        self.config.output_hidden_states = True#需要设置为true才输出
        self.bert = AutoModel.from_pretrained(bert_config, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(13)
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size // 12) for _ in range(13)
        ])
        self.ernie_linear = nn.Linear(hidden_size, hidden_size)

        self.label = nn.Linear(self.hidden_size, label_size)  # output logits

    def forward(self, inputs=None, tokenized_inputs=None, attn_mask=None, model_type=None):

        tfidf = self.TFIDF(inputs)
        tfidf = self.dropout(self.tfidf_linear(tfidf))

        outputs = self.bert(input_ids=tokenized_inputs, attention_mask=attn_mask)
        last_hidden_states = outputs['last_hidden_state'] #(batch_size, seq_len, 768)
        pool = outputs['pooler_output'] #(batch_size, 768)
        all_hidden_states = outputs['hidden_states'] # tuple:13
        batch_size = tokenized_inputs.shape[0]

        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768) # 取13个的(batch_size,0,768)得到(batch_size * 13, 1, 768)
        # ht_cls = ht_cls[1:]
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2]) #(batch_size, 768)
        hs = []
        for i, fc in enumerate(self.fcs):
            if i == 0:
                continue
            h = fc(self.dropout(all_hidden_states[i][:, 0, :]))
            hs.append(h)
        hs = torch.cat(hs, dim=1)
        ernie = self.ernie_linear(hs)

        h = self.label(self.dropout(ernie + tfidf))

        supcon_fea_cls_logits = h
        supcon_fea_cls = F.normalize(pool, dim=1)
        return supcon_fea_cls_logits, supcon_fea_cls
# AFF fusion Statistical and Multi-layer Semantic features
