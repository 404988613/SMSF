# -*- coding: utf-8 -*-
# @Time : 2023/11/15 9:09
# @File : bert.py
# @Software : PyCharm

from transformers import BertModel
from torch import nn
import torch.nn.functional as F

class Bert(nn.Module):
    def __init__(self, bert_config='/home/xuzh/code/bert-base-uncased/', hidden_size=768, label_size=2):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained(bert_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.label = nn.Linear(hidden_size, label_size)  # output logits

    def forward(self, inputs=None, tokenized_inputs=None, attn_mask=None, model_type=None):

        outputs = self.bert(input_ids=tokenized_inputs, attention_mask=attn_mask)
        outputs = outputs['pooler_output']
        supcon_fea_cls_logits = self.label(outputs)
        supcon_fea_cls = F.normalize(outputs, dim=1)

        return supcon_fea_cls_logits, supcon_fea_cls
