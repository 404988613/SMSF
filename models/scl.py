from torch import nn
from torch.nn import functional as F
from transformers import ElectraModel, BertModel

class ELECTRA(nn.Module):

    def __init__(self, hidden_size=768, label_size=2, options_name = "/home/xuzh/code/electra-base-discriminator/"):
        super(ELECTRA, self).__init__()

        self.bert = ElectraModel.from_pretrained(options_name, num_labels=label_size)
        # self.encoder_supcon.encoder.config.gradient_checkpointing = True
        for param in self.bert.parameters():
            param.requires_grad = True

        self.pooler_fc = nn.Linear(hidden_size, hidden_size)  # fully-connected layer
        self.pooler_dropout = nn.Dropout(0.1)
        self.label = nn.Linear(hidden_size,label_size)  # output logits

    def get_emedding(self, features):
        x = features[:, 0, :]
        x = self.pooler_fc(x)
        x = self.pooler_dropout(x)
        x = F.relu(x)
        return x

    def forward(self, inputs=None, tokenized_inputs=None, attn_mask=None, model_type=None):
        supcon_fea = self.bert(tokenized_inputs, attn_mask, output_hidden_states=True, output_attentions=True, return_dict=True)
        supcon_fea_cls_logits = self.get_emedding(supcon_fea.hidden_states[-1])
        supcon_fea_cls_logits = self.pooler_dropout(self.label(supcon_fea_cls_logits))
        supcon_fea_cls = F.normalize(supcon_fea.hidden_states[-1][:, 0, :], dim=1)

        # outputs = self.bert(input_ids=tokenized_inputs, attention_mask=attn_mask)
        # outputs = outputs['pooler_output']
        # supcon_fea_cls_logits = self.label(outputs)
        # supcon_fea_cls = F.normalize(outputs, dim=1)

        return supcon_fea_cls_logits, supcon_fea_cls
