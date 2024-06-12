import torch
import itertools

def collate_fn(data):

    def merge(sequences, N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: # pads to the max length of the batch
            N = max(lengths)

        padded_seqs = torch.zeros(len(sequences),N).long()
        attention_mask = torch.zeros(len(sequences),N).long()

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i, : end] = torch.ones(end).long()

        return padded_seqs, attention_mask,lengths

    data.sort(key=lambda x: len(x["tokenized_sentence"]), reverse=True) # sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    sentence_batch, sentence_attn_mask, sentence_lengths = merge(item_info['tokenized_sentence'])

    d = {}
    d["label"] = item_info["label"]
    d["sentence"] = item_info["sentence"]
    d["tokenized_sentence"] = sentence_batch
    d["tokenized_sentence_attn_mask"] = sentence_attn_mask

    return d

def collate_fn_graph(data):

    data.sort(key=lambda x: len(x["tokenized_sentence"]), reverse=True) # sort by source seq

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    d = {}
    d["label"] = item_info["label"]
    d["sentence"] = item_info["sentence"]
    d["tokenized_sentence"] = torch.stack(item_info['tokenized_sentence'])  # tensor:[len,128]
    d["tokenized_sentence_attn_mask"] = torch.stack(item_info["tokenized_sentence_attn_mask"])  # tensor:[len,128]
    d["token_type_ids"] = torch.stack(item_info["token_type_ids"])  # tensor:[len,128]
    d["sentence_dependency_matrix"] = torch.stack(item_info["sentence_dependency_matrix"])  # tensor:[len,128,128]
    return d

def collate_fn_w_aug(data):

    def merge(sequences,N=None):
        lengths = [len(seq) for seq in sequences]

        if N == None: # pads to the max length of the batch
            N = max(lengths)

        padded_seqs = torch.zeros(len(sequences), N).long()
        attention_mask = torch.zeros(len(sequences), N).long()

        for i, seq in enumerate(sequences):
            seq = torch.LongTensor(seq)
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            attention_mask[i, : end] = torch.ones(end).long()

        return padded_seqs, attention_mask, lengths

    # each data sample has two views
    data.sort(key=lambda x: max(len(x["tokenized_sentence"][0]), len(x["tokenized_sentence"][1])), reverse=True) # sort all the seq incl augmented

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
        # unbinding the two views here as both views needs to be within the batch
        flat = itertools.chain.from_iterable(item_info[key])
        item_info[key] = list(flat)

    # input
    sentence_batch, sentence_attn_mask, sentence_lengths = merge(item_info['tokenized_sentence'])

    d = {}
    d["label"] = item_info["label"]
    d["sentence"] = item_info["sentence"]
    d["tokenized_sentence"] = sentence_batch
    d["tokenized_sentence_attn_mask"] = sentence_attn_mask

    return d
