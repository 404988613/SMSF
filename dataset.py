import pickle
import torch
import torch.utils.data
from torch.utils.data import Dataset

from collate_fns import collate_fn_w_aug, collate_fn, collate_fn_graph
from processors import graph_process
import config as train_config

class Stega_dataset(Dataset):

    def __init__(self, data, training=True, w_aug=False, task=None):

        self.data = data
        self.training = training
        self.w_aug = w_aug
        self.task = task

    def __getitem__(self, index):
        item = {}

        if self.training and self.w_aug:
            item["tokenized_sentence"] = self.data["tokenized_sentence"][index]

        else:
            item["tokenized_sentence"] = torch.LongTensor(self.data["tokenized_sentence"][index])
        item["label"] = self.data["label"][index]
        item["sentence"] = self.data["sentence"][index]
        if self.task == "chi_square":
            item["sentence_chi_square"] = self.data["sentence_chi_square"][index]
        if self.task == "graph_steganalysis":
            item["tokenized_sentence_attn_mask"] = self.data["tokenized_sentence_attn_mask"][index]
            item["token_type_ids"] = self.data["token_type_ids"][index]  # tensor:[len,128]
            item["sentence_dependency_matrix"] = self.data["sentence_dependency_matrix"][index]

        return item

    def __len__(self):
        return len(self.data["label"])


def get_dataloader(batch_size, corpus, stego_method, dataset, seed=None, w_aug=True, label_list=None, task=None, pkl_file=None):

    path = '/home/xuzh/code/stega-analysis/SCL-Stega/data/Steganalysis/'
    if w_aug:
        with open(path+corpus+"/"+stego_method+"/"+dataset+pkl_file, "rb") as f:
            data = pickle.load(f)
            f.close()
    else:
        print("============================== No Aug ===========================================")
        if task == "graph_steganalysis":

            data = graph_process.preprocess_data(corpus = train_config.corpus, stego_method = train_config.stego_method, dataset = train_config.dataset[0], tokenizer_type = "/home/xuzh/code/bert-base-uncased/")

        else:
            with open(path + corpus + "/" + stego_method + "/" + dataset + pkl_file, "rb") as f:
                data = pickle.load(f)
            f.close()

    train_dataset = Stega_dataset(data["train"], training=True, w_aug=w_aug, task=task)
    valid_dataset = Stega_dataset(data["dev"], training=False, w_aug=w_aug, task=task)
    test_dataset = Stega_dataset(data["test"], training=False, w_aug=w_aug, task=task)

    if w_aug:
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_w_aug, num_workers=0)
        valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                                 num_workers=0)
        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                                num_workers=0)
    else:
        if task == "graph_steganalysis":
            train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_graph, num_workers=0)

            valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_graph, num_workers=0)
            test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_graph, num_workers=0)

        else:
            train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

            valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
            test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    return train_iter, valid_iter, test_iter

