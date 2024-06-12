import argparse
import torch
import pandas as pd
import pickle
import config_preprocess as config
from transformers import AutoTokenizer,GPT2Tokenizer,GPT2LMHeadModel
pkl_file = "_preprocessed_bert.pkl" # ['_preprocessed_ernie.pkl']

def normalize_tensor(tensor, dim):

    sum_values = torch.sum(tensor, dim=dim, keepdim=True)
    normalized_tensor = tensor / sum_values
    # 有负值，需要全部转为正值，求和为1，表示概率
    normalized_tensor = torch.softmax(normalized_tensor, dim = dim)
    return normalized_tensor

def preprocess_data(corpus, stego_method, dataset,tokenizer_type,w_aug=True):
        print("Extracting data")
        path = './data/Steganalysis/'
        data_home = path+corpus+"/"+stego_method+"/"+dataset+"/"

        data_dict = {}
        task = "steganalysis"
        if task == "steganalysis":
            for datatype in ["train", "dev", "test"]:
                if datatype == "train" and w_aug:
                    data = pd.read_csv(data_home+datatype+".csv")
                    data.dropna()
                    final_sentence, final_label = [], []
                    for i,val in enumerate(data["label"]):
                        final_sentence.append(data["sentence"][i])
                        final_label.append(val)

                    augmented_sentence = list(data["augment"])

                    print("Tokenizing data")
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
                    tokenized_sentence_original = tokenizer.batch_encode_plus(final_sentence).input_ids
                    tokenized_sentence_augmented = tokenizer.batch_encode_plus(augmented_sentence).input_ids

                    tokenized_combined_sentence = [list(i) for i in zip(tokenized_sentence_original, tokenized_sentence_augmented)]
                    combined_sentence = [list(i) for i in zip(final_sentence, augmented_sentence)]
                    combined_label = [list(i) for i in zip(final_label, final_label)]

                    processed_data = {}

                    # ## changed sentence --> sentence for uniformity
                    processed_data["tokenized_sentence"] = tokenized_combined_sentence
                    processed_data["label"] = combined_label
                    processed_data["sentence"] = combined_sentence

                    data_dict[datatype] = processed_data

                else:
                    data = pd.read_csv(data_home+datatype+".csv", header=0)
                    data.dropna()
                    final_sentence,final_label = [],[]
                    for i,val in enumerate(data["label"]):

                        final_sentence.append(data["sentence"][i])
                        final_label.append(val)

                    print("Tokenizing data")
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
                    tokenized_sentence_original = tokenizer.batch_encode_plus(final_sentence).input_ids

                    processed_data = {}
                    processed_data["tokenized_sentence"] = tokenized_sentence_original
                    processed_data["label"] = final_label
                    processed_data["sentence"] = final_sentence
                    data_dict[datatype] = processed_data

                if w_aug:
                    with open(path+corpus+"/"+stego_method+"/"+dataset+pkl_file, 'wb') as f:
                        pickle.dump(data_dict, f)
                    f.close()
                else:
                    with open(path+corpus+"/"+stego_method+"/"+dataset+pkl_file, 'wb') as f:
                        pickle.dump(data_dict, f)
                    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Enter tokenizer type')
    parser.add_argument('-corpus', default = "Twitter", type = str, help = 'Enter dataset=')
    parser.add_argument('-stego_method', default = "VLC", type = str, help = 'Enter dataset=')
    parser.add_argument('-d', default = "1bpw", type = str, help = 'Enter dataset=')
    parser.add_argument('-t', default = "bert-base-uncased", type = str, help = 'Enter tokenizer type')
    parser.add_argument('--aug', default = True, action = 'store_true')

    args = parser.parse_args()
    args.corpus = config.corpus
    args.stego_method = config.stego_method
    args.d = config.dataset[0]
    args.t = config.t
    args.aug = config.is_waug

    preprocess_data(args.corpus, args.stego_method, args.d, args.t, w_aug = args.aug)
    print("预处理完成！")
