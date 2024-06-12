# -*- coding: utf-8 -*-
# @Time : 2023/12/13 20:45
# @File : data_split.py
# @Software : PyCharm
import random
import os
import csv
from sklearn.model_selection import train_test_split

corpus = "IMDB"
stego_method = "AC"
dataset = "1bpw"
data_path = "./data/Steganalysis/"+corpus+"/"+stego_method+"/"
SEED=0
random.seed(SEED)
os.makedirs(data_path, exist_ok=True)
with open(data_path + "cover.txt", 'r', encoding='utf-8') as f:
    covers = f.read().split("\n")
covers = list(filter(lambda x: x not in ['', None], covers))
random.shuffle(covers)
with open(data_path + "stego1.txt", 'r', encoding='utf-8') as f:
    stegos = f.read().split("\n")
stegos = list(filter(lambda x: x not in ['', None],  stegos))
random.shuffle(stegos)
def split_data(texts,labels):
    train_texts,test_texts,train_labels,test_labels = train_test_split(texts,labels,train_size=6/7, random_state = random.seed(SEED), shuffle = False)
    train_texts,val_texts, train_labels,val_labels,  = train_test_split(train_texts, train_labels, train_size=5/6,  random_state = random.seed(SEED), shuffle = False)
    return train_texts,val_texts, train_labels,val_labels, test_texts,test_labels
texts_cover = covers
labels_cover = [0]*len(covers)
c_train_texts,c_val_texts, c_train_labels,c_val_labels, c_test_texts,c_test_labels = split_data(texts_cover, labels_cover)

texts_stego = stegos
labels_stego = [1] * len(stegos)
s_train_texts,s_val_texts, s_train_labels,s_val_labels, s_test_texts,s_test_labels = split_data(texts_stego, labels_stego)
train_texts = c_train_texts + s_train_texts
train_labels = c_train_labels + s_train_labels
val_texts = c_val_texts + s_val_texts
val_labels = c_val_labels + s_val_labels
test_texts = c_test_texts + s_test_texts
test_labels = c_test_labels + s_test_labels

def write2file(X, Y, filename):
    with open(filename, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "label"])
        for x, y in zip(X, Y):
            writer.writerow([x, y])
write2file(train_texts,train_labels, os.path.join(data_path + dataset,"train.csv"))
write2file(val_texts, val_labels, os.path.join(data_path + dataset, "dev.csv"))
write2file(test_texts, test_labels, os.path.join(data_path + dataset, "test.csv"))