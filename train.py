import numpy as np
import json
import random
import os
from easydict import EasyDict as edict
import time
from datetime import datetime
import torch
import torch.utils.data
from torch import nn
import config_ge as train_config
from dataset import get_dataloader
from util import iter_product
from sklearn.metrics import f1_score
import loss as loss
from models import bert,cnn,rnn,bilstm_dense,r_bilstm_c,sesy,scl,SMSF,GE
from transformers import AdamW, get_linear_schedule_with_warmup

def train(epoch, train_loader, model_main, loss_function, optimizer, lr_scheduler, log, stage=None):

    model_main.cuda()
    model_main.train()

    total_true, total_pred, acc_curve = [], [], []
    train_loss = 0
    total_epoch_acc = 0
    steps = 0
    start_train_time = time.time()

    # if log.param.loss_type == "ce":
    #     train_batch_size = log.param.batch_size
    # else:
    #     train_batch_size = log.param.batch_size * 2
    train_batch_size = log.param.batch_size
    for idx, batch in enumerate(train_loader):
        if "bpw" in log.param.dataset:
            tokenized_text_name = "tokenized_sentence"
            label_name = "label"
            text_name = "sentence"
            token_type_ids = "token_type_ids"
            graph_name = "sentence_dependency_matrix"

        text = batch[text_name]
        tokenized_text = batch[tokenized_text_name]
        attn = batch[tokenized_text_name+"_attn_mask"]
        label = batch[label_name]
        label = torch.tensor(label)
        label = torch.autograd.Variable(label).long()

        if (label.size()[0] is not train_batch_size):# Last batch may have length different than log.param.batch_size
            continue

        if torch.cuda.is_available():
            # text = text.cuda()
            tokenized_text = tokenized_text.cuda()
            attn = attn.cuda()
            label = label.cuda()

        if log.param.task == "graph_steganalysis":
            token_type_ids = batch[token_type_ids]
            token_type_ids = token_type_ids.cuda()
            graph = batch[graph_name]
            graph = graph.cuda()
            pred, supcon_feature = model_main(inputs=text, tokenized_inputs=tokenized_text, graph=graph, attn_mask=attn, token_type_ids=token_type_ids, model_type=log.param.model_type)
        elif log.param.task == "stage":
            pred, supcon_feature = model_main(inputs=text, tokenized_inputs=tokenized_text, attn_mask=attn,
                                              model_type=log.param.model_type, stage=None)
        else:
            pred, supcon_feature = model_main(inputs=text, tokenized_inputs=tokenized_text, attn_mask=attn, model_type=log.param.model_type)

        if log.param.loss_type == "scl":
            loss = (loss_function["lambda_loss"]*loss_function["label"](pred, label)) + ((1-loss_function["lambda_loss"]) * loss_function["contrastive"](supcon_feature, label))
        else:
            loss = loss_function["label"](pred, label)

        train_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model_main.parameters(), max_norm=1.0)

        optimizer.step()
        model_main.zero_grad()

        lr_scheduler.step()
        optimizer.zero_grad()

        steps += 1

        if steps % 100 == 0:
            print(f'Epoch: {epoch:02}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Time taken: {((time.time() - start_train_time) / 60): .2f} min')
            # logger.info('Epoch: {:d}, Idx: {}, Training Loss: {:.4f}, Time taken: {:.2f} min'.format(epoch, idx+1,loss.item(),((time.time() - start_train_time) / 60)))
            start_train_time = time.time()

        true_list = label.data.detach().cpu().tolist()
        total_true.extend(true_list)
        num_corrects = (torch.max(pred, 1)[1].view(label.size()).data == label.data).float().sum()
        pred_list = torch.max(pred, 1)[1].view(label.size()).data.detach().cpu().tolist()
        total_pred.extend(pred_list)

        acc = 100.0 * (num_corrects/train_batch_size)
        acc_curve.append(acc.item())
        total_epoch_acc += acc.item()

    return train_loss/len(train_loader), total_epoch_acc/len(train_loader), acc_curve


def test(epoch, test_loader, model_main, loss_function, log, stage=None):
    model_main.eval()
    total_epoch_acc = 0
    total_pred, total_true, total_pred_prob = [], [], []
    save_pred = {"true": [], "pred": [], "pred_prob": [], "feature": []}
    acc_curve = []
    total_feature = []
    test_acc = []
    test_tp = []
    tfn = []
    tpfn = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if "bpw" in log.param.dataset:
                tokenized_text_name = "tokenized_sentence"
                label_name = "label"
                text_name = "sentence"
                token_type_ids = "token_type_ids"
                graph_name = "sentence_dependency_matrix"

            text = batch[text_name]
            tokenized_text = batch[tokenized_text_name]
            attn = batch[tokenized_text_name + "_attn_mask"]
            label = batch[label_name]
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()

            if torch.cuda.is_available():
                # text = text.cuda()
                tokenized_text = tokenized_text.cuda()
                attn = attn.cuda()
                label = label.cuda()

            if log.param.task == "graph_steganalysis":
                token_type_ids = batch[token_type_ids]
                token_type_ids = token_type_ids.cuda()
                graph = batch[graph_name]
                graph = graph.cuda()
                pred, supcon_feature = model_main(inputs=text, tokenized_inputs=tokenized_text, graph=graph, attn_mask=attn, token_type_ids=token_type_ids, model_type=log.param.model_type)
            elif log.param.task == "stage":
                pred, supcon_feature = model_main(inputs=text, tokenized_inputs=tokenized_text, attn_mask=attn,
                                                  model_type=log.param.model_type, stage=None)
            else:
                pred, supcon_feature = model_main(inputs=text, tokenized_inputs=tokenized_text, attn_mask=attn,
                                                  model_type=log.param.model_type)

            num_corrects = (torch.max(pred, 1)[1].view(label.size()).data == label.data).float().sum()

            pred_list = torch.max(pred, 1)[1].view(label.size()).data.detach().cpu().tolist()
            true_list = label.data.detach().cpu().tolist()

            acc = 100.0 * num_corrects / 1
            acc_curve.append(acc.item())
            total_epoch_acc += acc.item()

            total_pred.extend(pred_list)
            total_true.extend(true_list)
            total_feature.extend(supcon_feature.data.detach().cpu().tolist())
            total_pred_prob.extend(pred.data.detach().cpu().tolist())

            y = pred.cpu().numpy()
            label = label.cpu().numpy()
            test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
            test_tp += [1 if np.argmax(y[i]) == label[i] and label[i] == 1 else 0 for i in range(len(y))]
            tfn += [1 if np.argmax(y[i]) == 1 else 0 for i in range(len(y))]
            tpfn += [1 if label[i] == 1 else 0 for i in range(len(y))]

    acc = np.mean(test_acc)
    tpsum = np.sum(test_tp)
    test_precision = tpsum / (np.sum(tfn) + 1e-5)
    test_recall = tpsum / np.sum(tpfn)
    test_Fscore = 2 * test_precision * test_recall / (test_recall + test_precision + 1e-10)

    f1_score_m = f1_score(total_true, total_pred, average="macro")
    f1_score_w = f1_score(total_true, total_pred, average="weighted")

    f1_score_all = {"macro": f1_score_m, "weighted": f1_score_w, "acc": acc, "precision": test_precision, "recall": test_recall, "Fscore": test_Fscore}

    save_pred["true"] = total_true
    save_pred["pred"] = total_pred

    save_pred["feature"] = total_feature
    save_pred["pred_prob"] = total_pred_prob

    return total_epoch_acc/len(test_loader), f1_score_all, save_pred, acc_curve


def stega_train(log):

    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)
    run_start = datetime.now()
    train_data, valid_data, test_data = get_dataloader(log.param.batch_size, log.param.corpus, log.param.stego_method, log.param.dataset, w_aug=log.param.is_waug, task=log.param.task, pkl_file = log.param.pkl_file)
    run_end = datetime.now()
    run_time = str((run_end - run_start).seconds / 60) + ' minutes'
    print("data process time: ", run_time)
    save_home = "./results/" + log.param.corpus + "-" + log.param.stego_method + "-" + log.param.dataset + "-" + log.param.loss_type + "-" + log.param.model_type

    if log.param.loss_type == "scl":
        losses = {"contrastive": loss.SupConLoss(temperature=log.param.temperature), "label": nn.CrossEntropyLoss(), "lambda_loss": log.param.lambda_loss}
    else:
        losses = {"label": nn.CrossEntropyLoss(), "lambda_loss": log.param.lambda_loss, "contrastive": loss.SupConLoss(temperature=log.param.temperature)}

    if log.param.model_type == "ls-cnn":
        model_main = cnn.TC()
    elif log.param.model_type == "rnn":
        model_main = rnn.TC()
    elif log.param.model_type == "bilstm_dense":
        model_main = bilstm_dense.TC()
    elif log.param.model_type == "r_bilstm_c":
        model_main = r_bilstm_c.TC()
    elif log.param.model_type == "sesy":
        if log.param.use_plm == True:
            model_main = sesy.BERT_TC()
        else:
            model_main = sesy.TC()
    elif log.param.model_type == "bert":
        model_main = bert.Bert()
    elif log.param.model_type == "scl":
        model_main = scl.ELECTRA()
    elif log.param.model_type == "GE":
        CELL = "bi-gru"  # rnn, bi-rnn, gru, bi-gru, lstm, bi-lstm
        EMBED_SIZE = 300  # 128
        HIDDEN_DIM = 100  # 256
        NUM_LAYERS = 2
        CLASS_NUM = 2
        DROPOUT_RATE = 0.2  # 0.2
        K = 10
        G = 10
        FILTER_NUM = 100
        FILTER_SIZE = [3, 5]
        model_main = GE.ge(
            cell = CELL,
            vocab_size = 30522,
            embed_size = EMBED_SIZE,
            filter_num = FILTER_NUM,
            filter_size = FILTER_SIZE,
            hidden_dim = HIDDEN_DIM,
            num_layers = NUM_LAYERS,
            class_num = CLASS_NUM,
            dropout_rate = DROPOUT_RATE,
            k = K,
            g = G,
        )
    elif log.param.model_type == "TDA-FSMS":
        model_main = SMSF.ETAFF(data=train_data.dataset.data)

    total_params = list(model_main.named_parameters())
    num_training_steps = int(len(train_data)*log.param.nepoch)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in total_params if not any(nd in n for nd in no_decay)], 'weight_decay': log.param.decay},
                                    {'params': [p for n, p in total_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=log.param.main_learning_rate)
    optimizer = torch.optim.Adam(model_main.parameters(), log.param.main_learning_rate, weight_decay = 1e-6)  # 优化函数
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    print("num_training_steps: ", num_training_steps)
    # logger.info('num_training_steps：{}'.format(num_training_steps))

    total_train_log, total_val_log, total_test_log = [], [], []

    early_stop = True
    for epoch in range(1, log.param.nepoch + 1):

        train_loss, train_acc, train_acc_curve = train(epoch, train_data, model_main, losses, optimizer, lr_scheduler, log)
        val_acc, val_f1, val_save_pred, val_acc_curve = test(epoch, valid_data, model_main, losses, log)
        test_acc, test_f1, test_save_pred, test_acc_curve = test(epoch, test_data, model_main, losses, log)

        total_train_log.extend([train_loss, train_acc])
        total_val_log.extend([val_acc, val_f1])
        total_test_log.extend([test_acc, test_f1])

        print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        # logger.info('====> Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        os.makedirs(save_home, exist_ok=True)
        with open(save_home+"/loss_acc.json", 'w') as fp:
            json.dump({"train_loss and train_acc": total_train_log, "val_acc and val_f1": total_val_log, "test_acc and test_f1": total_test_log}, fp, indent=4)
        fp.close()

        if epoch == 1:
            best_criterion = 0
        if val_acc <= best_criterion:
            early_stop = False
        is_best = val_acc > best_criterion
        best_criterion = max(val_acc, best_criterion)

        print(f'Valid Accuracy: {val_acc:.2f}  Valid F1: {val_f1["macro"]:.2f}')
        print(f'Test Accuracy: {test_acc:.2f}  Test F1: {test_f1["macro"]:.2f}')
        # logger.info(f'Valid Accuracy: {val_acc:.2f}  Valid F1: {val_f1["macro"]:.2f}')
        # logger.info(f'Test Accuracy: {test_acc:.2f}  Test F1: {test_f1["macro"]:.2f}')

        if is_best and early_stop:
            print("======> Best epoch <======")
            log.train_loss = train_loss
            log.stop_epoch = epoch
            log.valid_f1_score = val_f1
            log.test_f1_score = test_f1
            log.valid_accuracy = val_acc
            log.test_accuracy = test_acc
            log.train_accuracy = train_acc

            # save the model
            best_model = model_main
            torch.save(model_main.state_dict(), save_home+'/best.pt')
            run_end = datetime.now()
            best_time = str((run_end - run_start).seconds / 60) + ' minutes'
            log.best_time = best_time

            with open(save_home+"/log.json", 'w') as fp:
                json.dump(dict(log), fp, indent=4)
            fp.close()

            with open(save_home+"/feature.json", 'w') as fp:
                json.dump(test_save_pred, fp, indent=4)
            fp.close()

        run_end = datetime.now()
        run_time = str((run_end - run_start).seconds / 60) + ' minutes'
        print("corpus: ", log.param.corpus, "stego_method: ", log.param.stego_method, "dataset: ", log.param.dataset, "model_type: ", log.param.model_type, "run_time: ", run_time)

if __name__ == '__main__':

    tuning_param = train_config.tuning_param
    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = train_config.param

        for num, val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        if "allbpw" in log.param.dataset:
            log.param.label_size = 6
        elif "bpw" in log.param.dataset:
            log.param.label_size = 2

        stega_train(log)

