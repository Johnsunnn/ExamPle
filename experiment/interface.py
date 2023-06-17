# !/usr/bin/python3
# --coding:utf-8--
# @File: motif_extraction_base.py
# @Author:junru jin
# @Time: 2022年12月 02日20
# @description:

import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from keras_preprocessing.sequence import pad_sequences

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

device = torch.device("cuda", 0)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
           'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
           'W': 20, 'Y': 21, 'V': 22, 'X': 23}


def genData(pep):
    pep_codes = []
    for aa in pep:
        pep_codes.append(aa_dict[aa])

    pep_codes = pad_sequences([pep_codes], maxlen=299, padding='post', dtype=int)[0]
    # pep_codes = pad_sequences([pep_codes], maxlen=299, padding='post', dtype=int)[0]
    # data = rnn_utils.pad_sequence([pep_codes])  # 把所有序列弄成一样长度
    # print(data)
    pep_codes = torch.LongTensor(pep_codes)
    # print(pep_codes)

    return pep_codes


class newModel(nn.Module):
    def __init__(self, vocab_size=24):
        super().__init__()
        self.hidden_dim = 25
        self.batch_size = 32
        self.emb_dim = 512

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)

        self.transformer_encoder_seq = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.2)

        # 对embedding进行降维先
        self.block_seq = nn.Sequential(nn.Linear(15050, 2048),
                                       nn.LeakyReLU(),
                                       nn.Linear(2048, 1024),
                                       )

        self.block1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
        )

        self.block2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

        self.rnn = torch.nn.GRU(input_size=128,
                                hidden_size=512,
                                num_layers=2,
                                bidirectional=True,
                                batch_first=True,
                                dropout=0.2
                                )

        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer_encoder_seq(x).permute(1, 0, 2)
        output, hn = self.gru(output)
        output = output.permute(1, 0, 2)
        hn = hn.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)
        hn = hn.reshape(output.shape[0], -1)
        output = torch.cat([output, hn], 1)  # 这里是最终的GRU模块的输出
        output = self.block_seq(output)
        # 归一化
        output = self.layer_norm(output)
        output = self.block1(output)

        return output

    def trainModel(self, x):
        with torch.no_grad():
            output = self.forward(x)
        return self.block2(output)


def load_params(model, param_path):
    pretrained_dict = torch.load(param_path)
    new_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)

    return model


def read_sequence(path='./SSP_dataset.csv'):
    with open(path, 'r') as f:
        lines = f.readlines()
    positive_sequence, negative_sequence = [], []
    for line in lines:
        pep, label, secondary_stru = line.split(",")
        if label == '1':
            positive_sequence.append(pep)
        elif label == '0':
            negative_sequence.append(pep)

    return positive_sequence, negative_sequence


# def read_sequence(path='./SSP_dataset.csv'):
#     with open(path, 'r') as f:
#         lines = f.readlines()
#     positive_sequence, negative_sequence = [], []
#     for line in lines:
#         pep, label, secondary_stru = line.split(",")
#         if label == '1':
#             positive_sequence.append(pep)
#         elif label == '0':
#             negative_sequence.append(pep)
#
#     return positive_sequence, negative_sequence
def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # print('tp\tfp\ttn\tfn')
    # print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Precision
    # if tp + fp == 0:
    #     Precision = 0
    # else:
    #     Precision = float(tp) / (tp + fp)

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    # if Recall + Precision == 0:
    #     F1 = 0
    # else:
    #     F1 = 2 * Recall * Precision / (Recall + Precision)

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)  # Default 1 is positive sample
    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    # ROC(FPR, TPR, AUC)
    # PRC(Recall, Precision, AP)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data


if __name__ == '__main__':

    # 定义模型
    model = newModel()

    # 加载参数
    load_params(model, '/mnt/solid/lzs/SSP_prediction/Model/interpret, epoch[10], ACC[0.985].pt')
    model.to(device).eval()

    # 获取氨基酸列表，用于替换氨基酸，去除了X
    aa_list = list(aa_dict.keys())
    aa_list.remove('X')

    # 选取某一个样本序列来进行实验
    data = 'MVKTLKFVYDMILFIFLYLVAKNVAESIECRTVADCPKLISSKFVIKCIEKRCVAQFFD'

    # 定义结果矩阵储存得分
    result_matrix = np.zeros((len(aa_list), len(data)))

    data = list(data)

    # 遍历序列每一个位置
    for position in tqdm(range(len(data))):

        # 每一个位置上都会被替换成所有种类的氨基酸来计算得分
        for aa_type in range(len(aa_list)):
            data[position] = aa_list[aa_type]
            pep = "".join(data)
            pep = genData(pep)
            pep = pep.to(device)
            pep = pep.unsqueeze(0)
            output = model.trainModel(pep)[0]
            output = output.cpu()
            output = F.softmax(output, dim=0)
            pred_prob_positive = output[1]  # 是正例的概率

            # 把结果加到热力图矩阵里
            result_matrix[aa_type][position] = pred_prob_positive

    # 储存结果矩阵来画图
    f = open('./interpret_matrix.pkl', 'wb')
    pickle.dump(result_matrix, f)
    f.close()
