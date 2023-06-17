# !/usr/bin/python3
# --coding:utf-8--
# @File: motif_score_base.py
# @Author:junru jin
# @Time: 2022年12月 03日21
# @description:

import torch
import os
import numpy as np

from tqdm import tqdm
from plot import plot_seaborn, plot_box
from interface import newModel, genData, load_params, read_sequence, aa_dict

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda", 0)

if __name__ == '__main__':
    # 定义模型
    model = newModel()

    # 加载参数
    load_params(model, '/mnt/solid/lzs/SSP_prediction/Model/not_bn_interpret_model, epoch[20], ACC[0.997].pt')
    model.to(device).eval()

    # 获取氨基酸列表，用于替换氨基酸，去除了X
    aa_list = list(aa_dict.keys())
    aa_list.remove('X')
    for i in range(len(aa_list)):
        print(i, aa_list[i])

    # 选取某一个样本序列来进行实验
    positive_sequences, negative_sequences = read_sequence('./SSP_dataset.csv')

    cut_number = 25
    sequences_acid_value = [[] for i in range(cut_number)]

    # positive_sequences = [negative_sequences[1]]
    save_data = []
    iter_sequence = negative_sequences
    # iter_sequence = positive_sequences
    for k in tqdm(range(len(iter_sequence))):
        # 生成原始序列的数据
        positive_sequence = iter_sequence[k]
        result_matrix = np.zeros((len(aa_list), len(positive_sequence)))

        sequence = genData(positive_sequence)
        sequence = sequence.unsqueeze(0).to(device)
        # 生成原始序列的预测结果
        with torch.no_grad():
            output = model.trainModel(sequence)

        output = torch.softmax(output, dim=1)[0][1].item()

        # 生成替换后的序列的数据
        for i in range(len(positive_sequence)):
            for j in range(len(aa_list)):
                sequence_new = positive_sequence[:i] + aa_list[j] + positive_sequence[i + 1:]
                sequence_new = genData(sequence_new)
                sequence_new = sequence_new.unsqueeze(0).to(device)

                with torch.no_grad():
                    output_modification = model.trainModel(sequence_new)
                output_modification = torch.softmax(output_modification, dim=1)[0][1].item()

                # 计算差异
                # diff = output_modification - output
                diff = abs(output_modification - output)
                result_matrix[j][i] = diff
        save_data.append([result_matrix, iter_sequence[k]])

    # path = './result/'
    # if os.path.exists(path) is False:
    #     os.mkdir(path)
    torch.save(save_data, './save_data/abs_negative.pt')
