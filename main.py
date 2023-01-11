import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def generate_data(file):
    # Amino acid dictionary
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}

    # Secondary structure dictionary
    ss_dict = {'C': 1, 'H': 2, 'E': 3}

    with open(file, 'r') as inf:
        lines = inf.read().splitlines()

    pep_codes = []
    labels = []
    peps = []
    secondary_strus_codes = []
    for pep in lines:
        pep, label, secondary_struct = pep.split(",")
        peps.append(pep)
        labels.append(int(label))
        current_pep = []
        for aa in pep:
            current_pep.append(aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep))

        # Adding secondary structure
        current_ss = []
        for s in secondary_struct:
            current_ss.append(ss_dict[s])
        secondary_strus_codes.append(torch.tensor(current_ss))

    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)  # Fill the sequence to the same length
    data_ss = rnn_utils.pad_sequence(secondary_strus_codes, batch_first=True)

    return data, torch.tensor(labels), data_ss


data, label, data_ss = generate_data("./dataset/SSP_dataset.csv")

train_data, train_label, train_ss = data[:1894], label[:1894], data_ss[:1894]
test_data, test_label, test_ss = data[1894:], label[1894:], data_ss[1894:]

train_dataset = Data.TensorDataset(train_data, train_label, train_ss)
test_dataset = Data.TensorDataset(test_data, test_label, test_ss)

batch_size = 256
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class ExamPle(nn.Module):
    def __init__(self, vocab_size=24):
        super().__init__()
        self.hidden_dim = 25
        self.batch_size = 32
        self.emb_dim = 512

        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)

        # Pass the sequence information and secondary structure information through different transformer encoders
        self.transformer_encoder_seq = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.transformer_encoder_ss = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.2)

        # Reduce the dimension of the embedding
        self.block_seq = nn.Sequential(nn.Linear(15050, 2048),
                                       nn.BatchNorm1d(2048),
                                       nn.LeakyReLU(),
                                       nn.Linear(2048, 1024))

        self.block_ss = nn.Sequential(nn.Linear(15050, 2048),
                                      nn.BatchNorm1d(2048),
                                      nn.LeakyReLU(),
                                      nn.Linear(2048, 1024))

        self.block1 = nn.Sequential(nn.Linear(2048, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 256))

        self.block2 = nn.Sequential(nn.Linear(256, 8),
                                    nn.ReLU(),
                                    nn.Linear(8, 2))

    def forward(self, x, ss):
        x = self.embedding(x)
        output = self.transformer_encoder_seq(x).permute(1, 0, 2)
        output, hn = self.gru(output)
        output = output.permute(1, 0, 2)
        hn = hn.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)
        hn = hn.reshape(output.shape[0], -1)
        output = torch.cat([output, hn], 1)
        output = self.block_seq(output)

        # Process the secondary structure information
        ss = self.embedding(ss)
        ss_output = self.transformer_encoder_ss(ss).permute(1, 0, 2)
        ss_output, ss_hn = self.gru(ss_output)
        ss_output = ss_output.permute(1, 0, 2)
        ss_hn = ss_hn.permute(1, 0, 2)
        ss_output = ss_output.reshape(ss_output.shape[0], -1)
        ss_hn = ss_hn.reshape(ss_output.shape[0], -1)
        ss_output = torch.cat([ss_output, ss_hn], 1)
        ss_output = self.block_ss(ss_output)

        # Fusion of features
        representation = torch.cat([output, ss_output], dim=1)

        return self.block1(representation)

    def train_model(self, x, ss):
        with torch.no_grad():
            output = self.forward(x, ss)
        return self.block2(output)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []

    secondarystru1_ls = []
    secondarystru2_ls = []

    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        seq1, label1, secondarystru1 = batch[i][0], batch[i][1], batch[i][2]
        seq2, label2, secondarystru2 = batch[i + int(batch_size / 2)][0], \
                                       batch[i + int(batch_size / 2)][1], \
                                       batch[i + int(batch_size / 2)][2]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label = (label1 ^ label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))

        secondarystru1_ls.append(secondarystru1.unsqueeze(0))
        secondarystru2_ls.append(secondarystru2.unsqueeze(0))

    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)

    ss1 = torch.cat(secondarystru1_ls).to(device)
    ss2 = torch.cat(secondarystru2_ls).to(device)

    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return seq1, seq2, label, label1, label2, ss1, ss2


train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, collate_fn=collate)

device = torch.device("cuda", 0)


def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    for x, y, ss in data_iter:
        x, y, ss = x.to(device), y.to(device), ss.to(device)
        outputs = net.train_model(x, ss)
        outputs_cpu = outputs.cpu()
        y_cpu = y.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data


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

    # Accuracy
    ACC = float(tp + tn) / test_num

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

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data


def to_log(log):
    with open("./results/ExamPle_Log.log", "a+") as f:
        f.write(log + '\n')


net = ExamPle().to(device)
lr = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = ContrastiveLoss()
criterion_model = nn.CrossEntropyLoss(reduction='sum')
best_acc = 0
EPOCH = 30
for epoch in range(EPOCH):
    loss_ls = []
    loss1_ls = []
    loss2_3_ls = []
    t0 = time.time()
    net.train()
    for seq1, seq2, label, label1, label2, ss1, ss2 in train_iter_cont:
        output1 = net(seq1, ss1)
        output2 = net(seq2, ss2)
        output3 = net.train_model(seq1, ss1)
        output4 = net.train_model(seq2, ss2)
        loss1 = criterion(output1, output2, label)
        loss2 = criterion_model(output3, label1)
        loss3 = criterion_model(output4, label2)
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ls.append(loss.item())
        loss1_ls.append(loss1.item())
        loss2_3_ls.append((loss2 + loss3).item())

    net.eval()
    with torch.no_grad():
        train_performance, train_roc_data, train_prc_data = evaluate(train_iter, net)
        test_performance, test_roc_data, test_prc_data = evaluate(test_iter, net)

    results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
    results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
    results += '\n' + '=' * 16 + ' Test Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
               + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        test_performance[0], test_performance[1], test_performance[2], test_performance[3],
        test_performance[4]) + '\n' + '=' * 60
    print(results)
    to_log(results)
    test_acc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
    if test_acc > best_acc:
        best_acc = test_acc
        best_performance = test_performance
        filename = '{}, {}[{:.3f}].pt'.format('SSPformer' + ', epoch[{}]'.format(epoch + 1), 'ACC', best_acc)
        save_path_pt = os.path.join('./Model', filename)
        torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
        best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(epoch + 1) + '=' * 16 \
                       + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            best_performance[0], best_performance[1], best_performance[2], best_performance[3],
            best_performance[4]) \
                       + '\n' + '=' * 60
        print(best_results)
        best_ROC = test_roc_data
        best_PRC = test_prc_data
