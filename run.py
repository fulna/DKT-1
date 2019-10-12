import math

import numpy as np
import torch
from sklearn import metrics
from torch import nn

import utils as utils


def train(model, epoch, params, optimizer, q_data, q_target_data, qa_data):
    N = int(math.floor(len(q_data) / params.batch_size))

    # shuffle data
    shuffle_index = np.random.permutation(q_data.shape[0])
    q_data = q_data[shuffle_index] # 用来Input的pair
    q_target_data = q_target_data[shuffle_index] # 下一题题号
    qa_data = qa_data[shuffle_index] # 下一题pair

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    for idx in range(N):
        q_target_seq = q_target_data[idx * params.batch_size:(idx + 1) * params.batch_size, :] # 下一题题号
        qa_batch_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :] # 用于input
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :] # 下一题答案

        target = (target - 1) / params.n_question
        target = np.floor(target)# 1是正确，等于0是错误，-1是填充的0补全的位置
        input_q_target = utils.variable(torch.LongTensor(q_target_seq), params.gpu)
        input_x = utils.variable(torch.LongTensor(qa_batch_seq), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)
        input_q_target_to_1d = torch.chunk(input_q_target, params.batch_size, 0)
        input_q_target_1d = torch.cat([input_q_target_to_1d[i] for i in range(params.batch_size)], 1)
        input_q_target_1d = input_q_target_1d.permute(1, 0)

        model.zero_grad()
        loss, filtered_pred, filtered_target = model(input_x, input_q_target_1d, target_1d) # 答案编码后10x800,题号8000x1，答案8000x1
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    if (epoch + 1) % params.decay_epoch == 0:
        new_lr = params.init_lr * params.lr_decay
        if new_lr < params.final_lr:
            new_lr = params.final_lr
        utils.adjust_learning_rate(optimizer, new_lr)
    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc


def test(model, params, optimizer, q_data, q_target_data, qa_data):
    N = int(math.floor(len(q_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()

    for idx in range(N):
        q_target_seq = q_target_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]

        target = (target - 1) / params.n_question
        target = np.floor(target)
        input_q_target = utils.variable(torch.LongTensor(q_target_seq), params.gpu)
        input_x = utils.variable(torch.LongTensor(qa_batch_seq), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)
        input_q_target_to_1d = torch.chunk(input_q_target, params.batch_size, 0)
        input_q_target_1d = torch.cat([input_q_target_to_1d[i] for i in range(params.batch_size)], 1)
        input_q_target_1d = input_q_target_1d.permute(1, 0)

        loss, filtered_pred, filtered_target = model(input_x, input_q_target_1d, target_1d)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc
