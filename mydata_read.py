#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
import h5py
# 原始文件中是在datasets.py中读取数据的，分别读取测试集和训练集
import random
import json
import os
from pathlib import Path

def _check_path(p, verbose=True):
    full = os.path.abspath(p)
    if verbose:
        print(f"[DATA] trying to read: {full}")
    if not os.path.exists(full):
        raise FileNotFoundError(f"File not found: {full}")
    return full


def load_npy(x_path, y_path):
    """
    保持兼容的旧接口：直接从 .npy 文件加载 (N, L, 2) + (N,) 形式的数据。
    """
    x_path = Path(x_path)
    y_path = Path(y_path)

    _check_path(x_path)
    _check_path(y_path)

    X = np.load(x_path).astype(np.float32)
    Y = np.load(y_path).astype(np.int64)

    if X.ndim != 3 or X.shape[2] != 2:
        raise RuntimeError(f"期望 X 的形状为 (N, L, 2)，但得到 {X.shape}")

    # 转换为 (N, 2, L) 以匹配 Conv1d 的输入格式
    X = np.transpose(X, (0, 2, 1))

    if Y.ndim != 1:
        Y = Y.reshape(-1)

    if len(X) != len(Y):
        raise RuntimeError(f"数据与标签长度不一致: X={len(X)}, Y={len(Y)}")

    index = [i for i in range(len(Y))]
    random.shuffle(index)
    data = X[index, :, :]
    label = Y[index]

    print(json.dumps({"str": f"读取 .npy 数据: X{X.shape}, Y{Y.shape}"}, ensure_ascii=False))
    print(json.dumps({"str": f"打乱数据集：{data.shape}"}, ensure_ascii=False))
    print(json.dumps({"str": f"打乱标签：{label.shape}"}, ensure_ascii=False))

    return data, label


def load_samples_from_list(list_path, base_dir=None):
    """
    从标注列表读取逐样本的 .npy 数据路径与标签（仅解析，不一次性加载全部数据）。

    参数:
        list_path: 包含“样本路径 + 标签”的 txt。每行形如：
                   <path_to_sample.npy> <label>
                   （路径中可能有空格，因此使用 rsplit 拆分标签）
        base_dir : 可选，将列表中的相对路径拼接在该目录下。

    返回:
        entries: List[Tuple[pathlib.Path, int]]
    """
    list_path = Path(list_path)
    _check_path(list_path)

    entries = []
    with list_path.open('r', encoding='utf-8') as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            if ' ' not in line:
                raise RuntimeError(f"第 {lineno} 行缺少标签，请确保格式为 '<path> <label>'")

            sample_str, label_str = line.rsplit(maxsplit=1)
            sample_path = Path(sample_str)
            if base_dir is not None and not sample_path.is_absolute():
                sample_path = Path(base_dir) / sample_path

            entries.append((sample_path, int(label_str)))

    if not entries:
        raise RuntimeError("未在列表中读取到任何样本")

    print(json.dumps({"str": f"读取样本列表：共 {len(entries)} 条"}, ensure_ascii=False))
    return entries



def load_h4(h5_path):
    # load training data  读取数据并处理
    h5_path = _check_path(h5_path)
    with h5py.File(h5_path, 'r') as hf:
        head = list(hf.keys())
        # print('List of arrays in input file:', hf.keys())
        X = np.transpose(np.array(hf.get(head[0]), dtype=np.float32))
        Y = np.transpose(np.array(hf.get(head[1]), dtype=np.float32))
        if X.ndim == 3:
            X1 = X.swapaxes(1, 2) # 将数组n个维度中两个维度进行调换
            # X1 = X
            Y1 = Y.astype(np.float32)
        elif Y.ndim == 3:
            X1 = Y.swapaxes(1, 2)  # 将数组n个维度中两个维度进行调换
            # X1 = X
            Y1 = X.astype(np.float32)
        else:
            raise RuntimeError("维度错误")

        X1 = np.expand_dims(X1, axis=0)
        X1 = X1.swapaxes(0, 1)
        X1 = X1.swapaxes(1, 2)
        Y1 = Y

        index = [i for i in range(len(Y1))]
        # random.seed(10)
        random.shuffle(index)
        data = X1[index,:,:]
        label = Y1[index]
        # print("打乱数据集",data.shape)
        # print("打乱数据集",label.shape)
        # print("打乱数据集",label[:6])
        txdata4 = {
            "str": "读取文件：" +  str(hf.keys())
        }
        json_string4 = json.dumps(txdata4, ensure_ascii=False)
        print(json_string4)
        txdata4 = {
            "str": "打乱数据集：" + str(data.shape)
        }
        json_string4 = json.dumps(txdata4, ensure_ascii=False)
        print(json_string4)
        txdata4 = {
            "str": "打乱标签：" + str(label.shape)
        }
        json_string4 = json.dumps(txdata4, ensure_ascii=False)
        print(json_string4)
    return data, label


def _load_single_sample(sample_path):
    sample_np = np.load(_check_path(sample_path, verbose=False)).astype(np.float32)
    if sample_np.shape == (240000, 2):
        sample_np = sample_np.T
    if sample_np.ndim != 2 or sample_np.shape[0] != 2:
        raise RuntimeError(f"样本 {sample_path} 维度异常: {sample_np.shape}")
    return sample_np
def load_h5(h5_path):
    # load training data  读取数据并处理
    h5_path = _check_path(h5_path)
    with h5py.File(h5_path, 'r') as hf:
        head = list(hf.keys())
        # print('List of arrays in input file:', hf.keys())
        X = np.transpose(np.array(hf.get(head[0]), dtype=np.float32))
        Y = np.transpose(np.array(hf.get(head[1]), dtype=np.float32))
        if X.ndim == 3:
            X1 = X.swapaxes(1, 2) # 将数组n个维度中两个维度进行调换
            # X1 = X
            Y1 = Y.astype(np.float32)
        elif Y.ndim == 3:
            X1 = Y.swapaxes(1, 2)  # 将数组n个维度中两个维度进行调换
            # X1 = X
            Y1 = X.astype(np.float32)
        else:
            raise RuntimeError("维度错误")

        index = [i for i in range(len(Y1))]
        # random.seed(10)
        random.shuffle(index)
        data = X1[index,:,:]
        label = Y1[index]
        # print("打乱数据集",data.shape)
        # print("打乱数据集",label.shape)
        # print("打乱数据集",label[:6])
        txdata5 = {
            "str": "读取文件：" + str(hf.keys())
        }
        json_string5 = json.dumps(txdata5, ensure_ascii=False)
        print(json_string5)
        txdata5 = {
            "str": "打乱数据集：" + str(data.shape)
        }
        json_string5 = json.dumps(txdata5, ensure_ascii=False)
        print(json_string5)
        txdata5 = {
            "str": "打乱标签：" + str(label.shape)
        }
        json_string5 = json.dumps(txdata5, ensure_ascii=False)
        print(json_string5)
    return data, label

def load_h6(h5_path):
    # load training data  读取数据并处理
    h5_path = _check_path(h5_path)
    with h5py.File(h5_path, 'r') as hf:
        head = list(hf.keys())
        print('List of arrays in input file:', hf.keys())
        X = np.transpose(np.array(hf.get(head[0]), dtype=np.float32))
        Y = np.transpose(np.array(hf.get(head[1]), dtype=np.float32))
        if X.ndim == 3:
            X1 = X.swapaxes(1, 2) # 将数组n个维度中两个维度进行调换
            Y1 = Y.astype(np.float32)
        elif Y.ndim == 3:
            X1 = Y.swapaxes(1, 2)  # 将数组n个维度中两个维度进行调换
            Y1 = X.astype(np.float32)
        else:
            raise RuntimeError("维度错误")
        m = 800
        # list = range(1000)
        import random
        # print()  random.sample(list,m)
        Empty = np.zeros([ m*100 ,2 , 2196], dtype=np.float32)
        Empty_y = np.zeros([m*100, 1], dtype=np.float32)
        for n in range(100):
            # X = X1[1000*n+random.sample(list,m),:,:]
            # for j in range(m):
            # Empty[m*n:m-1+m*n,:,:] = X1.take(1000*n+random.sample(list,m))
            # Empty_y[m*n:m-1+m*n] = Y1.take(1000*n+random.sample(list,m))
            Empty[m*n:m-1+m*n,:,:] = X1[1000*n:m-1+1000*n,:,:]
            Empty_y[m*n:m-1+m*n] = Y1[1000*n:m-1+1000*n]
        index = [i for i in range(len(Empty_y))]
        random.shuffle(index)
        data = Empty[index,:,:]
        label = Empty_y[index]
        print("打乱数据集",data.shape)
        print("打乱数据集",label.shape)
    return data, label

class SignalDataset(Dataset):
    """数据加载器  返回指定数目的数据"""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.sample_entries = None
        if isinstance(data_folder, (tuple, list)) and len(data_folder) == 2:
            pair_second = Path(data_folder[1])
            if pair_second.suffix.lower() == '.txt':
                self.sample_entries = load_samples_from_list(pair_second, base_dir=data_folder[0])
                self.X, self.Y = None, None
            else:
                self.X, self.Y = load_npy(data_folder[0], data_folder[1])
        else:
            self.X, self.Y = load_h6(data_folder)   # (3392, 8192, 1)

    def __getitem__(self, item):
        # 返回一个音频数据
        if self.sample_entries is not None:
            sample_path, label = self.sample_entries[item]
            X = _load_single_sample(sample_path)
            return X, label
        X = self.X[item]
        Y = self.Y[item]
        return X, Y
    def __len__(self):
        return len(self.sample_entries) if self.sample_entries is not None else len(self.X)


def load_h5_2D(h5_path):
    h5_path = _check_path(h5_path)
    # load training data  读取数据并处理
    with h5py.File(h5_path, 'r') as hf:
        head = list(hf.keys())
        print('List of arrays in input file:', hf.keys())
        X = np.transpose(np.array(hf.get(head[0]), dtype=np.float32))
        Y = np.transpose(np.array(hf.get(head[1]), dtype=np.float32))
        if X.ndim == 3:
            X1 = X.swapaxes(1, 2) # 将数组n个维度中两个维度进行调换
            X1 = X1[:, :, np.newaxis, :]
            Y1 = Y.astype(np.float32)
        elif Y.ndim == 3:
            X1 = Y.swapaxes(1, 2)  # 将数组n个维度中两个维度进行调换
            X1 = X1[:, :, np.newaxis, :]
            Y1 = X.astype(np.float32)
        else:
            raise RuntimeError("维度错误")

        index = [i for i in range(len(Y1))]
        # random.seed(10)
        random.shuffle(index)
        data = X1[index,:,:,:]
        label = Y1[index]
        print("打乱数据集",data.shape)
        print("打乱数据集",label.shape)
        # print("打乱数据集",label[:6])
    return X1, Y1

class SignalDataset1(Dataset):
    """数据加载器"""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.sample_entries = None
        if isinstance(data_folder, (tuple, list)) and len(data_folder) == 2:
            pair_second = Path(data_folder[1])
            if pair_second.suffix.lower() == '.txt':
                self.sample_entries = load_samples_from_list(pair_second, base_dir=data_folder[0])
                self.X, self.Y = None, None
            else:
                self.X, self.Y = load_npy(data_folder[0], data_folder[1])
        else:
            self.X, self.Y = load_h4(data_folder)   # (3392, 8192, 1)
        # self.X, self.Y = load_h5(data_folder)  # (3392, 8192, 1)

    def __getitem__(self, item):
        # 返回一个音频数据
        if self.sample_entries is not None:
            sample_path, label = self.sample_entries[item]
            X = _load_single_sample(sample_path)
            return X, label
        X = self.X[item]
        Y = self.Y[item]
        return X, Y
    def __len__(self):
        return len(self.sample_entries) if self.sample_entries is not None else len(self.X)

class SignalDataset2(Dataset):
    """数据加载器"""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.sample_entries = None
        if isinstance(data_folder, (tuple, list)) and len(data_folder) == 2:
            pair_second = Path(data_folder[1])
            if pair_second.suffix.lower() == '.txt':
                self.sample_entries = load_samples_from_list(pair_second, base_dir=data_folder[0])
                self.X, self.Y = None, None
            else:
                self.X, self.Y = load_npy(data_folder[0], data_folder[1])
        else:
            self.X, self.Y = load_h5(data_folder)  # (3392, 8192, 1)

    def __getitem__(self, item):
        # 返回一个音频数据
        if self.sample_entries is not None:
            sample_path, label = self.sample_entries[item]
            X = _load_single_sample(sample_path)
            return X, label
        X = self.X[item]
        Y = self.Y[item]
        return X, Y
    def __len__(self):
        return len(self.sample_entries) if self.sample_entries is not None else len(self.X)
class SignalDataset2D(Dataset):
    """数据加载器"""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.sample_entries = None
        if isinstance(data_folder, (tuple, list)) and len(data_folder) == 2:
            pair_second = Path(data_folder[1])
            if pair_second.suffix.lower() == '.txt':
                self.sample_entries = load_samples_from_list(pair_second, base_dir=data_folder[0])
                self.X, self.Y = None, None
            else:
                self.X, self.Y = load_npy(data_folder[0], data_folder[1])
        else:
            self.X, self.Y = load_h5_2D(data_folder)   # (3392, 8192, 1)
    def __getitem__(self, item):
        if self.sample_entries is not None:
            sample_path, label = self.sample_entries[item]
            X = _load_single_sample(sample_path)
            return X, label
        X = self.X[item]
        Y = self.Y[item]
        return X, Y
    def __len__(self):
        return len(self.sample_entries) if self.sample_entries is not None else len(self.X)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X, Y = load_h6('Case0_7.68M/Case0_7.68M_TX_100X1000_15dB_2196_withoutCFO.mat')
    print(Y[90:110])
    # X, Y = load_h5('Case0_7.68M_LOS_20dB.mat')  #(3845, 2, 4800)
    # # 信号长度变化
    # # len1 = np.random.randint(2400,4800)
    # # data1 = np.zeros([X.shape[0], X.shape[1], X.shape[2]])
    # #
    # # x_resampled = signal.resample(X[1,1,:], len1) # 函数使用FFT将信号重采样成n个点
    # # print(x_resampled.shape)
    # # 求取FFT
    # print(X.shape)
    # fig = plt.figure()
    # plt.subplot(4,1,1)
    # plt.plot(X[2,0,:],label="I")
    # plt.legend()
    # plt.subplot(4, 1, 2)
    # plt.plot(X[2,1,:],label="Q")
    # plt.legend()
    # plt.subplot(4, 1, 3)
    # plt.plot(X[2,2,:],label="FFT_I_abs")
    # plt.legend()
    # plt.subplot(4, 1, 4)
    # plt.plot(X[2,3,:],label="FFT_I_angle")
    # plt.legend()
    #
    # fig = plt.figure()
    # plt.subplot(4,1,1)
    # plt.plot(X[999,0,:],label="I")
    # plt.legend()
    # plt.subplot(4, 1, 2)
    # plt.plot(X[999,1,:],label="Q")
    # plt.legend()
    # plt.subplot(4, 1, 3)
    # plt.plot(X[999,2,:],label="FFT_I_abs")
    # plt.legend()
    # plt.subplot(4, 1, 4)
    # plt.plot(X[999,3,:],label="FFT_I_angle")
    # plt.legend()
    #
    # plt.show()
