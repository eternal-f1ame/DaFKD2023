import json
import logging
import os
import random
import h5py
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import logging
import torch.utils.data as data


def read_data(train_data_dir, test_data_dir):
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = sorted(cdata['users'])

    return clients, groups, train_data, test_data


def batch_data(data, batch_size, model_name):

    data_x = data['x']
    data_y = data['y']

    if model_name != "lr":
        data_x = np.array(data_x).reshape((-1, 1, 28, 28))

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    data_x = np.where(data_x > 0, 1, 0)
    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def non_iid_partition_with_dirichlet_distribution(label_list,
                                                  client_num,
                                                  classes,
                                                  alpha,
                                                  task='classification'):
   
    net_dataidx_map = {}
    K = classes
    # For multiclass labels, the list is ragged and not a numpy array
    N = len(label_list)

    # guarantee the minimum number of sample in each client
    min_size = 0
    while min_size < 100:
        # logging.debug("min_size = {}".format(min_size))
        idx_batch = [[] for _ in range(client_num)]

        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(label_list == k)[0]
            idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                      idx_batch, idx_k)
    for i in range(client_num):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def noniid_merge_data_with_dirichlet_distribution(client_num_in_total, train_data, test_data, alpha, class_num=10):
    new_users = []
    new_train_data = {}
    new_test_data = {}

    all_distillation_data = {"x": [], "y": []}
    new_distillation_data = {}
    length_train = len(train_data)
    length_test = len(test_data)
    # alpha = 1

    for i in range(client_num_in_total):
        if i < 10:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))

    count1 = 0
    all_train_data = {"x": [], "y": []}
    for (_, value) in train_data.items():
        count1 += 1
        if count1 / length_train < 0.5:
            all_train_data["x"] += value["x"]
            all_train_data["y"] += value["y"]
        else:
            all_distillation_data["x"] += value["x"]
            all_distillation_data["y"] += value["y"]

    train_label_list = np.asarray(all_train_data["y"])
    train_idx_map = non_iid_partition_with_dirichlet_distribution(train_label_list, client_num_in_total, class_num,
                                                                  alpha)
    for index, idx_list in train_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_train_data["x"][i] for i in idx_list],
                     "y": [all_train_data["y"][i] for i in idx_list]}
        new_train_data[key] = temp_data

    count2 = 0
    all_test_data = {"x": [], "y": []}
    for (_, value) in test_data.items():
        count2 += 1
        if count2 / length_test < 1:
            all_test_data["x"] += value["x"]
            all_test_data["y"] += value["y"]
        else:
            all_distillation_data["x"] += value["x"]
            all_distillation_data["y"] += value["y"]
 
    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)
    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data

    return new_users, new_train_data, new_test_data

def load_partition_data_mnist(batch_size,
                                client_num_in_total,
                                model_name,
                                alpha,
                                data_dir="data/MNIST/",
                                ):
    class_num = 10

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return data

    new_users = []
    groups=[]
    new_train_data = {}
    all_train_data = {"x": [], "y": []}
    new_test_data = {}
    all_test_data = {"x": [], "y": []}
    for i in range(client_num_in_total):
        if i < class_num:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))

    data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))
 
    data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))
 
    data = extract_data(data_dir + 't10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))
 
    data = extract_data(data_dir + 't10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trX = np.asarray(trX).tolist()
    teX = np.asarray(teX).tolist()
    trY = np.asarray(trY).tolist()
    teY = np.asarray(teY).tolist()

    for i in range(len(trX)):
        all_train_data['x'].append(remake_fashion_mnist(trX[i]))
        all_train_data['y'].append(trY[i])

    train_label_list = np.asarray(all_train_data["y"])
    train_idx_map = non_iid_partition_with_dirichlet_distribution(train_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in train_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_train_data["x"][i] for i in idx_list],
                     "y": [all_train_data["y"][i] for i in idx_list]}
        new_train_data[key] = temp_data

    for i in range(len(teX)):
        all_test_data['x'].append(remake_fashion_mnist(teX[i]))
        all_test_data['y'].append(teY[i])

    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data        

    if len(groups) == 0:
        groups = [None for _ in new_users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    distillation_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(new_users, groups):
        user_train_data_num = len(new_train_data[u]['x'])
        user_test_data_num = len(new_test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(new_train_data[u], batch_size, model_name)
        test_batch = batch_data(new_test_data[u], batch_size, model_name)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_batch), len(test_batch)))
        client_idx += 1

    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def load_partition_distillation_data_mnist(batch_size,
                                client_num_in_total,
                                model_name,
                                alpha,
                                data_dir="data/MNIST/",
                                ):
    class_num = 10

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return data

    new_users = []
    groups=[]
    new_train_data = {}
    all_train_data = {"x": [], "y": []}
    new_test_data = {}
    all_test_data = {"x": [], "y": []}
    for i in range(client_num_in_total):
        if i < class_num:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))

    data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))
 
    data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))
 
    data = extract_data(data_dir + 't10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))
 
    data = extract_data(data_dir + 't10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trX = np.asarray(trX).tolist()
    teX = np.asarray(teX).tolist()
    trY = np.asarray(trY).tolist()
    teY = np.asarray(teY).tolist()

    for i in range(len(trX)):
        all_train_data['x'].append(remake_fashion_mnist(trX[i]))
        all_train_data['y'].append(trY[i])

    train_label_list = np.asarray(all_train_data["y"])
    train_idx_map = non_iid_partition_with_dirichlet_distribution(train_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in train_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_train_data["x"][i] for i in idx_list],
                     "y": [all_train_data["y"][i] for i in idx_list]}
        new_train_data[key] = temp_data

    for i in range(len(teX)):
        all_test_data['x'].append(remake_fashion_mnist(teX[i]))
        all_test_data['y'].append(teY[i])

    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data        

    if len(groups) == 0:
        groups = [None for _ in new_users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(new_users, groups):
        user_train_data_num = len(new_train_data[u]['x'])
        user_test_data_num = len(new_test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(new_train_data[u], batch_size, model_name)
        test_batch = batch_data(new_test_data[u], batch_size, model_name)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1

    logging.info("finished the loading distillation data")
    class_num = 10

    return test_data_global

def remake(pic,size=28):
    new = [(1-i)*255 for i in pic]
    new_pic =  np.array(new).reshape(size,size)
    mu = np.mean(new_pic.astype(np.float32),0)
    sigma = np.std(new_pic.astype(np.float32),0)
    new_pic2 = (new_pic.astype(np.float32)-mu)/(sigma+0.001)
    return new_pic2.flatten().tolist()

import gzip
def remake_fashion_mnist(pic,size=28):
    new_pic = []
    for i in range(len(pic)):
        for j in range(size):
            new_pic.append(pic[i][j][0])

    return remake(new_pic)

def load_partition_data_fashion_mnist(batch_size,
                                client_num_in_total,
                                model_name,
                                alpha,
                                data_dir="data/FASHION_MNIST/",
                                ):
    class_num = 10

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return data

    new_users = []
    groups=[]
    new_train_data = {}
    all_train_data = {"x": [], "y": []}
    new_test_data = {}
    all_test_data = {"x": [], "y": []}
    for i in range(client_num_in_total):
        if i < class_num:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))

    data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))
 
    data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))
 
    data = extract_data(data_dir + 't10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))
 
    data = extract_data(data_dir + 't10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trX = np.asarray(trX).tolist()
    teX = np.asarray(teX).tolist()
    trY = np.asarray(trY).tolist()
    teY = np.asarray(teY).tolist()

    for i in range(len(trX)):
        all_train_data['x'].append(remake_fashion_mnist(trX[i]))
        all_train_data['y'].append(trY[i])

    train_label_list = np.asarray(all_train_data["y"])
    train_idx_map = non_iid_partition_with_dirichlet_distribution(train_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in train_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_train_data["x"][i] for i in idx_list],
                     "y": [all_train_data["y"][i] for i in idx_list]}
        new_train_data[key] = temp_data

    for i in range(len(teX)):
        all_test_data['x'].append(remake_fashion_mnist(teX[i]))
        all_test_data['y'].append(teY[i])

    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data        

    if len(groups) == 0:
        groups = [None for _ in new_users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    distillation_data_global = list()
    client_idx = 0
    logging.info("loading data...")

    for u, g in zip(new_users, groups):
        user_train_data_num = len(new_train_data[u]['x'])
        user_test_data_num = len(new_test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(new_train_data[u], batch_size, model_name)
        test_batch = batch_data(new_test_data[u], batch_size, model_name)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_batch), len(test_batch)))
        client_idx += 1

    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
 

def load_partition_data_emnist(batch_size,
                                client_num_in_total,
                                model_name,
                                alpha,
                                train_path="data/EMINIST/datasets/fed_emnist_train.h5",
                                test_path="data/EMINIST/datasets/fed_emnist_test.h5"):
    _EXAMPLE = 'examples'
    _IMGAE = 'pixels'
    _LABEL = 'label'
    client_idx = None
    class_num = 62
    new_users = []
    groups=[]
    for i in range(client_num_in_total):
        if i < 62:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))

    train_h5 = h5py.File(train_path, 'r')
    new_train_data = {}
    all_train_data = {"x": [], "y": []}
    client_ids_train = list(train_h5[_EXAMPLE].keys())
    train_data_global = list()
    if client_idx is None:
        # get ids of all clients
        train_ids = client_ids_train[:]
    else:
        # get ids of single client
        train_ids = [client_ids_train[client_idx]] 
        
    for client_id in train_ids:
        temp = train_h5[_EXAMPLE][client_id][_LABEL][()]
        for index, label in enumerate(temp):
            if label < 100:
                x = np.array(train_h5[_EXAMPLE][client_id][_IMGAE][index]).flatten().tolist()
                y = train_h5[_EXAMPLE][client_id][_LABEL][index].tolist()
                all_train_data['x'].append(remake(x))
                all_train_data['y'].append(y)
                
    train_label_list = np.asarray(all_train_data["y"])
    train_idx_map = non_iid_partition_with_dirichlet_distribution(train_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in train_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_train_data["x"][i] for i in idx_list],
                     "y": [all_train_data["y"][i] for i in idx_list]}
        new_train_data[key] = temp_data

    test_h5 = h5py.File(test_path, 'r')
    new_test_data = {}
    all_test_data = {"x": [], "y": []}
    client_ids_test = list(test_h5[_EXAMPLE].keys())
    test_data_global = list()
    # load data
    if client_idx is None:
        # get ids of all clients
        test_ids = client_ids_test[:]
    else:
        # get ids of single client
        test_ids = [client_ids_test[client_idx]]

    # load data in numpy format from h5 file
    for client_id in test_ids:
        temp = test_h5[_EXAMPLE][client_id][_LABEL][()]
        for index, label in enumerate(temp):
            if label < 100:
                x = np.array(test_h5[_EXAMPLE][client_id][_IMGAE][index]).flatten().tolist()
                y = test_h5[_EXAMPLE][client_id][_LABEL][index].tolist()
                all_test_data['x'].append(remake(x))
                all_test_data['y'].append(y)    

    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data

    if len(groups) == 0:
        groups = [None for _ in new_users]
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    distillation_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u, g in zip(new_users, groups):
        user_train_data_num = len(new_train_data[u]['x'])
        user_test_data_num = len(new_test_data[u]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(new_train_data[u], batch_size, model_name)
        test_batch = batch_data(new_test_data[u], batch_size, model_name)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_batch), len(test_batch)))
        client_idx += 1

    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 62

    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def load_partition_distillation_data_emnist(batch_size,
                                            client_num_in_total,
                                            model_name,
                                            alpha,
                                            test_path="data/EMINIST/datasets/fed_emnist_test.h5"):
    _EXAMPLE = 'examples'
    _IMGAE = 'pixels'
    _LABEL = 'label'
    client_idx = None
    class_num = 10
    test_h5 = h5py.File(test_path, 'r')
    new_users = []
    new_test_data = {}
    all_test_data = {"x": [], "y": []}
    client_ids_test = list(test_h5[_EXAMPLE].keys())
    test_data_global = list()
    # load data
    if client_idx is None:
        # get ids of all clients
        test_ids = client_ids_test[:]
    else:
        # get ids of single client
        test_ids = [client_ids_test[client_idx]]

    # load data in numpy format from h5 file
    for client_id in test_ids:
        temp = test_h5[_EXAMPLE][client_id][_LABEL][()]
        for index, label in enumerate(temp):
            if label < 10:
                x = np.array(test_h5[_EXAMPLE][client_id][_IMGAE][index]).flatten().tolist()
                y = test_h5[_EXAMPLE][client_id][_LABEL][index].tolist()
                all_test_data['x'].append(remake(x))
                all_test_data['y'].append(y)

    for i in range(client_num_in_total):
        if i < 10:
            new_users.append("f_0000" + str(i))
        else:
            new_users.append("f_000" + str(i))
    
    test_label_list = np.asarray(all_test_data["y"])
    test_idx_map = non_iid_partition_with_dirichlet_distribution(test_label_list, client_num_in_total, class_num,
                                                                 alpha)

    for index, idx_list in test_idx_map.items():
        key = new_users[index]
        temp_data = {"x": [all_test_data["x"][i] for i in idx_list],
                     "y": [all_test_data["y"][i] for i in idx_list]}
        new_test_data[key] = temp_data

    logging.info("loading distillation_data...")

    for u in new_users:
        test_batch = batch_data(new_test_data[u], batch_size, model_name)
        test_data_global += test_batch

    logging.info("finish loading distillation_data...")
    return test_data_global

class ToxicComments(data.Dataset):

    def __init__(self, 
                root="vmalperovich/toxic_comments", 
                dataidxs=None, 
                max_seq_length=128,
                tokenizer=None,
                split="train"):

        self.root = root
        self.split = split
        self.dataset = load_dataset(self.root)
        self.dataset = self.dataset.rename_column("label", "labels")
        self.label_names = self.dataset["train"].features["labels"].feature.names
        self.max_seq_length = max_seq_length

        self.get_ids2label = lambda ids: [self.label_names[t] for t in ids]
        self.num_labels = len(self.label_names)
        self.tokenizer = tokenizer
        self.tokenize = lambda x: self.tokenizer(
            x["text"], truncation=True, max_length=self.max_seq_length
        )

        self.labeled_size = 400
        self.unlabeled_size = 3000
        self.full_size = self.labeled_size + self.unlabeled_size
        multiplier = int(np.log2(self.full_size / self.labeled_size)) - 1
        self.multiplier = max(1, multiplier)
        self.tokenized_data = self.gen_tokenized_data()

    def get_bool_labels(self, labels, num_classes):
        new_labels = np.zeros(num_classes, dtype=bool)
        for i in labels:
            new_labels[i] = True
        return {"labels": new_labels}

    def gen_tokenized_data(self):
        tokenize = lambda x: self.tokenizer(
            x["text"], truncation=True, max_length=self.max_seq_length
        )
        tokenized_dataset = self.dataset.map(tokenize, batched=True)
        tokenized_dataset = tokenized_dataset.map(
            lambda x: self.get_bool_labels(x["labels"], self.num_labels)
        )
        tokenized_dataset = tokenized_dataset.select_columns(
            ["input_ids", "attention_mask", "labels"]
        )
        print(tokenized_dataset)

        tokenized_train_df = tokenized_dataset["train"].to_pandas()
        tokenized_train_df_labeled = tokenized_train_df.sample(self.labeled_size)
        tokenized_train_df_labeled["labeled_mask"] = True

        tokenized_train_df = tokenized_train_df.sample(self.unlabeled_size)
        tokenized_train_df["labeled_mask"] = False
        tokenized_train_df["labels"] = tokenized_train_df["labels"].apply(
            lambda x: np.ones(self.num_labels, np.int64) * -100
        )

        for _ in range(self.multiplier):
            tokenized_train_df = pd.concat([tokenized_train_df, tokenized_train_df_labeled])

        tokenized_dataset["train"] = Dataset.from_pandas(
            tokenized_train_df_labeled, preserve_index=False
        )
        return tokenized_dataset
    
    def __len__(self):
        return len(self.tokenized_data[self.split])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.tokenized_data[self.split][idx]



def get_dataloader_ToxicComments(datadir, train_bs, test_bs, dataidxs1=None, dataidxs2=None):
    dl_obj = ToxicComments
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    np.random.seed(42)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_tokenized_dataset = dl_obj(datadir, dataidxs=dataidxs1, tokenizer=tokenizer, split='train')
    validation_tokenized_dataset = dl_obj(datadir, dataidxs=dataidxs1, tokenizer=tokenizer, split='validation')
    train_dl = DataLoader(
        train_tokenized_dataset,
        batch_size=train_bs,
        sampler=RandomSampler(train_tokenized_dataset),
        collate_fn=data_collator,
        pin_memory=True,
    )

    test_dl = DataLoader(
        validation_tokenized_dataset,
        batch_size=test_bs,
        sampler=SequentialSampler(validation_tokenized_dataset),
        collate_fn=data_collator,
        pin_memory=True,
    )
    train_data = list()
    for batch_idx, (batched_data) in enumerate(train_dl):
        train_data.append(batched_data)

    test_data = list()
    for batch_idx, (batched_data) in enumerate(test_dl):
        test_data.append(batched_data)
    return train_data, test_data


def load_partition_data_ToxCom(data_dir, batch_size, client_num_in_total, partition_alpha):
    class_num = 7
    new_users = []
    for client_idx in range(client_num_in_total):
        new_users.append(client_idx)
    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    data_local_num_dict = dict()
    train_data_global, test_data_global = get_dataloader_ToxicComments(data_dir, batch_size, batch_size)
    new_users, new_train_data, new_test_data = noniid_merge_data_with_dirichlet_distribution(client_num_in_total, train_data_global, test_data_global, partition_alpha, 7)
    for client_idx in range(client_num_in_total):
        data_local_num_dict[client_idx] = len(new_train_data[client_idx])
        train_data_local_dict[client_idx] = new_train_data[client_idx]
        test_data_local_dict[client_idx] = new_test_data[client_idx]
        train_data_num += len(new_train_data[client_idx])
        test_data_num += len(new_test_data[client_idx])
    return client_num_in_total, train_data_num, test_data_num, train_data_global, test_data_global, \
              data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    print("finish")
