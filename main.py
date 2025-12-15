import os
import random
from collections import deque
from time import sleep

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from Graph_processing import Embedding
from sklearn.model_selection import train_test_split

from matplotlib import (pyplot as plt)

def load_graph(graph_file):
    """
    图加载
    :param graph_file:
    :return: graph地址
    调用使用用graph.nodes()(list)
    """
    graph = nx.read_edgelist(
        graph_file,
        nodetype=int,
        data=(('weight', float),)
    )

    return graph
    # graph = {}
    # with open(graph_file, "r") as file:
    #     for line in file:
    #         a, b = map(int, line.split())
    #         graph.setdefault(a, []).append(b)
    #         graph.setdefault(b, []).append(a)
    # return graph

def big_graph_mapping(graph_list):
    """
    训练集/测试集
    映射成一个big graph
    输入图嵌入神经网络
    :param graph_list:
    :return:
    """
    big_graph = nx.Graph()
    id_map = []
    new_id = 0  # 记录新的big_graph 每个节点的新id

    for gid, graph in enumerate(graph_list):
        global_node_number = {}
        for node in graph.nodes():
            global_node_number[node] = new_id
            big_graph.add_node(new_id)
            new_id = new_id + 1

        for u, v in graph.edges():
            big_graph.add_edge(global_node_number[u], global_node_number[v])

        id_map.append(global_node_number)

    print("Big graph:", big_graph, "nodes", big_graph.number_of_edges(), "edges")
    return big_graph, id_map

def get_landmarks(graph, num_landmarks=20):
    """
    随机采样地标
    :return:若干landmark
    """
    nodes = list(graph.nodes())
    return random.sample(nodes, min(num_landmarks, len(nodes)))

def bfs_landmark_sampling(graph, landmark):
    """
    bfs单地标遍历采样 单位长度为1
    从landmark出发 找整张图连接节点 遍历测最短路径
    """
    return dict(nx.single_source_shortest_path_length(graph, landmark))

def dijkstra_landmark_sampling(graph, landmark):
    """
    dijkstra单地标遍历 适用edge weight
    """
    return dict(nx.single_source_dijkstra_path_length(graph, landmark))

def sampling(graph, landmarks, max_distance=None, drop_distance_1=True):
    """
    收集地标采样对（单图）
    :param graph:
    :param landmarks:
    :param max_distance:
    :return: pairs
    """
    pairs = []

    for landmark in landmarks:
        # 形式为{node1:distance1,node2:distance2...}
        landmark_sample = dijkstra_landmark_sampling(graph, landmark)
        # 过滤样本
        for node, distance in landmark_sample.items():
            # 不考虑landmark自身距离为0
            if node == landmark:
                continue
            # 不考虑距离为1的邻居节点
            if drop_distance_1 and distance == 1:
                continue
            # 不考虑太远的节点
            if (max_distance is not None) and distance > max_distance:
                continue
            pairs.append((landmark, node, distance))

    return pairs

def goal_node_combination(emb_start, emb_end, mode="concat"):
    """
    对将要测量的目标节点的特征矩阵 做特征融合
    可选操作包括
    concat subtract average hadamard(逐元素乘)
    """
    if mode == "concat":
        return np.concatenate((emb_start, emb_end))
    elif mode == "subtract":
        return emb_start - emb_end
    elif mode == "average":
        return (emb_start + emb_end) / 2
    elif mode == "hadamard":
        return emb_start * emb_end
    else:
        raise ValueError(f"Unknown combine mode: {mode}")

class PairDataset(Dataset):
    """
    两目标节点间特征
    写成dataset
    方便给DataLoader用
    """
    def __init__(self, pairs, embeddings, combine_mode="concat"):
        self.X = [] # 线性特征concat (输入特征)
        self.Y = [] # 节点距离 (标签)
        self.combine_mode = combine_mode

        for start, end, distance in pairs:
            # 这里embedding是用字典写的 对big_graph来说也挺好找的
            emb_start = embeddings[start]
            emb_end = embeddings[end]
            embedding_combined = goal_node_combination(emb_start, emb_end, combine_mode)
            self.X.append(embedding_combined.astype(np.float32))
            self.Y.append(float(distance))

        self.X = np.stack(self.X, axis=0)
        self.Y = np.array(self.Y, dtype=np.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class DistanceMLP(nn.Module):
    """最小距离预测神经网络"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_distance_mlp(model, train_loader, device, epochs, lr):
    model.train()
    # Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # MSE损失
    criterion = nn.MSELoss()

    one_graph_loss = []

    for epoch in range(epochs):
        total_loss = 0.0
        # X_batch: 双节点特征融合矩阵 Y_batch: 节点距离
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, Y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        one_graph_loss.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, train MSE: {avg_loss:.4f}")

    return one_graph_loss

@torch.no_grad()    # 取消梯度更新
def evaluate(model, test_loader, device):
    model.eval()
    true_value = []   # 真实值
    pred_value = []   # 预测值

    for X_batch, Y_batch in test_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        y_pred = model(X_batch)

        true_value.append(Y_batch.cpu().numpy())
        pred_value.append(y_pred.cpu().numpy())

    y_true = np.concatenate(true_value)
    y_pred = np.concatenate(pred_value)

    # 四舍五入成整数距离(可选 后续待观察)
    y_pred_rounded = np.round(y_pred)

    # 绝对误差
    MAE = np.mean(np.abs(y_pred_rounded - y_true))
    # 相对误差
    MRE = np.mean(np.abs(y_pred - y_true) / y_true)

    print(f"MAE: {MAE:.4f}, MRE: {MRE:.4f}")
    return y_true, y_pred

def main():
    dataset_path = "Dataset"
    train_dataset = "Social Networks_EdgeWeight"
    test_dataset = "Social Networks_TestSet"

    training_path = os.path.join(dataset_path, train_dataset)
    test_path = os.path.join(dataset_path, test_dataset)

    """==================训练阶段=================="""
    # 处理data_set数据 每张图单独embedding
    # 取landmark 每张图获得pair对 作为训练集
    graphs_list = dataset_processing(training_path, dataset="Single Graph")

    # 先随便用一张图算出 input_dim
    example_graph = graphs_list[0]
    example_emb = Embedding(example_graph, dimension=128, walk_length=40, num_walks=50, workers=4)
    example_emb_list = example_emb.generate_node2vec_embedding()
    example_pairs = sampling(example_graph, get_landmarks(example_graph))
    example_dataset = PairDataset(example_pairs, example_emb_list, "concat")
    input_dim = example_dataset.X.shape[1]

    # 1. 模型只建一次
    model = DistanceMLP(input_dim=input_dim, hidden_dim=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_history = []
    # 2. 依次拿每一张图来“喂”这个模型
    for gid, graph in enumerate(graphs_list):
        print(f"\n=== Train on graph {gid} ===")

        # 2.1 对当前这张图做 node2vec
        embedding = Embedding(graph, dimension=128, walk_length=40, num_walks=50, workers=4)
        embedding_list = embedding.generate_node2vec_embedding()

        # 2.2 采样 landmark pair
        landmarks = get_landmarks(graph)
        pairs = sampling(graph, landmarks)

        # 2.3 为这张图建 dataset + loader
        train_dataset = PairDataset(pairs, embedding_list, combine_mode="concat")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 2.4 用“同一个模型”在这张图的数据上继续训练几轮
        one_graph_loss = train_distance_mlp(model, train_loader, device, epochs=30, lr=0.05)
        loss_history.extend(one_graph_loss)

        print("nodes, edges:", graph.number_of_nodes(), graph.number_of_edges())

    plt.plot(loss_history)
    plt.xlabel("Training Steps")
    plt.ylabel("MSE Loss")
    plt.title("Total Training Loss Curve Across All Graphs")
    plt.show()

    torch.save(model.state_dict(), "distance_mlp.pth")
    print("模型已保存为 distance_mlp.pth")

    """==================测试阶段=================="""

    test_graphs_list = dataset_processing(test_path, dataset="Single Graph")

    for gid, graph in enumerate(test_graphs_list):
        print(f"\n------ 测试图 {gid} ------")

        # 1. 对该图做 node2vec（测试集独立）
        embedding = Embedding(graph,
                              dimension=128,
                              walk_length=80,
                              num_walks=50,
                              workers=4)
        embedding_list = embedding.generate_node2vec_embedding()

        # 2. landmark + BFS 抽节点对
        landmarks = get_landmarks(graph)
        pairs = sampling(graph, landmarks)

        if len(pairs) == 0:
            print(f"图 {gid} 没抽到 pairs，跳过。")
            continue

        # 3. 构建 Dataset + DataLoader
        test_dataset = PairDataset(pairs, embedding_list, "concat")
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        print(f"测试图 {gid} 的样本量：", len(test_dataset))
        evaluate(model, test_loader, device)


def dataset_processing(root_path, dataset):
    graphs_list = []
    if dataset == "Dataset":
        for root, dirs, files in os.walk(root_path):
            for filename in files:
                path = os.path.join(root, filename)
                graphs = load_graph(path)
                graphs_list.append(graphs)
        big_graph, id_map = big_graph_mapping(graphs_list)
        print("Big graph:", big_graph.number_of_nodes(), "nodes",
              big_graph.number_of_edges(), "edges")
        return graphs_list, big_graph, id_map
    elif dataset == "Single Graph":
        for root, dirs, files in os.walk(root_path):
            for filename in files:
                path = os.path.join(root, filename)
                graphs = load_graph(path)
                graphs_list.append(graphs)
        return graphs_list
    else:
        return ValueError(f"Unknown dataset: {dataset}")

if __name__ == "__main__":
    main()
# def bfs_shortest_distance(graph, start, end):
#     """
#     对于给定两节点的直接测距
#     :param graph:
#     :param start:
#     :param end:
#     :return:
#     """
#     if start == end:
#         return 0
#
#     visited = set([start])
#     queue = deque([(start, 0)])
#
#     while queue:
#         node, distance = queue.popleft()
#         for neighbor in graph[node]:
#             if neighbor == end:
#                 return distance + 1
#             if neighbor not in visited:
#                 visited.add(neighbor)
#                 queue.append((neighbor, distance + 1))
#
#
# def bfs_shortest_path(graph, start, end):
#     if start == end:
#         return 0
#
#     visited = set([start])
#     queue = deque([[start]])
#
#     while queue:
#         path = queue.popleft()
#         node = path[-1]
#
#         for neighbor in graph[node]:
#             if neighbor == end:
#                 return path + [neighbor]
#             if neighbor not in visited:
#                 visited.add(neighbor)
#                 queue.append(path + [neighbor])
#
#     return None




# print(graphs)
# print(bfs_shortest_distance(graphs, 0, 1246))
# print(bfs_shortest_path(graphs, 0, 1246))