import os

import networkx as nx
import random


class GraphGenerator:
    def __init__(self, num_nodes, avg_degree):
        self.num_nodes = num_nodes
        self.avg_degree = avg_degree

    def generate_graph(self):
        """
        Barabási–Albert 小世界图（距离比较小，2~3 很正常）
        """
        m = max(1, self.avg_degree // 2)
        graph = nx.barabasi_albert_graph(self.num_nodes, m)
        return graph

    def generate_long_distance_graph(
        self,
        n_layers=30,        # 层数：越多，跨层越多，最短路径越长
        inter_edges=3,      # 相邻两层之间的边数（越少越远）
        intra_edges=2,      # 每层内部的随机边基数
        weight_low=1.0,
        weight_high=20.0,
    ):
        """
        生成“分层 + 稀疏跨层连接”的图，最短路径会明显比原图大很多。

        - 把节点按层分组：layer0, layer1, ..., layer_{n_layers-1}
        - 每层内部只加少量随机边（保证连通但不太密）
        - 相邻两层之间只加极少的边（形成瓶颈）
        - 整个图从最底层到最顶层，最短路大概会跨越很多层 → 距离变长
        """
        G = nx.Graph()
        layers = []

        # 1. 按层创建节点
        base_per_layer = max(1, self.num_nodes // n_layers)
        node_id = 0
        for layer_idx in range(n_layers):
            layer = []
            # 尽量平均分配节点到每一层
            for _ in range(base_per_layer):
                if node_id >= self.num_nodes:
                    break
                G.add_node(node_id)
                layer.append(node_id)
                node_id += 1
            if layer:
                layers.append(layer)
            if node_id >= self.num_nodes:
                break

        # 如果还有剩余节点，塞到最后一层
        while node_id < self.num_nodes:
            G.add_node(node_id)
            layers[-1].append(node_id)
            node_id += 1

        # 2. 每层内部加稀疏随机边
        for layer in layers:
            if len(layer) < 2:
                continue
            num_edges = intra_edges * len(layer)
            for _ in range(num_edges):
                u, v = random.sample(layer, 2)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=random.uniform(weight_low, weight_high))

        # 3. 相邻两层之间加很少的边（形成长距离的关键）
        for i in range(len(layers) - 1):
            L1 = layers[i]
            L2 = layers[i + 1]
            for _ in range(inter_edges):
                u = random.choice(L1)
                v = random.choice(L2)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=random.uniform(weight_low, weight_high))

        return G

    @staticmethod
    def add_edge_weights(graph):
        # 给每条边添加一个随机权重
        for u, v in graph.edges():
            graph[u][v]['weight'] = random.uniform(1.0, 20.0)

    @staticmethod
    def add_node_weights(graph):
        for node in graph.nodes():
            graph.nodes[node]['w1'] = random.randint(1, 20)
            graph.nodes[node]['w2'] = random.randint(5, 30)
            graph.nodes[node]['w3'] = random.randint(1, 10)

    @staticmethod
    def save(graph, path, folder_name, number):
        """
        把无向图 G 存成每行一条边： "u v weight"
        和 facebook_combined.txt 一样的格式
        """
        file_name = "social_network" + f"{number}" + ".txt"
        folder = os.path.join(path, folder_name)
        os.makedirs(folder, exist_ok=True)

        path = os.path.join(path, folder_name, file_name)

        with open(path, "w", encoding="utf-8") as f:
            for u, v in graph.edges():
                weights = round(graph[u][v].get('weight'))
                f.write(f"{u} {v} {weights}\n")


# ================== 下面是使用示例 ==================

folder_path = "Dataset"
create_folder_name = "Social Networks_EdgeWeight"

graph_generator = GraphGenerator(num_nodes=4039, avg_degree=44)

# ① 如果你还想用原来的 Barabási–Albert 图（小世界、距离很小），用这个：
# graphs = graph_generator.generate_graph()
# graph_generator.add_edge_weights(graphs)

# ② 想要“距离明显拉长”的图，用新的这个：
graphs = graph_generator.generate_long_distance_graph(
    n_layers=40,     # 层数多一点，路径就更长
    inter_edges=2,   # 层间边更少，距离更远
    intra_edges=2,   # 每层内部适当连一下，别太散
    weight_low=1.0,
    weight_high=20.0,
)

# 可选：如果想覆盖权重，用这句；如果保留上面的权重，可以不调
# graph_generator.add_edge_weights(graphs)

# 节点权重你之前有用就保留
graph_generator.add_node_weights(graphs)

for n in range(1, 101):
    graph_generator.save(graphs, folder_path, create_folder_name, n)
