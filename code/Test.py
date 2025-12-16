import os

import torch
from torch import nn
import networkx as nx

import Graph_processing
import main
import numpy as np

class TestDistance:
    def __init__(self):
        self.input_dim = 256
        self.hidden_dim = 128
        self.model = main.DistanceMLP(self.input_dim, self.hidden_dim)

        state_dict = torch.load("Social Networks_EdgeWeight_model.pth", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        dataset_path = "Dataset"
        test_dataset = "Social Networks_EdgeWeight"


        self.test_path = os.path.join(dataset_path, test_dataset, "social_network1.txt")
        self.graph = main.load_graph(self.test_path)

        self.embedding = Graph_processing.Embedding(
            self.graph, dimension=128, walk_length=80, num_walks=10, workers=4
        ).generate_node2vec_embedding()

        for name, param in self.model.named_parameters():
            if 'weight' in name or 'bias' in name:
                print(name, param.mean().item(), param.std().item())

    def predict_pair(self, source, target):
        source_vec = self.embedding[source]
        target_vec = self.embedding[target]
        input_matrix = main.goal_node_combination(source_vec, target_vec)

        print("source, target =", source, target)
        print("input_matrix L2 norm:", np.linalg.norm(input_matrix))

        input_tensor = torch.from_numpy(input_matrix).float().unsqueeze(0)

        with torch.no_grad():
            pred = self.model(input_tensor)

        return pred.item()


if __name__ == "__main__":
    tester = TestDistance()
    for s, t in [(0, 3096), (1, 3096), (10, 2000), (100, 2000)]:
        d = tester.predict_pair(s, t)

        distance_actual = nx.shortest_path(tester.graph, s, t)
        print(distance_actual)
        print(d)
        print(f"({s}, {t}) -> {d}")
