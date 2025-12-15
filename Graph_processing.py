from node2vec import Node2Vec
import networkx as nx


class Embedding:
    def __init__(self, graph, dimension, walk_length, num_walks, workers):
        self.graph = graph
        self.dimension = dimension
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers

    def generate_node2vec_embedding(self):
        """图嵌入"""
        node2vec = Node2Vec(
            graph=self.graph,
            dimensions=self.dimension,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers
        )

        model = node2vec.fit(window=5, min_count=1, batch_words=128)

        embedding = {}
        for node in self.graph.nodes():  # 而不是 range(num_nodes)
            key = str(node)  # 统一成字符串最稳
            if key in model.wv:
                embedding[node] = model.wv[key]
            else:
                pass
        walks = node2vec.walks  # ← 关键
        print("number of walks:", len(walks))
        print("example walk:", walks[0])

        print("wv size:", len(model.wv))

        return embedding
