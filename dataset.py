import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class GraphPairReader:
    def __init__(self, pkl_path):
        self.pkl_path = str(pkl_path)
        self.file = None

    def read(self, offset, expected_id):
        if self.file is None:
            self.file = open(self.pkl_path, "rb", buffering=8 * 1024 * 1024)
        self.file.seek(offset)
        event_id, graphs = pickle.load(self.file)
        if str(event_id) != str(expected_id):
            raise ValueError(f"Graph index mismatch: {expected_id} != {event_id}")
        return graphs["control"], graphs["treatment"]

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()


class UrbanROIDataset(Dataset):
    def __init__(self, df, graph_pkl, graph_index, max_edges=200000, cache_size=2048):
        self.event_ids = df["new_poi_id"].astype(str).tolist()
        self.past = np.stack(df["roi_past"].values).astype(np.float32)
        self.future = np.stack(df["roi_future"].values).astype(np.float32)
        self.graph_pkl = str(graph_pkl)
        self.graph_index = graph_index
        self.max_edges = max_edges
        self.cache_size = cache_size
        self.reader = None
        self.cache = None

    def __len__(self):
        return len(self.event_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["reader"] = None
        state["cache"] = None
        return state

    def __getitem__(self, index):
        if self.reader is None:
            self.reader = GraphPairReader(self.graph_pkl)
            self.cache = OrderedDict()
        event_id = self.event_ids[index]
        pair = self.cache.pop(event_id, None)
        if pair is None:
            g0, g1 = self.reader.read(self.graph_index[event_id], event_id)
            # The paper requires aligned graphs with a maximum of 200k edges.
            # Unlike the original optional independent clipping, rejecting oversized
            # graphs avoids silently modifying their shared pre-entry structure.
            if g1.num_nodes != g0.num_nodes + 1:
                raise ValueError(f"Event {event_id}: treatment must add exactly one POI")
            for graph in (g0, g1):
                if graph.edge_index.size(1) > self.max_edges:
                    raise ValueError(f"Event {event_id}: graph exceeds {self.max_edges} edges; "
                                     "cap aligned graphs in preprocessing")
                if graph.edge_attr is None or graph.edge_attr.numel() != graph.edge_index.size(1):
                    raise ValueError(f"Event {event_id}: expected one scalar weight per edge")
            pair = (g0, g1)
        if self.cache_size > 0:
            self.cache[event_id] = pair
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        g0, g1 = pair
        return g0, g1, torch.from_numpy(self.past[index]), torch.from_numpy(self.future[index])


def collate_pairs(batch):
    g0, g1, xp, yf = zip(*batch)
    return (Batch.from_data_list(list(g0)), Batch.from_data_list(list(g1)),
            torch.stack(xp), torch.stack(yf))
