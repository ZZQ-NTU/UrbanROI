import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd


def build_graph_index(graphs_pkl, index_pkl):
    """Index an appended pickle stream: (new_poi_id, {control, treatment})."""
    index = []
    seen = set()
    with open(graphs_pkl, "rb") as f:
        while True:
            offset = f.tell()
            try:
                event_id, _ = pickle.load(f)
            except EOFError:
                break
            event_id = str(event_id)
            if event_id in seen:
                raise ValueError(f"Duplicate graph key: {event_id}")
            seen.add(event_id)
            index.append((event_id, offset))
    if not index:
        raise ValueError("Graph pickle stream is empty")
    Path(index_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(index_pkl, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {key: offset for key, offset in index}


def load_graph_index(index_pkl):
    with open(index_pkl, "rb") as f:
        index = pickle.load(f)
    if isinstance(index, dict):
        return {str(key): int(offset) for key, offset in index.items()}
    keys = [str(key) for key, _ in index]
    if len(set(keys)) != len(keys):
        raise ValueError("Graph index contains duplicate event IDs; rebuild it")
    return {str(key): int(offset) for key, offset in index}


def load_roi_csv(roi_csv, past_len=30, future_len=8, region_map_csv=None):
    """Read original CSV layout: new_poi_id, create_date, [region_id], 38 ROI values.

    The graph pickle is keyed by the new POI/event ID. The *split* uses a separate,
    genuine region_id; different entrants in one region must be in one fold.
    """
    df = pd.read_csv(roi_csv, dtype=str)
    if df.shape[1] < 2:
        raise ValueError("ROI CSV needs an event ID and entry date")
    df = df.rename(columns={df.columns[0]: "new_poi_id", df.columns[1]: "create_date"})
    if df["new_poi_id"].isna().any() or df["new_poi_id"].duplicated().any():
        raise ValueError("The current pickle index requires unique, non-null new_poi_id")

    if region_map_csv:
        if "region_id" in df.columns:
            raise ValueError("region_id is already in ROI CSV; omit --region_map_csv")
        region_map = pd.read_csv(region_map_csv, dtype=str)
        if not {"new_poi_id", "region_id"}.issubset(region_map.columns):
            raise ValueError("Region mapping must have new_poi_id,region_id")
        if region_map["new_poi_id"].duplicated().any():
            raise ValueError("Duplicate new_poi_id in region mapping")
        df = df.merge(region_map[["new_poi_id", "region_id"]], on="new_poi_id",
                      how="left", validate="one_to_one")

    if "region_id" not in df.columns or df["region_id"].isna().any():
        raise ValueError(
            "Paper uses REGION-level splits: provide a real region_id column "
            "or --region_map_csv with new_poi_id,region_id"
        )
    if (df["region_id"].str.strip() == "").any():
        raise ValueError("Empty region_id is not allowed")

    metadata = {"new_poi_id", "create_date", "region_id"}
    roi_cols = [c for c in df.columns if c not in metadata]
    if len(roi_cols) != past_len + future_len:
        raise ValueError(f"Expected {past_len + future_len} chronological ROI columns, "
                         f"found {len(roi_cols)}")
    seq = df[roi_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    # The paper excludes invalid/zero ROI; further extreme-value and concurrent
    # entry/exit filtering must be done during upstream data preparation.
    valid = np.isfinite(seq).all(axis=1) & (seq != 0).all(axis=1)
    if (~valid).sum():
        print(f"[DATA] excluded {(~valid).sum()} invalid/zero ROI events")
    df = df.loc[valid].copy().reset_index(drop=True)
    seq = seq[valid]
    if not len(df):
        raise ValueError("No valid ROI events")
    df["roi_past"] = list(seq[:, :past_len])
    df["roi_future"] = list(seq[:, past_len:])
    return df[["new_poi_id", "create_date", "region_id", "roi_past", "roi_future"]]


def split_by_region(df, seed, train_ratio=0.70, val_ratio=0.15):
    regions = sorted(df["region_id"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(regions)
    n_train = int(len(regions) * train_ratio)
    n_val = int(len(regions) * val_ratio)
    if min(n_train, n_val, len(regions) - n_train - n_val) < 1:
        raise ValueError("Need enough regions for nonempty train/val/test partitions")
    ids = (set(regions[:n_train]),
           set(regions[n_train:n_train + n_val]),
           set(regions[n_train + n_val:]))
    return tuple(df[df["region_id"].isin(group)].reset_index(drop=True)
                 for group in ids)
