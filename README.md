# UrbanROI

This folder splits the original single training script into several files and aligns its core model and training settings with the UrbanROI paper.

## Files

- `main.py`: entry point, region-level splits, training runs, and checkpoints.
- `data_utils.py`: ROI CSV loading, region-level splitting, and graph index construction.
- `dataset.py`: prebuilt control/treatment graph loading, cache, and collate function.
- `models.py`: shared GATv2 encoder, GRU, attention fusion, baseline and uplift heads.
- `train_eval.py`: training, validation, and loss.
- `utils.py`: metrics, random seeds, and simple time-series baseline checks.

## Important changes

1. Fixed the missing `dz = z1 - z0` in the uplift head (paper Eq. 13).
2. The baseline regularization uses the **last pre-entry ROI repeated across 8 steps**, rather than the last 8 observations (Eqs. 17–19).
3. Train/validation/test splitting uses the **real region ID** (70/15/15), not the new POI ID.
4. Attention fusion attends to both graph and ROI representations rather than a single key/value token.
5. Defaults match the paper: 30 input days, 8 output days, hidden size 768, GATv2 heads 4/1, dropout 0.2, AdamW, batch size 8, 10 epochs, alpha 0.1, and 10 runs.
6. MAPE is displayed as a percentage. Oversized graphs are rejected rather than independently clipping control and treatment edges.

## Run

Place these files in the same folder and install PyTorch, PyTorch Geometric, NumPy, Pandas, and scikit-learn. Then run:

```bash
python main.py --city Shanghai --runs 10 --region_map_csv dataset/Shanghai/region_map.csv
```

If your ROI CSV already contains a genuine `region_id` column, omit `--region_map_csv`.

Expected inputs (as in the original script):

```text
dataset/{city}/{city}_roi.csv
dataset/{city}/pyg_graphs/{city}_graphs_weight_sum_new.pkl
```

The ROI CSV contains `new_poi_id`, `create_date`, and 30 pre-entry plus 8 post-entry daily ROI values, optionally with `region_id`. When the region ID is absent, `region_map.csv` must contain `new_poi_id,region_id`. The graph pickle is an appended stream of `(new_poi_id, {"control": Data, "treatment": Data})` records.

Saved checkpoints:

```text
dataset/{city}/ckpt_causal/{city}_best_runXX.pt
```

**Note:** Like the original script, this code loads already-constructed graphs. It does not include the upstream user-request/POI graph construction or the additional experiments reported in the paper. Data filtering for concurrent POI entries/exits and extreme ROI values must be done beforehand.
