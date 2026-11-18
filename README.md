# UrbanROI

This folder splits the original single training script into several files and aligns its core model and training settings with the UrbanROI paper.

## Files

- `main.py`: entry point, region-level splits, training runs, and checkpoints.
- `data_utils.py`: ROI CSV loading, region-level splitting, and graph index construction.
- `dataset.py`: prebuilt control/treatment graph loading, cache, and collate function.
- `models.py`: shared GATv2 encoder, GRU, attention fusion, baseline and uplift heads.
- `train_eval.py`: training, validation, and loss.
- `utils.py`: metrics, random seeds, and simple time-series baseline checks.

## Run

Place these files in the same folder and install PyTorch, PyTorch Geometric, NumPy, Pandas, and scikit-learn. Then run:

```bash
python main.py --city Shanghai
```

If your ROI CSV already contains a genuine `region_id` column, omit `--region_map_csv`.

Expected inputs (as in the original script):

```text
dataset/{city}/{city}_roi.csv
dataset/{city}/pyg_graphs/{city}_graphs_weight_sum_new.pkl
```
