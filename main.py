import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils import build_graph_index, load_graph_index, load_roi_csv, split_by_region
from dataset import UrbanROIDataset, collate_pairs
from models import UrbanROI
from train_eval import train_one_epoch, evaluate
from utils import seed_everything, mean_std, print_simple_baselines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--graphs_pkl", default=None)
    parser.add_argument("--roi_csv", default=None)
    parser.add_argument("--index_pkl", default=None)
    parser.add_argument("--region_map_csv", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--past_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=8)
    parser.add_argument("--hid_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--alpha_y0", type=float, default=0.1)
    parser.add_argument("--max_edges", type=int, default=200000)
    parser.add_argument("--graph_cache_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--simple_baselines", action="store_true")
    return parser.parse_args()


def make_loader(df, args, graph_index, train, device):
    dataset = UrbanROIDataset(df, args.graphs_pkl, graph_index,
                              max_edges=args.max_edges,
                              cache_size=args.graph_cache_size)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.num_workers, collate_fn=collate_pairs,
                      pin_memory=(device.type == "cuda"),
                      persistent_workers=args.num_workers > 0)


def run_once(args, graph_index, df, run_id, device):
    run_seed = args.seed + run_id
    seed_everything(run_seed)
    train_df, val_df, test_df = split_by_region(df, run_seed)
    print(f"[RUN {run_id + 1}/{args.runs}] seed={run_seed} "
          f"events={len(train_df)}/{len(val_df)}/{len(test_df)}")

    if args.simple_baselines:
        print_simple_baselines(train_df, test_df, args.future_len)

    train_loader = make_loader(train_df, args, graph_index, True, device)
    val_loader = make_loader(val_df, args, graph_index, False, device)
    test_loader = make_loader(test_df, args, graph_index, False, device)

    g0, _, _, _ = train_loader.dataset[0]
    model = UrbanROI(node_dim=g0.x.size(1), hid_dim=args.hid_dim,
                     future_len=args.future_len, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val = float("inf")
    ckpt_path = Path(args.save_dir) / f"{args.city}_best_run{run_id:02d}.pt"
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, args.alpha_y0,
                               use_amp, scaler)
        val = evaluate(model, val_loader, device, use_amp)
        print(f"[RUN{run_id:02d} E{epoch:03d}] loss={loss:.5f} "
              f"val_rmse={val['rmse']:.5f} val_mae={val['mae']:.5f} "
              f"val_mape={val['mape']:.2f}%")
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            torch.save({"model": model.state_dict(), "seed": run_seed}, ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = evaluate(model, test_loader, device, use_amp)
    print(f"[RUN{run_id:02d} TEST] RMSE={result['rmse']:.5f} "
          f"MAE={result['mae']:.5f} MAPE={result['mape']:.2f}%")
    return result


def main():
    args = parse_args()
    base = Path("dataset") / args.city
    args.graphs_pkl = args.graphs_pkl or str(base / "pyg_graphs" / f"{args.city}_graphs_weight_sum_new.pkl")
    args.roi_csv = args.roi_csv or str(base / f"{args.city}_roi.csv")
    args.index_pkl = args.index_pkl or str(base / "pyg_graphs" / f"{args.city}_graphs.index_weight_sum_new.pkl")
    args.save_dir = args.save_dir or str(base / "ckpt_causal")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if not Path(args.graphs_pkl).is_file() or not Path(args.roi_csv).is_file():
        raise FileNotFoundError("Missing graphs_pkl or roi_csv; check dataset paths")

    graph_index = (load_graph_index(args.index_pkl) if Path(args.index_pkl).is_file()
                   else build_graph_index(args.graphs_pkl, args.index_pkl))
    df = load_roi_csv(args.roi_csv, args.past_len, args.future_len,
                      region_map_csv=args.region_map_csv)
    missing = set(df["new_poi_id"]) - set(graph_index)
    if missing:
        print(f"[DATA] Excluding {len(missing)} events without prebuilt graphs")
        df = df[df["new_poi_id"].isin(graph_index)].reset_index(drop=True)
    device = torch.device(args.device)
    results = [run_once(args, graph_index, df, i, device) for i in range(args.runs)]
    print(f"\n[FINAL over {args.runs} runs]")
    for key in ("rmse", "mae", "mape"):
        avg, sd = mean_std([r[key] for r in results])
        unit = "%" if key == "mape" else ""
        print(f"{key.upper()}: {avg:.5f} ± {sd:.5f}{unit}")


if __name__ == "__main__":
    main()
