from __future__ import annotations

import argparse
import os
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO = Path(r"C:\Users\cyl\Desktop\iTransformer-phasec-clean")


PROFILES = {
    "traffic96_static": {
        "data": "custom",
        "root_path": "./dataset/traffic/",
        "data_path": "traffic.csv",
        "target": "OT",
        "freq": "h",
        "enc_in": 862,
        "dec_in": 862,
        "c_out": 862,
        "e_layers": 4,
        "d_model": 512,
        "d_ff": 512,
        "batch_size": 16,
        "learning_rate": 0.001,
        "train_epochs": 10,
        "patience": 3,
        "interface_dir": r"C:\Users\cyl\Desktop\data\interfaces\Traffic_graph_interface_parcorr",
        "variants": {
            "baseline": {
                "model_id": "traffic_96_96_baseline_itr3",
                "des": "Exp_baseline_itr3",
                "graph_enable": False,
            },
            "static": {
                "model_id": "traffic_96_96_staticcausal_softmax_itr3",
                "des": "Exp_staticcausal_softmax_itr3",
                "graph_enable": True,
            },
        },
    },
    "solar96_static": {
        "data": "Solar",
        "root_path": "./dataset/Solar/",
        "data_path": "solar_AL.txt",
        "target": "OT",
        "freq": "h",
        "enc_in": 137,
        "dec_in": 137,
        "c_out": 137,
        "e_layers": 2,
        "d_model": 512,
        "d_ff": 512,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "train_epochs": 10,
        "patience": 3,
        "interface_dir": r"C:\Users\cyl\Desktop\data\interfaces\Solar_graph_interface_parcorr",
        "variants": {
            "baseline": {
                "model_id": "solar_96_96_baseline_itr3",
                "des": "Exp_baseline_itr3",
                "graph_enable": False,
            },
            "static": {
                "model_id": "solar_96_96_staticcausal_softmax_itr3",
                "des": "Exp_staticcausal_softmax_itr3",
                "graph_enable": True,
            },
        },
    },
}


def build_args(profile_name: str, variant: str) -> Namespace:
    profile = PROFILES[profile_name]
    cfg = profile["variants"][variant]
    return Namespace(
        is_training=0,
        model_id=cfg["model_id"],
        model="iTransformer",
        data=profile["data"],
        root_path=profile["root_path"],
        data_path=profile["data_path"],
        phasec_split_path="",
        phasec_gating_lambda_path="",
        phasec_gating_lambda_hash="",
        phasec_gating_mode="none",
        phasec_gating_weight_polarity="inverse",
        phasec_gating_alpha=1.0,
        phasec_regime_lambda_path="",
        phasec_regime_lambda_hash="",
        phasec_regime_mode="none",
        graph_enable=bool(cfg["graph_enable"]),
        graph_mode="static_causal_residual",
        graph_interface_dir=profile["interface_dir"],
        graph_use_static_bias=False,
        graph_use_dynamic_bias=False,
        graph_use_lambda_gate=False,
        graph_lambda_gate_polarity="inverse",
        graph_shuffle_lambda=False,
        graph_eval_use_static_bias=False,
        graph_beta_static=0.10,
        graph_beta_dynamic=0.05,
        graph_soft_bias_scale_mode="fixed",
        graph_residual_alpha=0.10,
        graph_residual_scale_mode="fixed",
        graph_static_mix_mode="softmax",
        graph_lambda_logit_bias=False,
        graph_lambda_logit_bias_polarity="favor_base",
        graph_causal_pool_mode="auto",
        graph_causal_pool_budget_mb=512.0,
        graph_support_topk=32,
        graph_support_topk_metric="abs_a_base",
        graph_pool_dim=64,
        graph_lambda_loss_weighting=False,
        graph_lambda_loss_polarity="direct",
        graph_lambda_loss_alpha=1.0,
        seed=2023,
        features="M",
        target=profile["target"],
        freq=profile["freq"],
        checkpoints="./checkpoints/",
        seq_len=96,
        label_len=48,
        pred_len=96,
        enc_in=profile["enc_in"],
        dec_in=profile["dec_in"],
        c_out=profile["c_out"],
        d_model=profile["d_model"],
        n_heads=8,
        e_layers=profile["e_layers"],
        d_layers=1,
        d_ff=profile["d_ff"],
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        do_predict=False,
        num_workers=0,
        itr=3,
        train_epochs=profile["train_epochs"],
        batch_size=profile["batch_size"],
        patience=profile["patience"],
        learning_rate=profile["learning_rate"],
        des=cfg["des"],
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0,1,2,3",
        exp_name="MTSF",
        channel_independence=False,
        inverse=False,
        class_strategy="projection",
        target_root_path="./data/electricity/",
        target_data_path="electricity.csv",
        efficient_training=False,
        use_norm=1,
        partial_start_index=0,
    )


def setting_name(args: Namespace, projection: int) -> str:
    return (
        f"{args.model_id}_{args.model}_{args.data}_{args.features}_"
        f"ft{args.seq_len}_sl{args.label_len}_ll{args.pred_len}_"
        f"pl{args.d_model}_dm{args.n_heads}_nh{args.e_layers}_"
        f"el{args.d_layers}_dl{args.d_ff}_df{args.factor}_"
        f"fc{args.embed}_eb{args.distil}_dt{args.des}_{args.class_strategy}_{projection}"
    )


def collect_split_predictions(exp, flag: str) -> tuple[np.ndarray, np.ndarray]:
    data_set, _ = exp._get_data(flag=flag)
    loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    preds = []
    trues = []
    exp.model.eval()
    with torch.no_grad():
        for batch in loader:
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                _,
                batch_regime_x_aux,
                batch_regime_y_aux,
                batch_graph_lambda,
                batch_graph_delta,
            ) = exp._unpack_batch(batch)
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            outputs, batch_y = exp._forward_batch(
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_regime_x_aux=batch_regime_x_aux,
                batch_regime_y_aux=batch_regime_y_aux,
                batch_graph_lambda=batch_graph_lambda,
                batch_graph_delta=batch_graph_delta,
            )
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds_arr = np.asarray(preds).reshape(-1, exp.args.pred_len, exp.args.c_out)
    trues_arr = np.asarray(trues).reshape(-1, exp.args.pred_len, exp.args.c_out)
    return preds_arr, trues_arr


def mse_mae(preds: np.ndarray, trues: np.ndarray) -> tuple[float, float]:
    err = preds.astype(np.float64) - trues.astype(np.float64)
    return float(np.mean(err * err)), float(np.mean(np.abs(err)))


def load_checkpoint(exp, ckpt: Path, graph_enabled: bool) -> None:
    state_dict = torch.load(ckpt, map_location=exp.device)
    if graph_enabled:
        missing, unexpected = exp.model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"graph_causal_support"}
        missing = list(missing)
        unexpected = list(unexpected)
        if unexpected or any(name not in allowed_missing for name in missing):
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    else:
        exp.model.load_state_dict(state_dict)


def run_profile(
    profile: str,
    variant: str,
    split: str,
    *,
    projection_start: int = 0,
    projection_count: int | None = None,
) -> None:
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast

    args = build_args(profile, variant)
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    count = args.itr if projection_count is None else int(projection_count)
    if projection_start < 0 or count <= 0:
        raise ValueError(f"Invalid projection range: start={projection_start}, count={count}")

    for projection in range(int(projection_start), int(projection_start) + count):
        setting = setting_name(args, projection)
        ckpt = REPO / "checkpoints" / setting / "checkpoint.pth"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"[Backfill] profile={profile} variant={variant} split={split} setting={setting}", flush=True)
        exp = Exp_Long_Term_Forecast(args)
        load_checkpoint(exp, ckpt, graph_enabled=args.graph_enable)
        pred, true = collect_split_predictions(exp, flag=split)
        result_dir = REPO / "results" / setting
        result_dir.mkdir(parents=True, exist_ok=True)
        pred_name = "val_pred.npy" if split == "val" else "pred.npy"
        true_name = "val_true.npy" if split == "val" else "true.npy"
        metric_name = "val_metrics.npy" if split == "val" else "metrics.npy"
        np.save(result_dir / pred_name, pred)
        np.save(result_dir / true_name, true)
        mse, mae = mse_mae(pred, true)
        np.save(result_dir / metric_name, np.array([mae, mse], dtype=np.float32))
        print(f"[Backfill] wrote={result_dir} mse={mse:.6f} mae={mae:.6f} shape={pred.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill profile validation/test predictions for post-hoc calibration.")
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--variant", choices=["baseline", "static"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--projection-start", type=int, default=0)
    parser.add_argument("--projection-count", type=int, default=None)
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    run_profile(
        args.profile,
        args.variant,
        args.split,
        projection_start=args.projection_start,
        projection_count=args.projection_count,
    )


if __name__ == "__main__":
    main()
