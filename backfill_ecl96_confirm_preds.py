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


VARIANTS = {
    "baseline": {
        "model_id": "ecl96_confirm_lr5e4_baseline_itr3",
        "des": "ECL96ConfirmBaseLR5e4",
        "graph_enable": False,
    },
    "static": {
        "model_id": "ecl96_confirm_lr5e4_static_anchor_itr3",
        "des": "ECL96ConfirmAnchorLR5e4",
        "graph_enable": True,
    },
}


def build_args(variant: str) -> Namespace:
    cfg = VARIANTS[variant]
    return Namespace(
        is_training=0,
        model_id=cfg["model_id"],
        model="iTransformer",
        data="custom",
        root_path="./dataset/",
        data_path="ECL.csv",
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
        graph_interface_dir=r"C:\Users\cyl\Desktop\data\interfaces\ECL_graph_interface_parcorr",
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
        target="MT_320",
        freq="h",
        checkpoints="./checkpoints/",
        seq_len=96,
        label_len=48,
        pred_len=96,
        enc_in=321,
        dec_in=321,
        c_out=321,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=1,
        d_ff=512,
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
        train_epochs=10,
        batch_size=16,
        patience=3,
        learning_rate=0.0005,
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
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if data_set.scale and exp.args.inverse:
                shape = outputs.shape
                outputs = data_set.inverse_transform(outputs.squeeze(0)).reshape(shape)
                batch_y = data_set.inverse_transform(batch_y.squeeze(0)).reshape(shape)
            preds.append(outputs)
            trues.append(batch_y)

    preds = np.asarray(preds).reshape(-1, exp.args.pred_len, exp.args.c_out)
    trues = np.asarray(trues).reshape(-1, exp.args.pred_len, exp.args.c_out)
    return preds, trues


def mse_mae(preds: np.ndarray, trues: np.ndarray) -> tuple[float, float]:
    err = preds.astype(np.float64) - trues.astype(np.float64)
    return float(np.mean(err * err)), float(np.mean(np.abs(err)))


def setting_name(args: Namespace, projection: int) -> str:
    return (
        f"{args.model_id}_{args.model}_{args.data}_{args.features}_"
        f"ft{args.seq_len}_sl{args.label_len}_ll{args.pred_len}_"
        f"pl{args.d_model}_dm{args.n_heads}_nh{args.e_layers}_"
        f"el{args.d_layers}_dl{args.d_ff}_df{args.factor}_"
        f"fc{args.embed}_eb{args.distil}_dt{args.des}_{args.class_strategy}_{projection}"
    )


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


def run_variant(variant: str, split: str) -> None:
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast

    args = build_args(variant)
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

    for projection in range(args.itr):
        setting = setting_name(args, projection)
        ckpt = REPO / "checkpoints" / setting / "checkpoint.pth"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"[Backfill] variant={variant} split={split} setting={setting}", flush=True)
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
        print(f"[Backfill] done | mse={mse:.6f} mae={mae:.6f} shape={pred.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill ECL-96 validation/test predictions from saved checkpoints.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="static")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    run_variant(args.variant, args.split)


if __name__ == "__main__":
    main()
