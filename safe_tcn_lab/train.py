from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from safe_tcn_lab.models import SafeTCNForecaster, TaskConditionedTCN


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _evaluate_multitask(model: TaskConditionedTCN, loader: DataLoader, profile_bank: torch.Tensor, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for x_feat, x_tgt, y, task_idx in loader:
            profiles = profile_bank[task_idx.to(device)]
            pred = model(x_feat.to(device), x_tgt.to(device), profiles)
            losses.append(float(criterion(pred, y.to(device)).item()))
    return float(np.mean(losses)) if losses else float("inf")


def _attach_training_artifacts(model: torch.nn.Module, history: list[dict[str, float]], best_epoch: int, best_val_loss: float, duration_sec: float) -> None:
    model._training_history = history
    model._training_summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "duration_sec": float(duration_sec),
        "epochs_ran": int(len(history)),
    }


def train_multitask_pretrain(
    model: TaskConditionedTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    profile_bank: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 4,
) -> TaskConditionedTCN:
    start_time = time.perf_counter()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_feat, x_tgt, y, task_idx in train_loader:
            profiles = profile_bank[task_idx.to(device)]
            pred = model(x_feat.to(device), x_tgt.to(device), profiles)
            loss = criterion(pred, y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss = _evaluate_multitask(model, val_loader, profile_bank, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "best_val_loss_so_far": float(min(best_loss, val_loss)),
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    _attach_training_artifacts(model, history, best_epoch, best_loss, time.perf_counter() - start_time)
    return model


def train_multitask_target_model(
    model: TaskConditionedTCN,
    train_loader: DataLoader,
    target_val_loader: DataLoader,
    profile_bank: torch.Tensor,
    target_profile: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 4,
) -> TaskConditionedTCN:
    start_time = time.perf_counter()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_feat, x_tgt, y, task_idx in train_loader:
            profiles = profile_bank[task_idx.to(device)]
            pred = model(x_feat.to(device), x_tgt.to(device), profiles)
            loss = criterion(pred, y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss = _evaluate_local(model, target_val_loader, target_profile, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "best_val_loss_so_far": float(min(best_loss, val_loss)),
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    _attach_training_artifacts(model, history, best_epoch, best_loss, time.perf_counter() - start_time)
    return model


def _evaluate_local(model: TaskConditionedTCN, loader: DataLoader, profile: torch.Tensor, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for x_feat, x_tgt, y in loader:
            pred = model(
                x_feat.to(device),
                x_tgt.to(device),
                profile.to(device).unsqueeze(0).expand(x_feat.shape[0], -1),
            )
            losses.append(float(criterion(pred, y.to(device)).item()))
    return float(np.mean(losses)) if losses else float("inf")


def configure_trainable_parts(model: TaskConditionedTCN, mode: str = "all") -> TaskConditionedTCN:
    for parameter in model.parameters():
        parameter.requires_grad = True
    if mode == "all":
        return model
    if mode == "head":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        return model
    raise ValueError(f"Unsupported trainable mode '{mode}'.")


def train_local_model(
    model: TaskConditionedTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    profile: torch.Tensor,
    device: torch.device,
    trainable_parts: str = "all",
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 4,
) -> TaskConditionedTCN:
    start_time = time.perf_counter()
    model = model.to(device)
    model = configure_trainable_parts(model, mode=trainable_parts)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_feat, x_tgt, y in train_loader:
            pred = model(
                x_feat.to(device),
                x_tgt.to(device),
                profile.to(device).unsqueeze(0).expand(x_feat.shape[0], -1),
            )
            loss = criterion(pred, y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss = _evaluate_local(model, val_loader, profile, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "best_val_loss_so_far": float(min(best_loss, val_loss)),
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    _attach_training_artifacts(model, history, best_epoch, best_loss, time.perf_counter() - start_time)
    return model


def evaluate_safe_tcn(
    model: SafeTCNForecaster,
    loader: DataLoader,
    target_profile: torch.Tensor,
    source_profiles: torch.Tensor,
    relation_features: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for x_feat, x_tgt, y in loader:
            out = model(
                x_feat.to(device),
                x_tgt.to(device),
                target_profile.to(device),
                source_profiles.to(device),
                relation_features.to(device),
            )
            losses.append(float(criterion(out["final"], y.to(device)).item()))
    return float(np.mean(losses)) if losses else float("inf")


def train_safe_tcn(
    model: SafeTCNForecaster,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_profile: torch.Tensor,
    source_profiles: torch.Tensor,
    relation_features: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    gate_lambda: float = 0.25,
    safety_lambda: float = 1.0,
    sparsity_lambda: float = 0.02,
    relation_lambda: float = 0.05,
    budget_lambda: float = 0.02,
    harm_margin: float = 0.0,
    patience: int = 4,
) -> SafeTCNForecaster:
    start_time = time.perf_counter()
    model = model.to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_feat, x_tgt, y in train_loader:
            y_batch = y.to(device)
            out = model(
                x_feat.to(device),
                x_tgt.to(device),
                target_profile.to(device),
                source_profiles.to(device),
                relation_features.to(device),
            )
            final_se = (out["final"] - y_batch) ** 2
            target_se = (out["target"] - y_batch) ** 2
            final_loss = final_se.mean()
            per_window_final = final_se.mean(dim=1)
            per_window_target = target_se.mean(dim=1)

            if out["sources"].numel() > 0:
                source_se = (out["sources"] - y_batch.unsqueeze(1)) ** 2
                gains = target_se.unsqueeze(1) - source_se
                best_gain, best_idx = gains.max(dim=1)
                positive = best_gain > 0
                transfer_target = torch.clamp(best_gain / (target_se + 1e-6), min=0.0, max=1.0)
                transfer_gate_loss = nn.functional.binary_cross_entropy(out["transfer_strength"], transfer_target)
                if positive.any():
                    ce = nn.functional.cross_entropy(
                        out["source_logits"].permute(0, 2, 1)[positive],
                        best_idx[positive],
                        reduction="none",
                    )
                    source_gate_loss = (ce * transfer_target[positive].detach()).mean()
                else:
                    source_gate_loss = torch.tensor(0.0, device=device)
                entropy = -torch.sum(out["weights"] * torch.log(out["weights"] + 1e-6), dim=1) / math.log(max(int(out["weights"].shape[1]), 2))
                sparsity_loss = (entropy * out["transfer_strength"]).mean()
                source_usage = out["source_gates"] * out["transfer_strength"].unsqueeze(1)
                relation_penalty = (source_usage * out["source_mismatch"].view(1, -1, 1)).mean()
            else:
                transfer_gate_loss = torch.tensor(0.0, device=device)
                source_gate_loss = torch.tensor(0.0, device=device)
                sparsity_loss = torch.tensor(0.0, device=device)
                relation_penalty = torch.tensor(0.0, device=device)

            horizon_harm = torch.relu(final_se - target_se + harm_margin).mean()
            window_harm = torch.relu(per_window_final - per_window_target + harm_margin).mean()
            safety_loss = 0.7 * horizon_harm + 0.3 * window_harm
            strength_loss = out["transfer_strength"].mean()
            budget_loss = out["residual_budget"].mean()
            loss = (
                final_loss
                + gate_lambda * (transfer_gate_loss + source_gate_loss)
                + safety_lambda * safety_loss
                + sparsity_lambda * (0.7 * sparsity_loss + 0.3 * strength_loss)
                + relation_lambda * relation_penalty
                + budget_lambda * budget_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss = evaluate_safe_tcn(model, val_loader, target_profile, source_profiles, relation_features, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "best_val_loss_so_far": float(min(best_loss, val_loss)),
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    _attach_training_artifacts(model, history, best_epoch, best_loss, time.perf_counter() - start_time)
    return model


def collect_local_predictions(model: TaskConditionedTCN, loader: DataLoader, profile: torch.Tensor, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for x_feat, x_tgt, y in loader:
            pred = model(
                x_feat.to(device),
                x_tgt.to(device),
                profile.to(device).unsqueeze(0).expand(x_feat.shape[0], -1),
            )
            preds.append(pred.cpu().numpy())
            truths.append(y.numpy())
    return np.concatenate(truths, axis=0), np.concatenate(preds, axis=0)


def collect_safe_predictions(
    model: SafeTCNForecaster,
    loader: DataLoader,
    target_profile: torch.Tensor,
    source_profiles: torch.Tensor,
    relation_features: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for x_feat, x_tgt, y in loader:
            out = model(
                x_feat.to(device),
                x_tgt.to(device),
                target_profile.to(device),
                source_profiles.to(device),
                relation_features.to(device),
            )
            preds.append(out["final"].cpu().numpy())
            truths.append(y.numpy())
    return np.concatenate(truths, axis=0), np.concatenate(preds, axis=0)


def collect_safe_components(
    model: SafeTCNForecaster,
    loader: DataLoader,
    target_profile: torch.Tensor,
    source_profiles: torch.Tensor,
    relation_features: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truths = []
    target_preds = []
    transfer_terms = []
    with torch.no_grad():
        for x_feat, x_tgt, y in loader:
            out = model(
                x_feat.to(device),
                x_tgt.to(device),
                target_profile.to(device),
                source_profiles.to(device),
                relation_features.to(device),
            )
            truths.append(y.numpy())
            target_preds.append(out["target"].cpu().numpy())
            transfer_terms.append((out["transfer_strength"] * out["bounded_transfer"]).cpu().numpy())
    return (
        np.concatenate(truths, axis=0),
        np.concatenate(target_preds, axis=0),
        np.concatenate(transfer_terms, axis=0),
    )


def collect_safe_outputs(
    model: SafeTCNForecaster,
    loader: DataLoader,
    target_profile: torch.Tensor,
    source_profiles: torch.Tensor,
    relation_features: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    rows: dict[str, list[np.ndarray]] = {
        "truths": [],
        "final": [],
        "target": [],
        "transfer_strength": [],
        "raw_transfer": [],
        "bounded_transfer": [],
        "calibrated_transfer": [],
        "residual_budget": [],
        "source_preds": [],
        "source_weights": [],
        "source_gates": [],
    }
    with torch.no_grad():
        for x_feat, x_tgt, y in loader:
            out = model(
                x_feat.to(device),
                x_tgt.to(device),
                target_profile.to(device),
                source_profiles.to(device),
                relation_features.to(device),
            )
            rows["truths"].append(y.numpy())
            rows["final"].append(out["final"].cpu().numpy())
            rows["target"].append(out["target"].cpu().numpy())
            rows["transfer_strength"].append(out["transfer_strength"].cpu().numpy())
            rows["raw_transfer"].append(out["raw_transfer"].cpu().numpy())
            rows["bounded_transfer"].append(out["bounded_transfer"].cpu().numpy())
            rows["calibrated_transfer"].append(out["calibrated_transfer"].cpu().numpy())
            rows["residual_budget"].append(out["residual_budget"].cpu().numpy())
            rows["source_preds"].append(out["sources"].cpu().numpy())
            rows["source_weights"].append(out["weights"].cpu().numpy())
            rows["source_gates"].append(out["source_gates"].cpu().numpy())
    result: dict[str, np.ndarray] = {}
    for key, parts in rows.items():
        if not parts:
            result[key] = np.zeros((0,), dtype=np.float32)
            continue
        result[key] = np.concatenate(parts, axis=0)
    result["calibration_alpha"] = model.calibration_alpha.detach().cpu().numpy().copy()
    return result


def calibrate_safe_tcn(
    model: SafeTCNForecaster,
    loader: DataLoader,
    target_profile: torch.Tensor,
    source_profiles: torch.Tensor,
    relation_features: torch.Tensor,
    device: torch.device,
    harm_limit: float = 0.45,
    grid_size: int = 11,
) -> SafeTCNForecaster:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    y_true, target_pred, transfer_delta = collect_safe_components(
        model,
        loader,
        target_profile=target_profile,
        source_profiles=source_profiles,
        relation_features=relation_features,
        device=device,
    )
    if y_true.size == 0:
        return model

    base_se = (target_pred - y_true) ** 2
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    alpha = np.zeros(target_pred.shape[1], dtype=np.float32)

    for horizon in range(target_pred.shape[1]):
        base_mse = float(base_se[:, horizon].mean())
        best_alpha = 0.0
        best_mse = base_mse
        for candidate in grid[1:]:
            pred = target_pred[:, horizon] + candidate * transfer_delta[:, horizon]
            se = (pred - y_true[:, horizon]) ** 2
            mse = float(se.mean())
            harm = float((se > base_se[:, horizon]).mean())
            if mse <= base_mse and harm <= harm_limit:
                if mse < best_mse - 1e-8 or (abs(mse - best_mse) <= 1e-8 and candidate < best_alpha):
                    best_mse = mse
                    best_alpha = float(candidate)
        alpha[horizon] = best_alpha

    model.calibration_alpha.copy_(torch.from_numpy(alpha).to(device=device, dtype=model.calibration_alpha.dtype))
    return model
