from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(self.act(self.norm1(self.conv1(x))))
        x = self.dropout(self.act(self.norm2(self.conv2(x))))
        return x + residual


class TemporalConvBackbone(nn.Module):
    def __init__(self, input_dim: int, model_dim: int = 64, levels: int = 4, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1)
        self.blocks = nn.ModuleList([ResidualBlock(model_dim, kernel_size, 2**idx, dropout) for idx in range(levels)])
        self.out_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x.transpose(1, 2))


class TaskConditionedTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        profile_dim: int,
        pred_len: int,
        model_dim: int = 64,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.backbone = TemporalConvBackbone(input_dim, model_dim, levels, kernel_size, dropout)
        self.context_proj = nn.Linear(model_dim * 2, model_dim)
        self.profile_encoder = nn.Sequential(
            nn.Linear(profile_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.film = nn.Linear(model_dim, model_dim * 2)
        self.base_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, pred_len),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, pred_len),
        )
        self.adapt_norm = nn.LayerNorm(model_dim)

    def encode(self, x_feat: torch.Tensor, x_tgt: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(torch.cat([x_feat, x_tgt], dim=-1))
        pooled = torch.cat([hidden[:, -1, :], hidden.mean(dim=1)], dim=-1)
        return self.context_proj(pooled)

    def predict_with_profile_from_context(
        self,
        context: torch.Tensor,
        profile: torch.Tensor,
    ) -> torch.Tensor:
        prof = self.profile_encoder(profile)
        gamma, beta = self.film(prof).chunk(2, dim=-1)
        adapted = self.adapt_norm(context * (1.0 + gamma) + beta)
        return self.base_head(context) + self.residual_head(adapted)

    def forward(self, x_feat: torch.Tensor, x_tgt: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        return self.predict_with_profile_from_context(self.encode(x_feat, x_tgt), profile)


class SafeTCNForecaster(nn.Module):
    def __init__(
        self,
        target_model: TaskConditionedTCN,
        source_model: TaskConditionedTCN,
        relation_dim: int,
        num_sources: int,
        gate_hidden_dim: int = 64,
        dropout: float = 0.1,
        residual_cap_scale: float = 0.75,
        residual_cap_floor: float = 0.02,
    ) -> None:
        super().__init__()
        self.target_model = deepcopy(target_model)
        self.source_model = deepcopy(source_model)
        model_dim = self.target_model.context_proj.out_features
        self.num_sources = num_sources
        self.pred_len = self.target_model.pred_len
        self.residual_cap_scale = residual_cap_scale
        self.residual_cap_floor = residual_cap_floor
        self.register_buffer("calibration_alpha", torch.ones(self.pred_len))
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
        )
        self.source_gate = nn.Sequential(
            nn.Linear(model_dim + gate_hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, self.pred_len),
        )
        self.transfer_gate = nn.Sequential(
            nn.Linear(model_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, self.pred_len),
        )
        self.residual_budget = nn.Sequential(
            nn.Linear(model_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, self.pred_len),
            nn.Sigmoid(),
        )

        for parameter in self.target_model.parameters():
            parameter.requires_grad = False
        for parameter in self.source_model.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        x_feat: torch.Tensor,
        x_tgt: torch.Tensor,
        target_profile: torch.Tensor,
        source_profiles: torch.Tensor,
        relation_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = x_feat.shape[0]
        target_context = self.target_model.encode(x_feat, x_tgt)
        target_pred = self.target_model.predict_with_profile_from_context(
            target_context,
            target_profile.expand(batch_size, -1),
        )

        if source_profiles.numel() == 0:
            return {
                "final": target_pred,
                "target": target_pred,
                "sources": torch.zeros(batch_size, 0, self.target_model.pred_len, device=target_context.device),
                "weights": torch.zeros(batch_size, 0, self.pred_len, device=target_context.device),
                "source_gates": torch.zeros(batch_size, 0, self.pred_len, device=target_context.device),
                "transfer_strength": torch.zeros(batch_size, self.pred_len, device=target_context.device),
                "source_logits": torch.zeros(batch_size, 0, self.pred_len, device=target_context.device),
                "raw_transfer": torch.zeros(batch_size, self.pred_len, device=target_context.device),
                "bounded_transfer": torch.zeros(batch_size, self.pred_len, device=target_context.device),
                "residual_budget": torch.zeros(batch_size, self.pred_len, device=target_context.device),
                "source_mismatch": torch.zeros(0, device=target_context.device),
            }

        source_preds = []
        with torch.no_grad():
            source_context = self.source_model.encode(x_feat, x_tgt)
            for source_idx in range(source_profiles.shape[0]):
                profile = source_profiles[source_idx].unsqueeze(0).expand(batch_size, -1)
                source_preds.append(self.source_model.predict_with_profile_from_context(source_context, profile))
        source_preds_t = torch.stack(source_preds, dim=1)

        relation_encoded = self.relation_encoder(relation_features).unsqueeze(0).expand(batch_size, -1, -1)
        context_rep = target_context.unsqueeze(1).expand(batch_size, self.num_sources, -1)
        gate_features = torch.cat([context_rep, relation_encoded], dim=-1)
        source_logits = self.source_gate(gate_features)
        source_gates = torch.sigmoid(source_logits)
        source_weights = source_gates / (source_gates.sum(dim=1, keepdim=True) + 1e-6)
        transfer_strength = torch.sigmoid(self.transfer_gate(target_context))
        source_residuals = source_preds_t - target_pred.unsqueeze(1)
        raw_transfer = torch.sum(source_weights * source_residuals, dim=1)
        source_dispersion = torch.sum(source_weights * source_residuals.abs(), dim=1)
        residual_budget = self.residual_cap_floor + (
            self.residual_cap_scale * self.residual_budget(target_context) * torch.clamp(source_dispersion, min=self.residual_cap_floor)
        )
        bounded_transfer = residual_budget * torch.tanh(raw_transfer / (residual_budget + 1e-6))
        calibrated_transfer = transfer_strength * bounded_transfer * self.calibration_alpha.view(1, -1)
        final = target_pred + calibrated_transfer

        similarity = relation_features[:, 0].clamp(min=-1.0, max=1.0)
        profile_gap = relation_features[:, 1].clamp(min=0.0)
        l2_gap = relation_features[:, 2].clamp(min=0.0)
        pc_corr = relation_features[:, 3].clamp(min=-1.0, max=1.0)
        source_mismatch = torch.relu(profile_gap + 0.5 * l2_gap - 0.5 * similarity - 0.5 * pc_corr)
        return {
            "final": final,
            "target": target_pred,
            "sources": source_preds_t,
            "weights": source_weights,
            "source_gates": source_gates,
            "transfer_strength": transfer_strength,
            "source_logits": source_logits,
            "residuals": source_residuals,
            "raw_transfer": raw_transfer,
            "bounded_transfer": bounded_transfer,
            "calibrated_transfer": calibrated_transfer,
            "residual_budget": residual_budget,
            "source_mismatch": source_mismatch,
        }
