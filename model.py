import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Learnable Multi-Scale Filterbank
# =============================================================================
class MultiScaleLearnableFilterbank(nn.Module):

    def __init__(
        self,
        in_channels,
        filters_per_scale=12,
        kernel_sizes=(16, 32),
        stride=12,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.banks = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    filters_per_scale,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False,
                )
                for kernel_size in kernel_sizes
            ]
        )

        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(filters_per_scale) for _ in kernel_sizes]
        )

    def forward(self, x):
        responses = []

        for conv, bn in zip(self.banks, self.norms):
            v = torch.sqrt(conv(x).pow(2) + self.eps)
            responses.append(bn(torch.log1p(v)))

        min_t = min(r.size(-1) for r in responses)
        responses = [r[..., :min_t] for r in responses]

        return torch.cat(responses, dim=1).unsqueeze(1)


# =============================================================================
# 2. Filterbank Encoder
# =============================================================================
class FilterbankEncoder(nn.Module):

    def __init__(self, output_dim=48):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(32, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.output_projection(self.cnn(x).flatten(1))


# =============================================================================
# 3. Perceiver-Style Time-Domain Encoder
# =============================================================================
class PerceiverSelfAttentionBlock(nn.Module):

    def __init__(
        self,
        latent_dim=48,
        num_heads=4,
        ff_mult=2,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(latent_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(latent_dim)

        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * ff_mult, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attention(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class PerceiverTimeEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        num_latents=12,
        latent_dim=48,
        num_heads=4,
        num_blocks=2,
        output_dim=48,
        dropout=0.1,
        max_length=512,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.position_embedding = nn.Parameter(
            torch.randn(1, max_length, latent_dim) * 0.02
        )

        self.latents = nn.Parameter(
            torch.randn(1, num_latents, latent_dim) * 0.02
        )

        self.norm_latent = nn.LayerNorm(latent_dim)
        self.norm_input = nn.LayerNorm(latent_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attention_blocks = nn.ModuleList(
            [
                PerceiverSelfAttentionBlock(
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    ff_mult=2,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(latent_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        B, T, _ = x.shape

        h = self.input_projection(x) + self.position_embedding[:, :T, :]

        latents = self.latents.expand(B, -1, -1)
        q = self.norm_latent(latents)
        kv = self.norm_input(h)
        latents = latents + self.cross_attention(q, kv, kv, need_weights=False)[0]

        for block in self.self_attention_blocks:
            latents = block(latents)

        return self.output_projection(self.final_norm(latents).mean(dim=1))


# =============================================================================
# 4. Sample-Wise Conflict Estimator
# =============================================================================
class ConflictEstimator(nn.Module):

    def __init__(
        self,
        n_classes,
        lambda_min=0.10,
    ):
        super().__init__()

        self.lambda_min = lambda_min
        self.log_k = math.log(max(2, n_classes))

        self.mlp = nn.Sequential(
            nn.Linear(5, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, z_t, z_f, o_t, o_f):
        zt = z_t.detach()
        zf = z_f.detach()
        pt = F.softmax(o_t.detach(), dim=-1)
        pf = F.softmax(o_f.detach(), dim=-1)

        c_hat = 0.5 * (1.0 + F.cosine_similarity(zt, zf, dim=-1, eps=1e-6))

        distance = torch.linalg.vector_norm(zt - zf, dim=-1)
        denominator = (
            torch.linalg.vector_norm(zt, dim=-1)
            + torch.linalg.vector_norm(zf, dim=-1)
            + 1e-6
        )
        d_hat = (distance / denominator).clamp(0.0, 1.0)

        H_t = -(pt * (pt + 1e-8).log()).sum(dim=-1) / self.log_k
        H_f = -(pf * (pf + 1e-8).log()).sum(dim=-1) / self.log_k

        agreement = (pt.argmax(dim=-1) == pf.argmax(dim=-1)).float()

        q = torch.stack([c_hat, d_hat, H_t, H_f, agreement], dim=-1)

        r = torch.sigmoid(self.mlp(q).squeeze(-1))

        return self.lambda_min + (1.0 - self.lambda_min) * r


# =============================================================================
# 5. Conflict-Aware Reconciliation
# =============================================================================
class ConflictAwareReconciliation(nn.Module):

    def __init__(
        self,
        feature_dim=48,
        lambda_min=0.10,
    ):
        super().__init__()

        self.lambda_min = lambda_min

        self.time_projection = nn.Linear(feature_dim, feature_dim)
        self.filterbank_projection = nn.Linear(feature_dim, feature_dim)

        self.gating_network = nn.Sequential(
            nn.Linear(2 * feature_dim + 1, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

        self.fusion_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, z_t, z_f, lam):
        lam_e = lam.reshape(-1).clamp(self.lambda_min, 1.0).unsqueeze(-1)

        z_t_tilde = self.time_projection(z_t)
        z_f_tilde = self.filterbank_projection(z_f)

        g = self.gating_network(torch.cat([z_t, z_f, lam_e], dim=-1))

        z_cross = g * z_t_tilde + (1.0 - g) * z_f_tilde
        z_mean = 0.5 * (z_t + z_f)
        z_mix = (1.0 - lam_e) * z_cross + lam_e * z_mean

        return self.fusion_projection(z_mix)


# =============================================================================
# 6. Complete Model
# =============================================================================
class ProposedReconciliationModel(nn.Module):

    def __init__(
        self,
        n_channels=9,
        n_classes=6,

        filters_per_scale=12,
        filter_kernel_sizes=(16, 32),
        filter_stride=12,

        num_latents=12,
        latent_dim=48,
        num_heads=4,
        num_blocks=2,

        feature_dim=48,
        dropout=0.10,
        lambda_min=0.10,
        residual_weight=0.20,
    ):
        super().__init__()

        self.residual_weight = residual_weight

        self.time_encoder = PerceiverTimeEncoder(
            input_dim=n_channels,
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            output_dim=feature_dim,
            dropout=dropout,
        )

        self.filterbank = MultiScaleLearnableFilterbank(
            in_channels=n_channels,
            filters_per_scale=filters_per_scale,
            kernel_sizes=filter_kernel_sizes,
            stride=filter_stride,
        )

        self.filterbank_encoder = FilterbankEncoder(output_dim=feature_dim)

        self.time_classifier = nn.Linear(feature_dim, n_classes)
        self.filterbank_classifier = nn.Linear(feature_dim, n_classes)

        self.conflict_estimator = ConflictEstimator(
            n_classes=n_classes,
            lambda_min=lambda_min,
        )

        self.reconciliation = ConflictAwareReconciliation(
            feature_dim=feature_dim,
            lambda_min=lambda_min,
        )

        self.residual_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(feature_dim, n_classes)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def extract_features(self, x):
        z_t = self.time_encoder(x.transpose(1, 2))
        z_f = self.filterbank_encoder(self.filterbank(x))

        o_t = self.time_classifier(z_t)
        o_f = self.filterbank_classifier(z_f)

        lam = self.conflict_estimator(z_t, z_f, o_t, o_f)

        z_fused = self.reconciliation(z_t, z_f, lam)
        z_res = self.residual_projection(z_t + z_f)
        z_out = z_fused + self.residual_weight * z_res

        return {
            "z_t": z_t,
            "z_f": z_f,
            "z_fused": z_fused,
            "z_res": z_res,
            "z_out": z_out,
            "lambda": lam,
            "o_t": o_t,
            "o_f": o_f,
        }

    def forward(self, x):
        features = self.extract_features(x)
        o = self.classifier(features["z_out"])
        return o, features["o_t"], features["o_f"], features


# =============================================================================
# 7. Training Objective
# =============================================================================
class ReconcileLoss(nn.Module):

    def __init__(
        self,
        auxiliary_weight=0.30,
        agreement_weight=0.50,
        entropy_weight=0.30,
        center_weight=0.30,
        lambda_min=0.10,
        lambda_center=0.50,
    ):
        super().__init__()

        self.eta = auxiliary_weight
        self.alpha = agreement_weight
        self.beta = entropy_weight
        self.gamma = center_weight
        self.lambda_min = lambda_min
        self.lambda_center = lambda_center

    def forward(self, logits, logits_t, logits_f, features, labels):
        lam = features["lambda"]

        L_main = F.cross_entropy(logits, labels)
        L_t = F.cross_entropy(logits_t, labels)
        L_f = F.cross_entropy(logits_f, labels)
        L_cls = L_main + self.eta * (L_t + L_f)

        p_t = F.softmax(logits_t, dim=-1)
        p_f = F.softmax(logits_f, dim=-1)
        log_p_t = F.log_softmax(logits_t, dim=-1)
        log_p_f = F.log_softmax(logits_f, dim=-1)
        KL_tf = (p_t * (log_p_t - log_p_f)).sum(dim=-1)
        KL_ft = (p_f * (log_p_f - log_p_t)).sum(dim=-1)
        L_agree = ((1.0 - lam) * 0.5 * (KL_tf + KL_ft)).mean()

        u = ((lam - self.lambda_min) / (1.0 - self.lambda_min)).clamp(1e-6, 1.0 - 1e-6)
        R_ent = -(u * u.log() + (1.0 - u) * (1.0 - u).log()).mean()

        R_center = (lam.mean() - self.lambda_center).pow(2)

        total_loss = L_cls + self.alpha * L_agree - self.beta * R_ent + self.gamma * R_center

        loss_dict = {
            "total": total_loss,
            "main_ce": L_main,
            "time_ce": L_t,
            "filterbank_ce": L_f,
            "agreement": L_agree,
            "entropy_reg": R_ent,
            "center_reg": R_center,
            "lambda_mean": lam.mean(),
        }

        return total_loss, loss_dict
