import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Learnable Multi-Scale Filterbank
# =============================================================================
class MultiScaleLearnableFilterbank(nn.Module):
    """
    Learnable multi-scale time-domain filterbank.

    Paper:
        v^(m) = sqrt(Conv1D^(m)(x)^2 + eps)
        s^(m) = BN^(m)[log(1 + v^(m))]

    No explicit Fourier transform or predefined spectral bands.
    """

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

        self.banks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for kernel_size in kernel_sizes:

            self.banks.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=filters_per_scale,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False,
                )
            )

            self.norms.append(
                nn.BatchNorm1d(filters_per_scale)
            )

    def forward(self, x):
        responses = []

        for conv, bn in zip(
            self.banks,
            self.norms,
        ):
            # Learnable temporal filtering
            c = conv(x)

            # Magnitude compression
            v = torch.sqrt(
                c.pow(2) + self.eps
            )

            # Log compression + normalization
            s = bn(
                torch.log1p(v)
            )

            responses.append(s)

        # Align temporal dimensions
        min_t = min(
            r.size(-1)
            for r in responses
        )

        responses = [
            r[..., :min_t]
            for r in responses
        ]

        # S = Concat_m(s^(m))
        S = torch.cat(
            responses,
            dim=1,
        )

        # [B, 1, filter, time]
        return S.unsqueeze(1)


# =============================================================================
# 2. Filterbank Encoder
# =============================================================================
class FilterbankEncoder(nn.Module):
    """
    Two-layer CNN:
        Conv2D 1 -> 16
        BN + GELU
        MaxPool 2x2
        Conv2D 16 -> 32
        BN + GELU
        GAP
        Linear + LN + GELU
    """

    def __init__(
        self,
        output_dim=48,
    ):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(32, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):

        h = self.cnn(x)
        h = h.flatten(1)

        # z_f
        z_f = self.output_projection(h)

        return z_f


# =============================================================================
# 3. Perceiver-Style Time-Domain Encoder
# =============================================================================
class PerceiverCrossAttention(nn.Module):

    def __init__(
        self,
        latent_dim,
        input_dim,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.norm_latent = nn.LayerNorm(
            latent_dim
        )

        self.norm_input = nn.LayerNorm(
            input_dim
        )

        self.input_projection = (
            nn.Linear(
                input_dim,
                latent_dim,
            )
            if input_dim != latent_dim
            else nn.Identity()
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        latents,
        inputs,
    ):

        q = self.norm_latent(
            latents
        )

        kv = self.input_projection(
            self.norm_input(inputs)
        )

        attended, _ = self.attention(
            q,
            kv,
            kv,
        )

        return latents + attended


class PerceiverSelfAttentionBlock(nn.Module):

    def __init__(
        self,
        latent_dim=48,
        num_heads=4,
        ff_mult=2,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            latent_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(
            latent_dim
        )

        self.ffn = nn.Sequential(

            nn.Linear(
                latent_dim,
                latent_dim * ff_mult,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                latent_dim * ff_mult,
                latent_dim,
            ),

            nn.Dropout(dropout),
        )

    def forward(self, x):

        h = self.norm1(x)

        x = x + self.attention(
            h,
            h,
            h,
        )[0]

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


class PerceiverTimeEncoder(nn.Module):
    """
    Paper configuration:
        N_l = 12 latent tokens
        d   = 48
        2 latent self-attention blocks
        4 attention heads
    """

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

        # phi_t(x^T)
        self.input_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                latent_dim,
            ),
            nn.LayerNorm(
                latent_dim
            ),
        )

        # P_1:T
        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                max_length,
                latent_dim,
            ) * 0.02
        )

        # Learnable latent tokens
        self.latents = nn.Parameter(
            torch.randn(
                1,
                num_latents,
                latent_dim,
            ) * 0.02
        )

        self.cross_attention = (
            PerceiverCrossAttention(
                latent_dim=latent_dim,
                input_dim=latent_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.self_attention_blocks = (
            nn.ModuleList([
                PerceiverSelfAttentionBlock(
                    latent_dim=latent_dim,
                    num_heads=num_heads,
                    ff_mult=2,
                    dropout=dropout,
                )
                for _ in range(
                    num_blocks
                )
            ])
        )

        self.final_norm = nn.LayerNorm(
            latent_dim
        )

        # rho_t
        self.output_projection = nn.Sequential(

            nn.Linear(
                latent_dim,
                output_dim,
            ),

            nn.LayerNorm(
                output_dim
            ),

            nn.GELU(),
        )

    def forward(self, x):
        """
        x: [B, T, C]
        """

        B, T, _ = x.shape

        # h_0 = phi_t(x^T) + P_1:T
        h = self.input_projection(x)

        h = (
            h
            + self.position_embedding[
                :, :T, :
            ]
        )

        # Learnable latent tokens
        latents = self.latents.expand(
            B,
            -1,
            -1,
        )

        # Cross attention
        latents = self.cross_attention(
            latents,
            h,
        )

        # Latent self-attention
        for block in (
            self.self_attention_blocks
        ):
            latents = block(latents)

        # LN(L_B)
        latents = self.final_norm(
            latents
        )

        # Mean pooling
        pooled = latents.mean(
            dim=1
        )

        # z_t
        z_t = self.output_projection(
            pooled
        )

        return z_t


# =============================================================================
# 4. Sample-Wise Conflict Estimator
# =============================================================================
class ConflictEstimator(nn.Module):
    """
    q = [
        normalized cosine similarity,
        relative L2 distance,
        temporal prediction entropy,
        filterbank prediction entropy,
        top-1 agreement
    ]

    q -> 5 -> 16 -> 1 -> sigmoid -> lambda
    """

    def __init__(
        self,
        n_classes,
        lambda_min=0.10,
        eps=1e-8,
    ):
        super().__init__()

        self.lambda_min = lambda_min
        self.eps = eps
        self.log_k = math.log(n_classes)

        self.mlp = nn.Sequential(

            nn.Linear(
                5,
                16,
            ),

            nn.GELU(),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(
        self,
        z_t,
        z_f,
        p_t,
        p_f,
    ):

        # stop-gradient for descriptor path
        zt = z_t.detach()
        zf = z_f.detach()

        pt = p_t.detach()
        pf = p_f.detach()

        # -------------------------------------------------------------
        # Normalized cosine similarity
        # -------------------------------------------------------------
        cosine = F.cosine_similarity(
            zt,
            zf,
            dim=-1,
            eps=self.eps,
        )

        c_hat = (
            1.0 + cosine
        ) / 2.0

        # -------------------------------------------------------------
        # Relative feature distance
        # -------------------------------------------------------------
        distance = torch.norm(
            zt - zf,
            p=2,
            dim=-1,
        )

        denominator = (
            torch.norm(
                zt,
                p=2,
                dim=-1,
            )
            +
            torch.norm(
                zf,
                p=2,
                dim=-1,
            )
            +
            self.eps
        )

        d_hat = (
            distance / denominator
        ).clamp(
            0.0,
            1.0,
        )

        # -------------------------------------------------------------
        # Normalized branch entropy
        # -------------------------------------------------------------
        H_t = -(
            pt
            * torch.log(
                pt + self.eps
            )
        ).sum(
            dim=-1
        ) / self.log_k

        H_f = -(
            pf
            * torch.log(
                pf + self.eps
            )
        ).sum(
            dim=-1
        ) / self.log_k

        H_t = H_t.clamp(
            0.0,
            1.0,
        )

        H_f = H_f.clamp(
            0.0,
            1.0,
        )

        # -------------------------------------------------------------
        # Top-1 prediction agreement
        # -------------------------------------------------------------
        agreement = (
            pt.argmax(dim=-1)
            ==
            pf.argmax(dim=-1)
        ).float()

        # q in R^5
        q = torch.stack(
            [
                c_hat,
                d_hat,
                H_t,
                H_f,
                agreement,
            ],
            dim=-1,
        )

        # h_c = GELU(W1 q + b1)
        # r   = sigmoid(W2 h_c + b2)
        r = torch.sigmoid(
            self.mlp(q).squeeze(-1)
        )

        # lambda =
        # lambda_min +
        # (1 - lambda_min) r
        lam = (
            self.lambda_min
            +
            (
                1.0
                - self.lambda_min
            )
            * r
        )

        return lam


# =============================================================================
# 5. Conflict-Aware Reconciliation
# =============================================================================
class ConflictAwareReconciliation(nn.Module):

    def __init__(
        self,
        feature_dim=48,
        dropout=0.1,
    ):
        super().__init__()

        # z_t_tilde = W_t z_t + b_t
        self.time_projection = nn.Linear(
            feature_dim,
            feature_dim,
        )

        # z_f_tilde = W_f z_f + b_f
        self.filterbank_projection = nn.Linear(
            feature_dim,
            feature_dim,
        )

        # G_f([z_t ; z_f ; lambda])
        self.gating_network = nn.Sequential(

            nn.Linear(
                2 * feature_dim + 1,
                feature_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                feature_dim,
                feature_dim,
            ),

            nn.Sigmoid(),
        )

        # phi_fuse
        self.fusion_projection = nn.Sequential(

            nn.Linear(
                feature_dim,
                feature_dim,
            ),

            nn.LayerNorm(
                feature_dim
            ),

            nn.GELU(),
        )

    def forward(
        self,
        z_t,
        z_f,
        lam,
    ):

        lam_e = lam.unsqueeze(-1)

        # Separate branch projections
        z_t_tilde = self.time_projection(
            z_t
        )

        z_f_tilde = (
            self.filterbank_projection(
                z_f
            )
        )

        # g = sigmoid(G_f([z_t; z_f; lambda]))
        gate_input = torch.cat(
            [
                z_t,
                z_f,
                lam_e,
            ],
            dim=-1,
        )

        g = self.gating_network(
            gate_input
        )

        # Selective fusion path
        z_cross = (
            g * z_t_tilde
            +
            (1.0 - g) * z_f_tilde
        )

        # Symmetric fallback path
        z_mean = (
            0.5
            * (z_t + z_f)
        )

        # Conflict-aware interpolation
        z_mix = (
            (1.0 - lam_e)
            * z_cross
            +
            lam_e
            * z_mean
        )

        # z_fused
        z_fused = (
            self.fusion_projection(
                z_mix
            )
        )

        return z_fused


# =============================================================================
# 6. Complete ReconcileHAR Model
# =============================================================================
class ReconcileHAR(nn.Module):

    def __init__(
        self,
        n_channels=9,
        n_classes=6,

        # Filterbank
        filters_per_scale=12,
        filter_kernel_sizes=(16, 32),
        filter_stride=12,

        # Perceiver
        num_latents=12,
        latent_dim=48,
        num_heads=4,
        num_blocks=2,

        feature_dim=48,

        dropout=0.10,

        lambda_min=0.10,

        # w in the paper
        residual_weight=0.20,
    ):
        super().__init__()

        self.residual_weight = (
            residual_weight
        )

        # -------------------------------------------------------------
        # Filterbank branch E_f
        # -------------------------------------------------------------
        self.filterbank = (
            MultiScaleLearnableFilterbank(
                in_channels=n_channels,
                filters_per_scale=(
                    filters_per_scale
                ),
                kernel_sizes=(
                    filter_kernel_sizes
                ),
                stride=filter_stride,
            )
        )

        self.filterbank_encoder = (
            FilterbankEncoder(
                output_dim=feature_dim
            )
        )

        # -------------------------------------------------------------
        # Time branch E_t
        # -------------------------------------------------------------
        self.time_encoder = (
            PerceiverTimeEncoder(
                input_dim=n_channels,
                num_latents=num_latents,
                latent_dim=latent_dim,
                num_heads=num_heads,
                num_blocks=num_blocks,
                output_dim=feature_dim,
                dropout=dropout,
            )
        )

        # -------------------------------------------------------------
        # Auxiliary classifiers C_t and C_f
        # -------------------------------------------------------------
        self.time_classifier = nn.Sequential(
            nn.Dropout(0.10),
            nn.Linear(
                feature_dim,
                n_classes,
            ),
        )

        self.filterbank_classifier = (
            nn.Sequential(
                nn.Dropout(0.10),
                nn.Linear(
                    feature_dim,
                    n_classes,
                ),
            )
        )

        # -------------------------------------------------------------
        # Conflict estimator
        # -------------------------------------------------------------
        self.conflict_estimator = (
            ConflictEstimator(
                n_classes=n_classes,
                lambda_min=lambda_min,
            )
        )

        # -------------------------------------------------------------
        # Conflict-aware reconciliation
        # -------------------------------------------------------------
        self.reconciliation = (
            ConflictAwareReconciliation(
                feature_dim=feature_dim,
                dropout=dropout,
            )
        )

        # -------------------------------------------------------------
        # Residual projection psi
        # -------------------------------------------------------------
        self.residual_projection = (
            nn.Sequential(
                nn.Linear(
                    feature_dim,
                    feature_dim,
                ),
                nn.LayerNorm(
                    feature_dim
                ),
            )
        )

        # -------------------------------------------------------------
        # Final classifier C
        # -------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                feature_dim,
                n_classes,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):
                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if (
                    module.bias
                    is not None
                ):
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                ),
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

    def extract_features(
        self,
        x,
    ):
        """
        x: [B, C, T]
        """

        # -------------------------------------------------------------
        # Filterbank-oriented representation
        # -------------------------------------------------------------
        S = self.filterbank(x)

        z_f = (
            self.filterbank_encoder(S)
        )

        # -------------------------------------------------------------
        # Time-domain representation
        # -------------------------------------------------------------
        z_t = self.time_encoder(
            x.transpose(1, 2)
        )

        # -------------------------------------------------------------
        # Auxiliary predictions
        # -------------------------------------------------------------
        o_t = self.time_classifier(
            z_t
        )

        o_f = (
            self.filterbank_classifier(
                z_f
            )
        )

        p_t = F.softmax(
            o_t,
            dim=-1,
        )

        p_f = F.softmax(
            o_f,
            dim=-1,
        )

        # -------------------------------------------------------------
        # Sample-wise conflict coefficient
        # -------------------------------------------------------------
        lam = self.conflict_estimator(
            z_t,
            z_f,
            p_t,
            p_f,
        )

        # -------------------------------------------------------------
        # Main conflict-aware reconciliation
        # -------------------------------------------------------------
        z_fused = self.reconciliation(
            z_t,
            z_f,
            lam,
        )

        # -------------------------------------------------------------
        # Fixed residual path
        #
        # z_res = psi(z_t + z_f)
        # z_out = z_fused + w z_res
        # -------------------------------------------------------------
        z_res = (
            self.residual_projection(
                z_t + z_f
            )
        )

        z_out = (
            z_fused
            +
            self.residual_weight
            * z_res
        )

        return {
            "z_t": z_t,
            "z_f": z_f,
            "z_fused": z_fused,
            "z_res": z_res,
            "z_out": z_out,
            "lambda": lam,
            "p_t": p_t,
            "p_f": p_f,
            "o_t": o_t,
            "o_f": o_f,
        }

    def forward(
        self,
        x,
    ):

        features = (
            self.extract_features(x)
        )

        # Final activity logits
        o = self.classifier(
            features["z_out"]
        )

        return (
            o,
            features["o_t"],
            features["o_f"],
            features,
        )


# =============================================================================
# 7. Training Objective
# =============================================================================
class ReconcileLoss(nn.Module):
    """
    Paper:

    L_cls =
        L_main
        + eta (L_t + L_f)

    L_agree =
        E[(1-lambda) * symmetric_KL]

    u =
        (lambda-lambda_min) /
        (1-lambda_min)

    R_ent =
        E[-u log u - (1-u) log(1-u)]

    R_center =
        (E_B[lambda] - lambda_c)^2

    L =
        L_cls
        + alpha L_agree
        - beta R_ent
        + gamma R_center
    """

    def __init__(
        self,

        # eta
        auxiliary_weight=0.30,

        # alpha
        agreement_weight=0.50,

        # beta
        entropy_weight=0.30,

        # gamma
        center_weight=0.30,

        lambda_min=0.10,
        lambda_center=0.50,

        eps=1e-8,
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.eta = auxiliary_weight
        self.alpha = agreement_weight
        self.beta = entropy_weight
        self.gamma = center_weight

        self.lambda_min = lambda_min
        self.lambda_center = (
            lambda_center
        )

        self.eps = eps

    def forward(
        self,
        logits,
        logits_t,
        logits_f,
        features,
        labels,
    ):

        lam = features["lambda"]

        p_t = features["p_t"]
        p_f = features["p_f"]

        # -------------------------------------------------------------
        # Classification objective
        # -------------------------------------------------------------
        L_main = self.ce(
            logits,
            labels,
        )

        L_t = self.ce(
            logits_t,
            labels,
        )

        L_f = self.ce(
            logits_f,
            labels,
        )

        L_cls = (
            L_main
            +
            self.eta
            * (L_t + L_f)
        )

        # -------------------------------------------------------------
        # Symmetric KL divergence
        # -------------------------------------------------------------
        KL_tf = F.kl_div(
            torch.log(
                p_t + self.eps
            ),
            p_f,
            reduction="none",
        ).sum(
            dim=-1
        )

        KL_ft = F.kl_div(
            torch.log(
                p_f + self.eps
            ),
            p_t,
            reduction="none",
        ).sum(
            dim=-1
        )

        agreement_per_sample = (
            0.5
            * (KL_tf + KL_ft)
        )

        L_agree = (
            (
                1.0 - lam
            )
            * agreement_per_sample
        ).mean()

        # -------------------------------------------------------------
        # Normalize lambda
        # -------------------------------------------------------------
        u = (
            (
                lam
                - self.lambda_min
            )
            /
            (
                1.0
                - self.lambda_min
                + self.eps
            )
        ).clamp(
            1e-6,
            1.0 - 1e-6,
        )

        # -------------------------------------------------------------
        # Entropy regularization
        # -------------------------------------------------------------
        R_ent = -(
            u * torch.log(
                u + self.eps
            )
            +
            (1.0 - u)
            * torch.log(
                1.0
                - u
                + self.eps
            )
        ).mean()

        # -------------------------------------------------------------
        # Center regularization
        # -------------------------------------------------------------
        R_center = (
            lam.mean()
            - self.lambda_center
        ).pow(2)

        # -------------------------------------------------------------
        # Complete objective
        # -------------------------------------------------------------
        total_loss = (
            L_cls
            +
            self.alpha
            * L_agree
            -
            self.beta
            * R_ent
            +
            self.gamma
            * R_center
        )

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

        return (
            total_loss,
            loss_dict,
        )
