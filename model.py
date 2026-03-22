"""
Two-Stage Prediction Model - Hybrid Architecture (Hybrid GNN-Transformer)
Combines Structure-Guided Attention (Global/Bias) and Dense GCN (Local/Aggregation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StructureGuidedAttention(nn.Module):
    """
    Structure-Guided Multi-Head Self-Attention (Bias Branch)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, geom_dim: int = 7):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Map 7-channel geometric features to bias for each attention head
        self.geom_proj = nn.Linear(geom_dim, n_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, geom_feat, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Base scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Inject structural bias
        struct_bias = self.geom_proj(geom_feat).permute(0, 3, 1, 2)
        scores = scores + struct_bias
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e4)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(context)


class DenseGCN(nn.Module):
    """
    [New] Dense Graph Convolution Layer (Dense Graph Convolution)
    Uses trRosetta distance matrix as adjacency matrix for feature aggregation
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask):
        """
        x: (Batch, Len, Dim)
        adj_mask: (Batch, Len, Len) -> Normalized adjacency matrix/contact map
        """
        # 1. Feature transformation
        x = self.proj(x) # (B, L, D)
        
        # 2. Graph Aggregation (Matrix Multiplication)
        # Adjacency Matrix @ Feature Matrix = Aggregated Neighbor Features
        # This operation forces features to flow along physical contact paths
        x = torch.matmul(adj_mask, x) # (B, L, L) @ (B, L, D) -> (B, L, D)
        
        return self.dropout(self.act(x))


class FeedForward(nn.Module):
    """Feed Forward Neural Network"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        x = self.dropout(x)
        return self.linear2(x)


class StructureTransformerLayer(nn.Module):
    """
    [Modified] Hybrid Architecture Encoder Layer
    Runs Attention (Global) and GCN (Local) in parallel
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 1. Semantic Flow: Structure-Guided Attention
        self.self_attn = StructureGuidedAttention(d_model, n_heads, dropout, geom_dim=7)
        
        # 2. Structural Flow: Dense GCN (New Branch)
        self.gcn = DenseGCN(d_model, dropout)
        
        # 3. Fusion Gating: Learn a parameter to balance contribution of Attention and GCN
        # Initialize to 0.5, letting the model learn which side to emphasize
        self.gcn_gate = nn.Parameter(torch.tensor(0.5))

        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, geom_feat, mask=None):
        """
        x: (B, L, D)
        geom_feat: (B, L, L, 7)
        mask: (B, L)
        """
        # --- A. Prepare Adjacency Matrix for GCN ---
        # geom_feat[:,:,:,0] is Distance (Expected Distance)
        dist_map = geom_feat[:, :, :, 0]
        
        # Convert distance to adjacency strength (RBF Kernel: closer distance, higher weight)
        # 5.0 is a temperature coefficient controlling decay rate
        adj = torch.exp(-dist_map / 5.0) 
        
        # If mask exists, cut off connections in padding area
        if mask is not None:
            # (B, L) -> (B, L, 1) * (B, 1, L) -> (B, L, L)
            mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(1)
            adj = adj * mask_2d
            
        # Normalize adjacency matrix (Row Normalize) to prevent feature explosion for high-degree nodes
        row_sum = adj.sum(dim=-1, keepdim=True) + 1e-6
        adj = adj / row_sum

        # --- B. Parallel Computation ---
        # 1. Attention Branch (Responsible for long-range semantics)
        # Note: We apply norm1 to x here
        attn_out = self.self_attn(self.norm1(x), geom_feat, mask)
        
        # 2. GCN Branch (Responsible for local physical microenvironment)
        gcn_out = self.gcn(self.norm1(x), adj)
        
        # --- C. Fusion ---
        # Residual Connection: Original Input + Attention + GCN
        # Simplify processing with dropout1
        # Formula: x = x + Attention(x) + Gate * GCN(x)
        x = x + self.dropout1(attn_out) + self.dropout1(gcn_out) * self.gcn_gate
        
        # --- D. FFN ---
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        
        return x


class FeatureFusion(nn.Module):
    """Feature Fusion Module"""
    def __init__(
        self,
        phychem_dim: int = 15,
        esmc_dim: int = 960,
        d_model: int = 256,
        dropout: float = 0.1,
        fusion_type: str = 'esmc_dominant',
        n_heads: int = 8
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.d_model = d_model

        self.esmc_proj = nn.Sequential(
            nn.Linear(esmc_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.phychem_proj = nn.Sequential(
            nn.Linear(phychem_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        if fusion_type == 'concat':
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        elif fusion_type == 'esmc_dominant':
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
            self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, phychem_feat, esmc_feat):
        esmc = self.esmc_proj(esmc_feat)
        phychem = self.phychem_proj(phychem_feat)

        if self.fusion_type == 'concat':
            combined = torch.cat([phychem, esmc], dim=-1)
            fused = self.fusion_proj(combined)
        elif self.fusion_type == 'gate':
            combined = torch.cat([phychem, esmc], dim=-1)
            gate = self.gate(combined)
            fused = gate * phychem + (1 - gate) * esmc
        elif self.fusion_type == 'esmc_dominant':
            fused = esmc + self.residual_weight * phychem
            fused = self.fusion_norm(fused)
        else:
            fused = esmc
            
        return fused


class TwoStagePredictor(nn.Module):
    """
    Two-Stage Prediction Model (Hybrid SG-MGT)
    """
    def __init__(
        self,
        phychem_dim: int = 15,
        esmc_dim: int = 960,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
        n_coarse_classes: int = 3,
        fusion_type: str = 'concat'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_coarse_classes = n_coarse_classes

        # Feature Fusion (1D)
        self.feature_fusion = FeatureFusion(
            phychem_dim=phychem_dim,
            esmc_dim=esmc_dim,
            d_model=d_model,
            dropout=dropout,
            fusion_type=fusion_type,
            n_heads=n_heads
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Structure-Aware Transformer Encoder (Using Hybrid Layer)
        self.encoder_layers = nn.ModuleList([
            StructureTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Global Pooling
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # First Stage: Coarse Classification Head
        self.coarse_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_coarse_classes)
        )

        # Second Stage: Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Calibrator
        self.calibrator = nn.Sequential(
            nn.Linear(d_model + n_coarse_classes, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, phychem_features, esmc_features, geom_features, mask=None):
        """
        Args:
            phychem_features: (batch, seq_len, 15)
            esmc_features: (batch, seq_len, esmc_dim)
            geom_features: (batch, seq_len, seq_len, 7)
            mask: (batch, seq_len)
        """
        # 1. 1D Feature Fusion
        x = self.feature_fusion(phychem_features, esmc_features)

        # 2. Positional Encoding
        x = self.pos_encoder(x)

        # 3. Hybrid Architecture Encoding (Transformer + GCN)
        for layer in self.encoder_layers:
            x = layer(x, geom_features, mask)

        x = self.final_norm(x)

        # 4. Global Pooling (Mean + Max)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            x_masked = x * mask_expanded
            sum_x = x_masked.sum(dim=1)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_pool = sum_x / count
            x_for_max = x_masked + (1 - mask_expanded) * (-1e4)
            max_pool = x_for_max.max(dim=1)[0]
        else:
            mean_pool = x.mean(dim=1)
            max_pool = x.max(dim=1)[0]

        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        pooled = self.pool_proj(pooled)

        # 5. Downstream Tasks
        coarse_logits = self.coarse_classifier(pooled)
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        regression_out = self.regression_head(pooled)
        
        calibrated_features = torch.cat([pooled, coarse_probs], dim=-1)
        calibrated_out = self.calibrator(calibrated_features)

        return {
            'coarse': coarse_logits,
            'coarse_probs': coarse_probs,
            'regression': regression_out,
            'calibrated': calibrated_out
        }


class TwoStageLoss(nn.Module):
    # Loss class remains unchanged
    def __init__(self, n_tasks: int = 2, use_calibrated: bool = True):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self.use_calibrated = use_calibrated

    def forward(self, outputs, targets):
        coarse_loss = F.cross_entropy(outputs['coarse'], targets['coarse_label'])
        if self.use_calibrated:
            reg_pred = outputs['calibrated']
        else:
            reg_pred = outputs['regression']
        reg_loss = F.mse_loss(reg_pred, targets['ph_value'])

        losses = [coarse_loss, reg_loss]
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]

        consistency_loss = self._compute_consistency_loss(outputs, targets)
        total_loss += 0.1 * consistency_loss

        loss_dict = {
            'coarse': coarse_loss.item(),
            'regression': reg_loss.item(),
            'consistency': consistency_loss.item(),
            'total': total_loss.item()
        }
        return total_loss, loss_dict

    def _compute_consistency_loss(self, outputs, targets):
        ph_pred = outputs['calibrated'] if self.use_calibrated else outputs['regression']
        coarse_probs = outputs['coarse_probs']
        temp = 0.5
        prob_acidic = torch.sigmoid((5 - ph_pred) / temp)
        prob_alkaline = torch.sigmoid((ph_pred - 9) / temp)
        prob_neutral = 1 - prob_acidic - prob_alkaline + prob_acidic * prob_alkaline
        ph_based_probs = torch.cat([prob_acidic, prob_neutral, prob_alkaline], dim=-1)
        ph_based_probs = F.softmax(ph_based_probs, dim=-1)
        consistency_loss = F.kl_div(
            torch.log(coarse_probs + 1e-8),
            ph_based_probs.detach(),
            reduction='batchmean'
        )
        return consistency_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)