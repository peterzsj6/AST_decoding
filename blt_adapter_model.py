from typing import Dict, List, Optional, Tuple
import os
# Note: CUDA_VISIBLE_DEVICES should be set by the calling script, not here
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2ForCausalLM, Qwen2Config


# =========================
# Span type vocabulary and rules
# =========================
SPAN_TYPE_LIST = [
    # Special/unknown first to ensure id 0
    "unknown",
    # Operators and punctuation-likes
    "%=", "&=", "**=", "*=", "+=", "-=", "//=", "/=", ":=", "<<=", "<>", "=", ">>=", "@", "@=", "\\", "^=", "|=",
    # Misc special tokens
    "ERROR", "_", "__future__",
    # AST structural/types
    "aliased_import", "argument_list", "as_pattern", "as_pattern_target",
    "assert_statement", "attribute", "block", "call",
    "case", "case_clause", "case_pattern", "class_definition",
    # Comments and strings (docstrings fall under 'string')
    "comment",
    "concatenated_string",
    "conditional_expression", "decorated_definition", "decorator",
    "default_parameter", "delete_statement", "dictionary", "dictionary_splat",
    "elif_clause", "ellipsis", "else_clause",
    "escape_interpolation", "escape_sequence",
    "except*", "except_clause", "exec",
    "expression_list", "expression_statement",
    "false", "finally_clause", "for_in_clause", "for_statement",
    "format_specifier", "function_definition", "future_import_statement",
    "generator_expression", "global_statement",
    "identifier", "if_clause", "if_statement",
    "import_from_statement", "import_statement",
    "in", "is", "is not",
    "keyword", "keyword_argument",
    "lambda_parameters", "line_continuation", "list", "list_comprehension", "list_splat",
    "match", "module",
    "named_expression", "none", "nonlocal_statement",
    "not in", "not_operator",
    "number", "operator",
    "pair", "parameters", "parenthesized_expression", "pattern_list",
    "print", "print_statement", "punctuation",
    "raise_statement", "return_statement",
    "set", "slice",
    "string", "string_end", "string_start",
    "subscript",
    "true", "try_statement", "tuple", "tuple_pattern",
    "type", "type_conversion", "typed_default_parameter", "typed_parameter",
    "while_statement", "with_clause", "with_item", "with_statement",
]
SPAN_TYPE_TO_ID: Dict[str, int] = {t: i for i, t in enumerate(SPAN_TYPE_LIST)}
ID_TO_SPAN_TYPE: Dict[int, str] = {i: t for t, i in SPAN_TYPE_TO_ID.items()}

# Treat these types as "textual": split any multi-token span into per-token singletons
TEXTUAL_SPAN_TYPES = {
    "comment",
    "string",
    "string_start",
    "string_end",
    "concatenated_string",
    "escape_sequence",
    "escape_interpolation",
}


class MeanPooledSpanEncoder(nn.Module):
    """
    Mean pooling local encoder over AST spans.
    - Produces token embeddings (for feeding the global/latent transformer)
    - Enables deriving a single latent vector per span via mean pooling
    """
    def __init__(self, config: Qwen2Config, span_dropout_prob: float = 0.0):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # Small trainable adapter for light adaptation without shifting the base distribution too much
        self.token_adapter = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        # Initialize adapter to near-identity
        nn.init.normal_(self.token_adapter[0].weight, std=0.02)
        nn.init.zeros_(self.token_adapter[0].bias)
        nn.init.normal_(self.token_adapter[2].weight, std=0.02)
        nn.init.zeros_(self.token_adapter[2].bias)

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Optional span dropout (entire sample)
        self.span_dropout_prob = float(span_dropout_prob)

    @property
    def weight(self):
        # Allow HF tie-weights flow to locate the weight parameter
        return self.token_embeddings.weight

    def forward(self, input_ids: torch.Tensor, span_metadata: Optional[Dict] = None) -> torch.Tensor:
        """
        Returns per-token embeddings. Pooled span representations are computed externally as needed.
        """
        token_emb = self.token_embeddings(input_ids)  # [B, L, H]
        token_emb = token_emb + self.token_adapter(token_emb)
        token_emb = self.layer_norm(token_emb)
        return self.dropout(token_emb)


class LocalCrossAttentionBlock(nn.Module):
    """
    Decoder-style block with:
      - causal self-attention over token sequence
      - cross-attention from tokens (queries) to a span latent (keys/values)
      - feed-forward network
    """
    def __init__(self, hidden_size: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.dim_ff = dim_ff

        self.ln1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=nhead, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=nhead, dropout=dropout, batch_first=True)

        self.ln3 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, hidden_size),
            nn.Dropout(dropout),
        )

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        # True where masked
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor, span_latent: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, H] tokens
        span_latent: [B, 1, H] latent per span (K/V)
        """
        bsz, tlen, _ = x.size()
        device = x.device

        # Causal self-attention
        xm = self.ln1(x)
        causal_mask = self._causal_mask(tlen, device)  # [T, T] for batch_first=True
        sa_out, _ = self.self_attn(xm, xm, xm, attn_mask=causal_mask)
        x = x + sa_out

        # Cross-attention (queries = tokens, keys/values = span latent)
        xm = self.ln2(x)
        # span_latent already [B, 1, H]
        ca_out, _ = self.cross_attn(xm, span_latent, span_latent)
        x = x + ca_out

        # Feed-forward
        xm = self.ln3(x)
        ff_out = self.ff(xm)
        x = x + ff_out
        return x


class LocalCausalTransformer(nn.Module):
    """
    Stack of decoder blocks with cross-attention to a per-span latent representation.
    """
    def __init__(self, hidden_size: int, nhead: int, dim_ff: int, num_layers: int = 2, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len + 1, hidden_size)
        self.layers = nn.ModuleList([
            LocalCrossAttentionBlock(hidden_size, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, tok_emb: torch.Tensor, span_latent: torch.Tensor) -> torch.Tensor:
        """
        tok_emb: [B, T, H]
        span_latent: [B, 1, H]
        """
        bsz, tlen, h = tok_emb.shape
        pos_ids = torch.arange(tlen, device=tok_emb.device).unsqueeze(0).expand(bsz, tlen)
        x = tok_emb + self.pos_embed(torch.clamp(pos_ids, max=self.pos_embed.num_embeddings - 1))

        for layer in self.layers:
            x = layer(x, span_latent)
        return x


class SDPAMultiheadCrossAttention(nn.Module):
    """
    Multihead cross-attention implemented via torch SDPA.

    Key feature for memory: supports using a single K/V batch (Bk=1) while queries
    are batched (Bq=N_nodes). We avoid [Bq, S, H] materialization by flattening
    queries into a single sequence length (Lq=Bq*T) with batch=1.
    """
    def __init__(self, hidden_size: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % nhead != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by nhead ({nhead})")
        self.hidden_size = int(hidden_size)
        self.nhead = int(nhead)
        self.head_dim = int(hidden_size // nhead)
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        x_q: torch.Tensor,                 # [Bq, T, H]
        mem_kv: torch.Tensor,              # [Bk, S, H] (we expect Bk==1 for global memory)
        key_padding_mask: Optional[torch.Tensor] = None,  # [Bk, S] True=mask/ignore
    ) -> torch.Tensor:
        if x_q.dim() != 3 or mem_kv.dim() != 3:
            raise ValueError(f"Expected x_q/mem_kv to be 3D, got {x_q.shape=} {mem_kv.shape=}")
        bq, t, h = x_q.shape
        bk, s, hk = mem_kv.shape
        if h != self.hidden_size or hk != self.hidden_size:
            raise ValueError(f"Hidden size mismatch: {h=} {hk=} expected {self.hidden_size}")
        if bk != 1:
            # We only need the broadcastable case for the global memory optimization.
            # If needed later, we can extend to bk==bq via chunking or a second path.
            raise ValueError(f"SDPAMultiheadCrossAttention currently expects mem_kv batch=1, got {bk}")

        # Project
        q = self.q_proj(x_q)     # [Bq, T, H]
        k = self.k_proj(mem_kv)  # [1, S, H]
        v = self.v_proj(mem_kv)  # [1, S, H]

        # Reshape to heads
        q = q.view(bq, t, self.nhead, self.head_dim).permute(0, 2, 1, 3).contiguous()  # [Bq, nh, T, hd]
        k = k.view(1, s, self.nhead, self.head_dim).permute(0, 2, 1, 3).contiguous()   # [1, nh, S, hd]
        v = v.view(1, s, self.nhead, self.head_dim).permute(0, 2, 1, 3).contiguous()   # [1, nh, S, hd]

        # Flatten queries: treat (Bq*T) as one long query sequence in batch=1
        q_flat = q.reshape(1, self.nhead, bq * t, self.head_dim)  # [1, nh, Lq, hd]

        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [1, S] True=ignore. SDPA bool mask: True=masked.
            if key_padding_mask.dim() != 2 or key_padding_mask.shape[0] != 1 or key_padding_mask.shape[1] != s:
                raise ValueError(f"Expected key_padding_mask shape [1, S], got {key_padding_mask.shape}")
            kpm = key_padding_mask.to(torch.bool)
            # [1, 1, 1, S] -> expand to [1, 1, Lq, S] (no materialization)
            attn_mask = kpm.view(1, 1, 1, s).expand(1, 1, bq * t, s)

        dropout_p = self.dropout if self.training and self.dropout > 0 else 0.0
        out = F.scaled_dot_product_attention(
            q_flat,           # [1, nh, Lq, hd]
            k,                # [1, nh, S, hd]
            v,                # [1, nh, S, hd]
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )  # [1, nh, Lq, hd]

        # Reshape back to [Bq, T, H]
        out = out.squeeze(0)  # [nh, Lq, hd]
        out = out.view(self.nhead, bq, t, self.head_dim).permute(1, 2, 0, 3).contiguous()  # [Bq, T, nh, hd]
        out = out.view(bq, t, self.hidden_size)  # [Bq, T, H]
        return self.out_proj(out)


class LocalHybridDecoderBlock(nn.Module):
    """
    Decoder block with:
      - causal self-attention over token sequence
      - cross-attention to global memory (global transformer token-wise hidden states)
      - cross-attention to span memory (local encoder token-wise info) OR fallback to span latent
      - feed-forward
    """
    def __init__(self, hidden_size: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.dim_ff = dim_ff

        self.ln1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=nhead, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(hidden_size)
        # Global cross-attention: SDPA path avoids materializing per-node global memory.
        self.use_sdpa_global_attn: bool = True
        self.cross_attn_global_sdpa = SDPAMultiheadCrossAttention(hidden_size, nhead, dropout=dropout)
        # Fallback path (kept for safety/debug)
        self.cross_attn_global = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=nhead, dropout=dropout, batch_first=True)

        self.ln3 = nn.LayerNorm(hidden_size)
        self.cross_attn_span = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=nhead, dropout=dropout, batch_first=True)

        self.ln4 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, hidden_size),
            nn.Dropout(dropout),
        )

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        # True where masked
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        span_latent: torch.Tensor,
        span_memory: Optional[torch.Tensor] = None,
        span_key_padding_mask: Optional[torch.Tensor] = None,
        global_memory: Optional[torch.Tensor] = None,
        global_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, T, H] tokens
        span_latent: [B, 1, H]
        span_memory: [B, S, H] or None
        span_key_padding_mask: [B, S] (True to mask/ignore)
        global_memory: [B, G, H] or None
        global_key_padding_mask: [B, G] (True to mask/ignore)
        """
        bsz, tlen, _ = x.size()
        device = x.device

        # Causal self-attention
        xm = self.ln1(x)
        causal_mask = self._causal_mask(tlen, device)  # [T, T] for batch_first=True
        sa_out, _ = self.self_attn(xm, xm, xm, attn_mask=causal_mask)
        x = x + sa_out

        # Cross-attention to global memory (if provided)
        xm = self.ln2(x)
        if global_memory is not None:
            if self.use_sdpa_global_attn and global_memory.size(0) == 1:
                ga_out = self.cross_attn_global_sdpa(xm, global_memory, key_padding_mask=global_key_padding_mask)
            else:
                ga_out, _ = self.cross_attn_global(xm, global_memory, global_memory, key_padding_mask=global_key_padding_mask)
            x = x + ga_out

        # Cross-attention to span memory (fallback to latent if memory is None)
        xm = self.ln3(x)
        if span_memory is None:
            kv = span_latent  # [B,1,H]
            span_kpm = None
        else:
            kv = span_memory  # [B,S,H]
            span_kpm = span_key_padding_mask
        ca_out, _ = self.cross_attn_span(xm, kv, kv, key_padding_mask=span_kpm)
        x = x + ca_out

        # Feed-forward
        xm = self.ln4(x)
        ff_out = self.ff(xm)
        x = x + ff_out
        return x


class LocalHybridDecoder(nn.Module):
    """
    Stack of decoder blocks with dual cross-attention:
      - span memory (or latent fallback)
      - global memory (token-level)
    """
    def __init__(self, hidden_size: int, nhead: int, dim_ff: int, num_layers: int = 2, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len + 1, hidden_size)
        self.layers = nn.ModuleList([
            LocalHybridDecoderBlock(hidden_size, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])

    def forward(
        self,
        tok_emb: torch.Tensor,
        span_latent: torch.Tensor,
        span_memory: Optional[torch.Tensor] = None,
        span_key_padding_mask: Optional[torch.Tensor] = None,
        global_memory: Optional[torch.Tensor] = None,
        global_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tok_emb: [B, T, H]
        span_latent: [B, 1, H]
        span_memory: [B, S, H] or None
        span_key_padding_mask: [B, S] (True=mask)
        global_memory: [B, G, H] or None
        global_key_padding_mask: [B, G] (True=mask)
        """
        bsz, tlen, h = tok_emb.shape
        pos_ids = torch.arange(tlen, device=tok_emb.device).unsqueeze(0).expand(bsz, tlen)
        x = tok_emb + self.pos_embed(torch.clamp(pos_ids, max=self.pos_embed.num_embeddings - 1))
        for layer in self.layers:
            x = layer(
                x,
                span_latent=span_latent,
                span_memory=span_memory,
                span_key_padding_mask=span_key_padding_mask,
                global_memory=global_memory,
                global_key_padding_mask=global_key_padding_mask,
            )
        return x


class BLTAdapterModel(Qwen2ForCausalLM):
    """
    BLT-style adapter:
      - Local encoder: mean-pooled span representations from per-token embeddings
      - Global/latent transformer: Qwen2.5 Coder (frozen or partially unfrozen)
      - Local decoder: small causal Transformer with cross-attention to span latent
    """
    def __init__(
        self,
        config: Qwen2Config,
        local_num_layers: int = 2,
        local_dropout: float = 0.1,
        max_node_length: int = 64,
        boundary_loss_weight: float = 0.1,
        latent_mse_weight: float = 0.1,
        num_node_types: Optional[int] = None,
        boundary_class_weight: Optional[torch.Tensor] = None,  # [2] tensor for class weights
        boundary_focal_gamma: float = 0.0,  # Focal loss gamma (0.0 = disabled, 2.0 = standard)
    ):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_node_length = int(max_node_length)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.latent_mse_weight = float(latent_mse_weight)
        # IMPORTANT: keep probe-head output dimension stable across save/load.
        # When resuming via from_pretrained(), the constructor may be called without explicit num_node_types.
        # In that case, read from config if present; otherwise fall back to the span-type vocab size.
        if num_node_types is None:
            try:
                num_node_types = int(getattr(config, "num_node_types"))
            except Exception:
                num_node_types = int(len(SPAN_TYPE_LIST))
        self.num_node_types = int(num_node_types)
        # Persist into config so future checkpoints can resume without mismatched probe dims.
        try:
            self.config.num_node_types = int(self.num_node_types)
        except Exception:
            pass
        # Class weighting for boundary head (to handle class imbalance)
        if boundary_class_weight is not None:
            self.register_buffer('boundary_class_weight', boundary_class_weight)
        else:
            self.boundary_class_weight = None
        self.boundary_focal_gamma = float(boundary_focal_gamma)
        # Probe controls can be toggled externally (e.g., from train_main)
        self.probe_only: bool = False
        # Additional weights and temperatures (can be updated during training)
        self.node_recon_loss_weight: float = 1.0
        self.lm_loss_weight: float = 0.0
        self.kl_weight: float = 0.0
        self.infonce_weight: float = 0.0
        self.infonce_tau: float = 0.07
        # Weight for probe losses when not in probe-only mode
        self.probe_loss_weight: float = 0.1
        # Whether boundary head treats single-token spans as positives (default: only starts)
        self.boundary_include_singles: bool = False

        # Keep the base/global embedding regular; add a separate node token encoder
        self.node_token_encoder = MeanPooledSpanEncoder(config)
        textual_ids = [
            SPAN_TYPE_TO_ID[t]
            for t in TEXTUAL_SPAN_TYPES
            if t in SPAN_TYPE_TO_ID
        ]
        if len(textual_ids) == 0:
            textual_tensor = torch.empty(0, dtype=torch.long)
        else:
            textual_tensor = torch.tensor(sorted(set(textual_ids)), dtype=torch.long)
        self.register_buffer(
            "textual_span_type_ids",
            textual_tensor,
            persistent=False,
        )

        # Local decoder components
        nhead = max(1, self.hidden_size // 64)
        dim_ff = max(self.hidden_size * 4, 512)
        self.local_token_embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.latent_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # Hybrid decoder: supports span- and global-memory cross-attention
        self.local_decoder = LocalHybridDecoder(
            hidden_size=self.hidden_size,
            nhead=nhead,
            dim_ff=dim_ff,
            num_layers=local_num_layers,
            dropout=local_dropout,
            max_len=self.max_node_length + 1,
        )
        # Backward-compat alias for any old calls
        self.local_transformer = self.local_decoder
        self.local_out_proj = nn.Linear(self.hidden_size, self.vocab_size)

        # Tie and freeze large vocab projections/embeddings to avoid duplicating VxH params
        try:
            # Tie local token embed to base token embeddings
            self.local_token_embed.weight = self.model.embed_tokens.weight  # type: ignore[attr-defined]
            self.local_token_embed.weight.requires_grad = False
        except Exception:
            pass
        try:
            # Tie local output projection to base lm_head
            self.local_out_proj.weight = self.lm_head.weight  # type: ignore[attr-defined]
            self.local_out_proj.weight.requires_grad = False
        except Exception:
            pass
        # LoRA adapters will be applied via PEFT during training if enabled

        # Optional auxiliary heads (node type / node length bins) - legacy
        self.node_type_head = None  # Set externally if needed (legacy path)
        self.node_len_head = None   # Set externally if needed

        # Probe heads for node type classification (encoder latent and decoder node repr)
        self.node_type_probe_encoder = nn.Linear(self.hidden_size, self.num_node_types)
        self.node_type_probe_decoder = nn.Linear(self.hidden_size, self.num_node_types)

        # BOS/eos handling for local decoding
        self.node_bos_id = getattr(config, 'bos_token_id', getattr(config, 'eos_token_id', 0))

        # Learned patch boundary head (binary classification: 1=start/single; 0=otherwise)
        self.boundary_head = nn.Linear(self.hidden_size, 2)
        # Optional boundary feature projection for multi-layer features (mid+last concatenation).
        # Kept lightweight to avoid impacting the base LM; used only for boundary prediction.
        self.boundary_feat_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # Latent-from-global projector to predict span latent from global hidden at boundary
        self.latent_from_global = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        # Cap for nodes per sample during training to bound memory
        self.max_nodes_per_sample = 16
        
        # === NEW: Combined latent from encoder + global (Option 2) ===
        # Projects concatenated [encoder_latent, global_latent] to span_latent
        self.latent_combine = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        # Initialize to favor encoder latent initially (more stable)
        with torch.no_grad():
            # First half of input (encoder) gets higher weight initially
            self.latent_combine[0].weight[:, :self.hidden_size] *= 1.0
            self.latent_combine[0].weight[:, self.hidden_size:] *= 0.5
        
        # === NEW: Residual connection from global hidden (for single-token shortcut) ===
        # Learnable scale for residual, initialized small so local decoder dominates initially
        self.global_residual_scale = nn.Parameter(torch.tensor(0.1))
        # Gate to learn when to use residual vs local decoder output
        self.global_residual_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

    def _resolve_hidden_index(self, hidden_states: List[torch.Tensor], idx: int) -> int:
        """
        Resolve a possibly-negative hidden state index safely into [0, len(hidden_states)-1].
        """
        n = int(len(hidden_states))
        if n <= 0:
            return 0
        i = int(idx)
        if i < 0:
            i = n + i
        i = max(0, min(n - 1, i))
        return i

    def compute_boundary_logits(
        self,
        *,
        last_hidden: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute boundary logits from global hidden states.
        - last_hidden: [B,L,H] (typically final layer)
        - hidden_states: optional list of [B,L,H] from the global transformer
        Returns: [B,L,2]
        """
        mode = str(getattr(self, "boundary_feature_mode", "last"))
        if mode == "concat_mid_last" and hidden_states is not None and len(hidden_states) >= 2:
            mid_idx = int(getattr(self, "boundary_mid_layer", -2))
            mi = self._resolve_hidden_index(hidden_states, mid_idx)
            mid = hidden_states[mi]
            # Ensure same dtype/device
            if mid.dtype != last_hidden.dtype:
                mid = mid.to(dtype=last_hidden.dtype)
            if mid.device != last_hidden.device:
                mid = mid.to(device=last_hidden.device)
            feat2 = torch.cat([mid, last_hidden], dim=-1)  # [B,L,2H]
            feat = self.boundary_feat_proj(feat2)  # [B,L,H]
            return self.boundary_head(feat)
        # Fallback: last-layer features
        return self.boundary_head(last_hidden)

    def copy_base_embeddings_from(self, base: Qwen2ForCausalLM) -> None:
        """
        Copy token embeddings from a base Qwen model into our node token encoder.
        """
        with torch.no_grad():
            self.node_token_encoder.token_embeddings.weight.copy_(base.model.embed_tokens.weight)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        span_metadata: Optional[Dict] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Standard causal LM forward for the global transformer (Qwen2.5),
        plus optional BLT-style local node reconstruction loss when labels+spans are provided.
        """
        # Compute token embeddings
        # - global_inputs_embeds: regular base embedding for the global transformer
        # - node_inputs_embeds: node token encoder outputs for span/node processing
        #
        # IMPORTANT (inference perf):
        # `node_token_encoder(...)` is only needed when we actually compute span/node losses
        # (i.e., when span_metadata is provided). For plain LM inference, skipping it
        # saves a substantial amount of compute per token.
        global_inputs_embeds = self.model.embed_tokens(input_ids)  # [B, L, H]
        if span_metadata is not None:
            _ = self.node_token_encoder(input_ids, span_metadata)  # [B, L, H]

        # Forward through global transformer using inputs_embeds
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ['input_ids', 'inputs_embeds', 'labels', 'output_hidden_states', 'return_dict']
        }
        # We need last hidden states for auxiliary heads when span metadata is provided (training OR validation)
        need_hidden_states = bool(span_metadata is not None and attention_mask is not None)
        # Respect caller request while ensuring we have hidden states when needed
        want_hidden_states = bool(kwargs.get('output_hidden_states', False) or need_hidden_states)
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=global_inputs_embeds,
            labels=labels,
            output_hidden_states=want_hidden_states,
            return_dict=True,  # force dict to access fields reliably
            **filtered_kwargs,
        )

        # Teacher-forced global argmax ids (for scheduled sampling of local-encoder span tokens)
        # Kept detached: it is used as discrete conditioning, not a differentiable path.
        global_argmax_ids = None
        try:
            if str(getattr(self, "span_ss_mode", "off")) != "off":
                if hasattr(outputs, "logits") and outputs.logits is not None:
                    global_argmax_ids = outputs.logits.argmax(dim=-1).detach()
                    outputs.global_argmax_ids = global_argmax_ids
        except Exception:
            global_argmax_ids = None

        # Preserve raw LM cross-entropy before composing total loss
        total_loss = None
        base_lm_ce = outputs.loss if hasattr(outputs, "loss") else None
        if base_lm_ce is not None:
            # Expose raw LM CE with or without gradient based on a runtime flag.
            # When training global CE externally, we want to keep gradients.
            expose_grad = bool(getattr(self, "expose_lm_ce_grad", False))
            try:
                outputs.lm_ce = base_lm_ce if expose_grad else base_lm_ce.detach()
            except Exception:
                outputs.lm_ce = base_lm_ce
            total_loss = self.lm_loss_weight * base_lm_ce

        # Add learned boundary + latent regression losses if spans are present (training OR validation)
        if span_metadata is not None and attention_mask is not None:
            # Retrieve the last hidden state from outputs; fallback to base model if needed
            last_hidden = None
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                last_hidden = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1]
            if last_hidden is None:
                # As a safety fallback (should not happen when need_hidden_states=True), run the base model
                base_out = self.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    inputs_embeds=global_inputs_embeds,
                    use_cache=False,
                    return_dict=True,
                )
                last_hidden = base_out.last_hidden_state  # [B, L, H]
            # Do not backprop into the global transformer for auxiliary heads to save memory
            last_hidden_for_heads = last_hidden.detach() if isinstance(last_hidden, torch.Tensor) else last_hidden

            # Boundary supervision from span_metadata['boundaries'] (1=start, 3=single => positive)
            if isinstance(last_hidden_for_heads, torch.Tensor) and 'boundaries' in span_metadata:
                boundaries = span_metadata['boundaries'].to(last_hidden_for_heads.device)  # [B, L]
                include_singles = bool(getattr(self, "boundary_include_singles", False))
                pos_mask = (boundaries == 1) | ((boundaries == 3) & include_singles)
                span_types = span_metadata.get('span_types')
                if (
                    isinstance(span_types, torch.Tensor)
                    and hasattr(self, "textual_span_type_ids")
                    and self.textual_span_type_ids.numel() > 0
                ):
                    span_types = span_types.to(boundaries.device)
                    textual_mask = torch.isin(span_types, self.textual_span_type_ids.to(boundaries.device))
                    pos_mask = pos_mask & (~textual_mask)

                # Optional: rewrite-worthy boundary targets.
                # Gate boundary positives to spans where rewriting is actually needed, measured by
                # teacher-forced global argmax != gold inside the span.
                try:
                    mode = str(getattr(self, "boundary_target_mode", "ast_start"))
                except Exception:
                    mode = "ast_start"
                if (
                    mode == "rewrite_worthy"
                    and global_argmax_ids is not None
                    and isinstance(input_ids, torch.Tensor)
                    and isinstance(span_metadata.get("raw_spans", None), list)
                ):
                    try:
                        mismatch_thr = float(getattr(self, "boundary_rewrite_mismatch_threshold", 0.2))
                    except Exception:
                        mismatch_thr = 0.2
                    try:
                        min_len = int(getattr(self, "boundary_rewrite_min_span_len", 2))
                    except Exception:
                        min_len = 2
                    raw_spans = span_metadata.get("raw_spans", [])
                    bsz = int(boundaries.size(0))
                    seqlen = int(boundaries.size(1))
                    rewrite_mask = torch.zeros_like(pos_mask, dtype=torch.bool)
                    for b in range(min(bsz, len(raw_spans))):
                        item_spans = raw_spans[b]
                        if not isinstance(item_spans, list):
                            continue
                        for sp in item_spans:
                            if not isinstance(sp, dict):
                                continue
                            idxs = sp.get("indices", None)
                            if idxs is None:
                                idxs = sp.get("token_indices", None)
                            if not isinstance(idxs, list) or len(idxs) == 0:
                                continue
                            # Default excludes singles by requiring len>=2 (min_len default=2)
                            if len(idxs) < int(min_len):
                                continue
                            valid = [int(i) for i in idxs if 0 <= int(i) < seqlen]
                            if len(valid) < int(min_len):
                                continue
                            idx_t = torch.tensor(valid, device=boundaries.device, dtype=torch.long)
                            try:
                                mism = (global_argmax_ids[b, idx_t] != input_ids[b, idx_t]).float().mean().item()
                            except Exception:
                                continue
                            if float(mism) <= float(mismatch_thr):
                                continue
                            start_pos = sp.get("start", None)
                            if start_pos is None:
                                start_pos = min(valid)
                            start_pos = int(start_pos)
                            if 0 <= start_pos < seqlen:
                                rewrite_mask[b, start_pos] = True
                    pos_mask = pos_mask & rewrite_mask
                    # Ensure singles are not treated as positives unless explicitly enabled.
                    if not include_singles:
                        pos_mask = pos_mask & (boundaries != 3)
                boundary_targets = pos_mask.long()  # [B, L]
                # Mask to valid positions
                mask = attention_mask.to(torch.bool) if attention_mask is not None else torch.ones_like(boundary_targets, dtype=torch.bool, device=boundary_targets.device)
                # Boundary logits: optionally use multi-layer features (mid+last) to avoid relying only on last-layer next-token features.
                hs_list = None
                try:
                    hs_list = list(outputs.hidden_states) if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None else None
                except Exception:
                    hs_list = None
                logits = self.compute_boundary_logits(last_hidden=last_hidden_for_heads, hidden_states=hs_list)  # [B, L, 2]
                
                # Compute class weights if needed (handle class imbalance)
                class_weight = None
                if self.boundary_class_weight is not None:
                    class_weight = self.boundary_class_weight.to(logits.device)
                elif self.boundary_focal_gamma == 0.0:
                    # Auto-compute class weights based on class frequency (inverse frequency weighting)
                    # This helps when most tokens are non-boundaries
                    with torch.no_grad():
                        masked_targets = boundary_targets[mask].view(-1)
                        if masked_targets.numel() > 0:
                            num_non_boundary = (masked_targets == 0).sum().float()
                            num_boundary = (masked_targets == 1).sum().float()
                            total = num_non_boundary + num_boundary
                            if num_boundary > 0 and num_non_boundary > 0:
                                # Inverse frequency weighting: weight = total / (num_classes * class_count)
                                weight_non_boundary = total / (2.0 * num_non_boundary)
                                weight_boundary = total / (2.0 * num_boundary)
                                class_weight = torch.tensor([weight_non_boundary, weight_boundary], device=logits.device, dtype=logits.dtype)
                
                # Compute loss with optional focal loss
                if self.boundary_focal_gamma > 0.0:
                    # Focal loss: focuses on hard examples
                    logits_flat = logits[mask].view(-1, 2)
                    targets_flat = boundary_targets[mask].view(-1)
                    ce_loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction='none', weight=class_weight)
                    probs_flat = F.softmax(logits_flat, dim=-1)
                    p_t = probs_flat.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
                    focal_weight = (1 - p_t) ** self.boundary_focal_gamma
                    ce_loss = (focal_weight * ce_loss_flat).mean()
                else:
                    # Standard cross-entropy with optional class weighting
                    ce_loss = F.cross_entropy(
                        logits[mask].view(-1, 2),
                        boundary_targets[mask].view(-1),
                        weight=class_weight,
                    )
                # Metrics for monitoring
                try:
                    with torch.no_grad():
                        probs = torch.softmax(logits, dim=-1)  # [B, L, 2]
                        preds = torch.argmax(probs, dim=-1)    # [B, L]
                        m = mask
                        # Overall accuracy on valid tokens
                        acc = (preds[m] == boundary_targets[m]).float().mean()
                        # Positive label and prediction rates
                        pos_rate = boundary_targets[m].float().mean()
                        pred_pos_rate = (preds[m] == 1).float().mean()
                        # Recall on starts vs singles
                        start_mask = m & (span_metadata['boundaries'].to(m.device) == 1)
                        single_mask = m & (span_metadata['boundaries'].to(m.device) == 3)
                        if torch.any(start_mask):
                            start_recall = (preds[start_mask] == 1).float().mean()
                            outputs.boundary_start_recall = start_recall
                        if torch.any(single_mask):
                            single_recall = (preds[single_mask] == 1).float().mean()
                            outputs.boundary_single_recall = single_recall
                        # Mean prob of positive class (calibration insight)
                        prob_mean = probs[..., 1][m].mean()
                        outputs.boundary_acc = acc
                        outputs.boundary_pos_rate = pos_rate
                        outputs.boundary_pred_pos_rate = pred_pos_rate
                        outputs.boundary_prob_mean = prob_mean
                except Exception:
                    pass
                if not getattr(self, "probe_only", False):
                    if total_loss is None:
                        total_loss = self.boundary_loss_weight * ce_loss
                    else:
                        total_loss = total_loss + self.boundary_loss_weight * ce_loss
                outputs.boundary_loss = ce_loss

            # Latent regression only at starts/single-token spans
            if isinstance(last_hidden_for_heads, torch.Tensor) and 'raw_spans' in span_metadata:
                bsz, seqlen, _ = last_hidden_for_heads.shape
                latent_preds = []
                latent_targets = []
                for b in range(bsz):
                    raw_list = span_metadata.get('raw_spans', [])
                    if not raw_list or b >= len(raw_list):
                        continue
                    item_spans = raw_list[b]
                    for sp in item_spans:
                        if not isinstance(sp, dict):
                            continue
                        idxs = sp.get('token_indices', [])
                        if isinstance(idxs, list):
                            idxs = np.array(idxs, dtype=np.int64)
                        if not isinstance(idxs, np.ndarray) or len(idxs) == 0:
                            continue
                        start = int(np.min(idxs))
                        if start < 0 or start >= seqlen:
                            continue
                        # Target latent = mean of node encoder embeddings on span indices
                        idxs_t = torch.tensor(idxs, device=node_inputs_embeds.device, dtype=torch.long)
                        target_latent = node_inputs_embeds[b, idxs_t, :].mean(dim=0)  # [H]
                        # Predicted latent from global hidden at start position
                        pred_latent = self.latent_from_global(last_hidden_for_heads[b:b+1, start, :]).squeeze(0)  # [H]
                        latent_targets.append(target_latent)
                        latent_preds.append(pred_latent)
                if len(latent_preds) > 0:
                    pred = torch.stack(latent_preds, dim=0)
                    targ = torch.stack(latent_targets, dim=0)
                    mse = F.mse_loss(pred, targ)
                    if not getattr(self, "probe_only", False):
                        if total_loss is None:
                            total_loss = self.latent_mse_weight * mse
                        else:
                            total_loss = total_loss + self.latent_mse_weight * mse
                    outputs.latent_mse = mse

        # Add local node reconstruction loss when labels/spans present (training OR validation)
        if labels is not None and span_metadata is not None and 'raw_spans' in span_metadata:
            node_losses = self._compute_local_node_recon_loss(
                input_ids=input_ids,
                inputs_embeds=node_inputs_embeds,
                span_metadata=span_metadata,
                last_hidden_for_heads=last_hidden_for_heads,
                attention_mask=attention_mask,
                global_argmax_ids=global_argmax_ids,
            )
            if node_losses is not None:
                node_recon_loss, aux = node_losses
                if not getattr(self, "probe_only", False):
                    if total_loss is None:
                        total_loss = self.node_recon_loss_weight * node_recon_loss
                    else:
                        total_loss = total_loss + self.node_recon_loss_weight * node_recon_loss
                outputs.node_recon_loss = node_recon_loss
                if aux is not None:
                    if 'node_type_loss' in aux:
                        outputs.node_type_loss = aux['node_type_loss']
                        if not getattr(self, "probe_only", False):
                            total_loss = total_loss + aux['node_type_loss']
                    if 'node_len_loss' in aux:
                        outputs.node_len_loss = aux['node_len_loss']
                        if not getattr(self, "probe_only", False):
                            total_loss = total_loss + aux['node_len_loss']
                    # KL teacher->student
                    if 'kl_loss' in aux:
                        outputs.kl_loss = aux['kl_loss']
                        if not getattr(self, "probe_only", False):
                            total_loss = total_loss + self.kl_weight * aux['kl_loss']
                    # InfoNCE
                    if 'infonce_loss' in aux:
                        outputs.infonce_loss = aux['infonce_loss']
                        if not getattr(self, "probe_only", False):
                            total_loss = total_loss + self.infonce_weight * aux['infonce_loss']
                    # Probe metrics from aux
                    if 'type_probe_encoder_loss' in aux:
                        outputs.type_probe_encoder_loss = aux['type_probe_encoder_loss']
                    if 'type_probe_encoder_acc' in aux:
                        outputs.type_probe_encoder_acc = aux['type_probe_encoder_acc']
                    if 'type_probe_decoder_loss' in aux:
                        outputs.type_probe_decoder_loss = aux['type_probe_decoder_loss']
                    if 'type_probe_decoder_acc' in aux:
                        outputs.type_probe_decoder_acc = aux['type_probe_decoder_acc']
                    # Scheduled sampling stats / monitoring
                    if 'span_ss_model_frac' in aux:
                        outputs.span_ss_model_frac = aux['span_ss_model_frac']
                    if 'teacher_ce' in aux:
                        outputs.teacher_ce = aux['teacher_ce']
                    # If probe-only, replace/compose total loss from probes
                    if getattr(self, "probe_only", False):
                        probe_total = None
                        if 'type_probe_encoder_loss' in aux:
                            probe_total = aux['type_probe_encoder_loss'] if probe_total is None else probe_total + aux['type_probe_encoder_loss']
                        if 'type_probe_decoder_loss' in aux:
                            probe_total = aux['type_probe_decoder_loss'] if probe_total is None else probe_total + aux['type_probe_decoder_loss']
                        # Fallback to zero if no probe loss available
                        if probe_total is None:
                            probe_total = torch.zeros((), device=node_inputs_embeds.device, dtype=node_inputs_embeds.dtype)
                        total_loss = probe_total
        outputs.loss = total_loss
        return outputs

    def _segment_non_overlapping(self, raw_spans: List[Dict], seq_len: int, max_nodes: Optional[int] = None) -> List[Dict]:
        """
        Build non-overlapping spans from possibly overlapping spans.
        Strategy: 
          1. Separate multi-token and single-token spans
          2. Prioritize multi-token spans (sorted by length desc, then start)
          3. Fill remaining slots with single-token spans up to max_nodes
        This ensures we train on meaningful multi-token spans, not just trivial single tokens.
        
        Args:
            raw_spans: List of span dictionaries with 'token_indices' and 'span_type_id'
            seq_len: Maximum sequence length
            max_nodes: Maximum number of nodes to return (if None, return all)
        """
        spans = []
        for sp in raw_spans:
            if not isinstance(sp, Dict):
                continue
            token_indices = sp.get('token_indices', [])
            if isinstance(token_indices, list):
                token_indices = np.array(token_indices, dtype=np.int64)
            if not isinstance(token_indices, np.ndarray):
                continue
            if len(token_indices) == 0:
                continue
            token_indices = np.unique(token_indices[(token_indices >= 0) & (token_indices < seq_len)])
            if len(token_indices) == 0:
                continue
            start = int(token_indices.min())
            length = int(len(token_indices))
            spans.append({
                'start': start,
                'length': length,
                'indices': token_indices,
                'span_type_id': int(sp.get('span_type_id', 0)),
            })
        
        # Separate multi-token and single-token spans
        multi_token = [sp for sp in spans if sp['length'] > 1]
        single_token = [sp for sp in spans if sp['length'] == 1]
        
        # Sort multi-token by length (descending), then by start position
        # This prioritizes longer, more meaningful spans
        multi_token.sort(key=lambda x: (-x['length'], x['start']))
        
        # Sort single-token by start position
        single_token.sort(key=lambda x: x['start'])
        
        # Greedily select non-overlapping spans, prioritizing multi-token
        used = np.zeros(seq_len, dtype=bool)
        selected_multi = []
        selected_single = []
        
        # First, select multi-token spans
        for sp in multi_token:
            idxs = sp['indices']
            if not used[idxs].any():
                used[idxs] = True
                selected_multi.append(sp)
        
        # Then, select single-token spans
        for sp in single_token:
            idxs = sp['indices']
            if not used[idxs].any():
                used[idxs] = True
                selected_single.append(sp)
        
        # Apply max_nodes limit: take multi-token first (up to max_nodes), then fill with single-token
        if max_nodes is not None:
            final = []
            # Take multi-token spans up to max_nodes
            final.extend(selected_multi[:max_nodes])
            remaining_slots = max_nodes - len(final)
            if remaining_slots > 0:
                # Fill remaining slots with single-token spans
                final.extend(selected_single[:remaining_slots])
        else:
            final = selected_multi + selected_single
        
        # Sort by start position for consistent ordering in the sequence
        final.sort(key=lambda x: x['start'])
        
        return final

    def _compute_local_node_recon_loss(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        span_metadata: Dict,
        last_hidden_for_heads: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_argmax_ids: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]]:
        """
        Teacher-forced next-token prediction (generation objective) using the local decoder.
        Cross-attends to span/global memories. The loss is computed on the student path
        using the predicted span latent derived from the global hidden state.
        Returns (recon_loss, aux_losses_dict) or None if no nodes.
        """
        device = inputs_embeds.device
        batch_size, seq_len = input_ids.shape

        node_input_seqs = []
        node_target_seqs = []
        node_type_ids: List[int] = []

        total_nodes_kept = 0

        # Track batch/position indices for gradient-enabled indexing later
        node_batch_indices: List[int] = []
        node_start_indices: List[int] = []
        node_span_indices: List[List[int]] = []  # Token indices for each span
        
        for b in range(batch_size):
            raw_list = span_metadata.get('raw_spans', [])
            if not raw_list or b >= len(raw_list):
                continue
            item_spans = raw_list[b]
            max_nodes = getattr(self, "max_nodes_per_sample", 16)
            sel = self._segment_non_overlapping(item_spans, seq_len, max_nodes=max_nodes)
            if not sel:
                continue
            # Training-time span filter for rewrite-parallel / local recon:
            # skip very short spans (e.g. single-token spans) to better match inference,
            # reduce noise, and avoid spending node budget on trivial spans.
            min_span_len = int(getattr(self, "rewrite_min_span_len", 2))
            for sp in sel:
                idxs = sp['indices']
                if len(idxs) == 0:
                    continue
                # Truncate span indices to match reconstruction cap (bounds both compute and memory)
                idxs = idxs[: min(len(idxs), int(self.max_node_length))]
                token_seq = input_ids[b, torch.tensor(idxs, device=device)].detach()
                L = int(token_seq.shape[0])
                if L <= 0:
                    continue
                if L < min_span_len:
                    continue
                # store start index if available
                start_idx = int(sp.get('start', int(torch.tensor(idxs, device=device).min().item())))
                
                # Store indices for batched gradient-enabled operations later
                # This avoids calling latent_combine N times in the loop
                
                # truncate for reconstruction
                Lc = min(L, self.max_node_length)
                target = token_seq[:Lc]  # [Lc]
                # teacher-forced inputs: BOS + tokens[:-1]
                if Lc == 1:
                    inp = torch.tensor([self.node_bos_id], device=device, dtype=torch.long)
                else:
                    inp = torch.cat([
                        torch.tensor([self.node_bos_id], device=device, dtype=torch.long),
                        target[:-1]
                    ], dim=0)

                node_input_seqs.append(inp)
                node_target_seqs.append(target)
                node_batch_indices.append(b)
                node_start_indices.append(start_idx)
                node_span_indices.append(idxs)
                # Span type id if present (defaults to 0)
                node_type_ids.append(int(sp.get('span_type_id', 0)))
                total_nodes_kept += 1

        if total_nodes_kept == 0:
            return None

        # Pad batch
        maxL = max(seq.shape[0] for seq in node_input_seqs)
        inp_batch = torch.full(
            (total_nodes_kept, maxL),
            fill_value=self.config.pad_token_id if self.config.pad_token_id is not None else 0,
            dtype=torch.long,
            device=device
        )
        tgt_batch = torch.full((total_nodes_kept, maxL), fill_value=-100, dtype=torch.long, device=device)
        mask_batch = torch.zeros((total_nodes_kept, maxL), dtype=torch.bool, device=device)
        for i in range(total_nodes_kept):
            L = node_input_seqs[i].shape[0]
            inp_batch[i, :L] = node_input_seqs[i]
            tgtL = node_target_seqs[i].shape[0]
            tgt_batch[i, :tgtL] = node_target_seqs[i]
            mask_batch[i, :tgtL] = True
        
        # === Scheduled sampling for local-encoder span tokens (train/infer mismatch) ===
        # Build span_memory and encoder_latents_batch from node_token_encoder(span_tokens),
        # where span_tokens are either gold tokens or teacher-forced global argmax tokens.
        pad_id = int(self.config.pad_token_id) if self.config.pad_token_id is not None else 0
        ss_mode = str(getattr(self, "span_ss_mode", "off"))
        p_gold = float(getattr(self, "span_ss_p_gold", 1.0))
        use_model = torch.zeros((total_nodes_kept,), device=device, dtype=torch.bool)
        if ss_mode != "off" and global_argmax_ids is not None:
            # per-span scheduled sampling
            use_model = torch.rand((total_nodes_kept,), device=device) > float(p_gold)

        # Build padded span token ids for local encoder
        span_len_list: List[int] = [len(idxs) for idxs in node_span_indices]
        max_span_mem = max(span_len_list) if len(span_len_list) > 0 else 0
        if max_span_mem <= 0:
            return None

        span_tokens_padded = torch.full(
            (total_nodes_kept, max_span_mem),
            fill_value=pad_id,
            dtype=torch.long,
            device=device,
        )
        span_kpm = torch.ones((total_nodes_kept, max_span_mem), device=device, dtype=torch.bool)  # True=mask/ignore

        model_used_count = 0
        for i in range(total_nodes_kept):
            b = int(node_batch_indices[i])
            idxs = node_span_indices[i]
            sl = int(len(idxs))
            if sl <= 0:
                continue
            idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
            if bool(use_model[i]):
                toks = global_argmax_ids[b, idxs_t]
                model_used_count += 1
            else:
                toks = input_ids[b, idxs_t]
            span_tokens_padded[i, :sl] = toks
            span_kpm[i, :sl] = False

        # Local encoder outputs for span memory: [N, S, H]
        span_mem_batch = self.node_token_encoder(span_tokens_padded, None)

        # Encoder latents: masked mean over non-pad positions: [N, H]
        valid = (~span_kpm).unsqueeze(-1).to(span_mem_batch.dtype)  # [N,S,1]
        denom = valid.sum(dim=1).clamp(min=1.0)  # [N,1]
        encoder_latents_batch = (span_mem_batch * valid).sum(dim=1) / denom  # [N,H]
        
        # Build global hiddens with gradient flow through last_hidden_for_heads
        global_hiddens_list = []
        for i in range(total_nodes_kept):
            b = node_batch_indices[i]
            start_idx = node_start_indices[i]
            if isinstance(last_hidden_for_heads, torch.Tensor) and 0 <= start_idx < seq_len:
                global_hiddens_list.append(last_hidden_for_heads[b, start_idx, :])
            else:
                global_hiddens_list.append(torch.zeros(self.hidden_size, device=device, dtype=inputs_embeds.dtype))
        global_hiddens_batch = torch.stack(global_hiddens_list, dim=0)  # [N, H]
        
        # === Combine encoder + global latent (Option 2) - BATCHED ===
        if hasattr(self, 'latent_combine'):
            combined_input = torch.cat([encoder_latents_batch, global_hiddens_batch], dim=-1)  # [N, 2H]
            latents_batch = self.latent_combine(combined_input)  # [N, H]
        else:
            latents_batch = encoder_latents_batch
        
        # Predicted latents for student path (via latent_from_global) - BATCHED
        pred_latents_batch = self.latent_from_global(global_hiddens_batch.detach())  # [N, H]

        # Local decoder forward (cross-attn to span/global memories)
        tok_emb = self.local_token_embed(inp_batch)  # [N, L, H]
        teacher_span_latent = self.latent_proj(latents_batch).unsqueeze(1)  # [N, 1, H]
        # span_mem_batch / span_kpm already built above from scheduled-sampled span tokens

        # === Structural memory fix ===
        # Do NOT replicate global memory per node as [N_nodes, L, H].
        # Instead, group nodes by sample b and pass global_memory as [1, L, H] so
        # SDPA global cross-attn can broadcast K/V without materializing [N_b, L, H].
        attn_mask_bool = (
            attention_mask.to(torch.bool)
            if attention_mask is not None
            else torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
        )
        by_b: List[List[int]] = [[] for _ in range(batch_size)]
        for i, b in enumerate(node_batch_indices):
            if 0 <= int(b) < batch_size:
                by_b[int(b)].append(i)

        # Assemble teacher outputs in the original node order to keep downstream probes unchanged
        dec_out_teacher = torch.empty_like(tok_emb)  # [N, L, H]

        teacher_num = torch.zeros((), device=device, dtype=torch.float32)
        teacher_den = torch.zeros((), device=device, dtype=torch.float32)

        for b in range(batch_size):
            idx_list = by_b[b]
            if not idx_list:
                continue
            idx = torch.tensor(idx_list, device=device, dtype=torch.long)

            tok_emb_b = tok_emb.index_select(0, idx)
            teacher_latent_b = teacher_span_latent.index_select(0, idx)
            span_mem_b = span_mem_batch.index_select(0, idx) if span_mem_batch is not None else None
            span_kpm_b = span_kpm.index_select(0, idx) if span_kpm is not None else None

            global_mem_b = last_hidden_for_heads[b:b+1, :, :] if isinstance(last_hidden_for_heads, torch.Tensor) else None  # [1,L,H]
            global_kpm_b = (~attn_mask_bool[b:b+1, :]) if global_mem_b is not None else None  # [1,L]

            # === Span-memory dropout (train analogue of inference disable_local_encoder_only) ===
            # With probability p_drop, do not provide span_memory/keys to the local decoder, forcing it to rely on:
            #   - span_latent (teacher_latent_b)
            #   - global_memory cross-attention
            # This reduces train–infer mismatch and improves robustness when span_memory is absent at inference.
            try:
                p_drop = float(getattr(self, "span_mem_drop_p", 0.0))
            except Exception:
                p_drop = 0.0

            if p_drop > 0.0 and span_mem_b is not None and span_kpm_b is not None:
                # Per-node dropout: split nodes into keep vs drop groups to preserve shapes.
                drop_mask = (torch.rand((tok_emb_b.size(0),), device=device) < float(p_drop))
                keep_mask = ~drop_mask
                if torch.any(drop_mask) and torch.any(keep_mask):
                    idx_keep = torch.nonzero(keep_mask, as_tuple=False).view(-1)
                    idx_drop = torch.nonzero(drop_mask, as_tuple=False).view(-1)
                    out_keep = self.local_decoder(
                        tok_emb_b.index_select(0, idx_keep),
                        teacher_latent_b.index_select(0, idx_keep),
                        span_memory=span_mem_b.index_select(0, idx_keep),
                        span_key_padding_mask=span_kpm_b.index_select(0, idx_keep),
                        global_memory=global_mem_b,
                        global_key_padding_mask=global_kpm_b,
                    )
                    out_drop = self.local_decoder(
                        tok_emb_b.index_select(0, idx_drop),
                        teacher_latent_b.index_select(0, idx_drop),
                        span_memory=None,
                        span_key_padding_mask=None,
                        global_memory=global_mem_b,
                        global_key_padding_mask=global_kpm_b,
                    )
                    dec_out_teacher_b = torch.empty_like(tok_emb_b)
                    dec_out_teacher_b.index_copy_(0, idx_keep, out_keep)
                    dec_out_teacher_b.index_copy_(0, idx_drop, out_drop)
                elif torch.any(drop_mask):
                    # All dropped
                    dec_out_teacher_b = self.local_decoder(
                        tok_emb_b,
                        teacher_latent_b,
                        span_memory=None,
                        span_key_padding_mask=None,
                        global_memory=global_mem_b,
                        global_key_padding_mask=global_kpm_b,
                    )
                else:
                    # None dropped
                    dec_out_teacher_b = self.local_decoder(
                        tok_emb_b,
                        teacher_latent_b,
                        span_memory=span_mem_b,
                        span_key_padding_mask=span_kpm_b,
                        global_memory=global_mem_b,
                        global_key_padding_mask=global_kpm_b,
                    )
            else:
                dec_out_teacher_b = self.local_decoder(
                    tok_emb_b,
                    teacher_latent_b,
                    span_memory=span_mem_b,
                    span_key_padding_mask=span_kpm_b,
                    global_memory=global_mem_b,
                    global_key_padding_mask=global_kpm_b,
                )  # [N_b, L, H]

            # Residual (per-node)
            if hasattr(self, 'global_residual_gate') and hasattr(self, 'global_residual_scale'):
                global_hidden_b = global_hiddens_batch.index_select(0, idx)
                global_hidden_expanded = global_hidden_b.unsqueeze(1).expand_as(dec_out_teacher_b)
                gate_input = torch.cat([dec_out_teacher_b, global_hidden_expanded], dim=-1)
                gate = self.global_residual_gate(gate_input)
                dec_out_teacher_b = (1 - gate) * dec_out_teacher_b + gate * self.global_residual_scale * global_hidden_expanded

            # Save into full tensor for downstream probes
            dec_out_teacher.index_copy_(0, idx, dec_out_teacher_b)
        
        def _linear_cross_entropy_chunked(
            hidden: torch.Tensor,
            out_proj: nn.Linear,
            targets: torch.Tensor,
            ignore_index: int = -100,
            chunk_size: int = 4096,
        ) -> torch.Tensor:
            """
            Memory-efficient CE for very large vocab projections.
            Computes CE without materializing logits of shape [M, V].
            - hidden: [..., H]
            - out_proj: Linear(H -> V)
            - targets: [...] (same leading shape), values in [0, V) or ignore_index
            Returns: scalar loss (float32)
            """
            x = hidden.reshape(-1, hidden.size(-1))
            t = targets.reshape(-1)
            valid = (t != ignore_index)
            if valid.sum().item() == 0:
                return torch.zeros((), device=x.device, dtype=torch.float32)
            x = x[valid]                      # [M, H]
            t = t[valid].to(torch.long)       # [M]

            weight = out_proj.weight          # [V, H]
            bias = out_proj.bias              # [V] or None
            vocab = int(weight.size(0))

            # logsumexp over vocab in chunks, accumulated in float32 for stability
            lse_total: Optional[torch.Tensor] = None
            for start in range(0, vocab, int(chunk_size)):
                end = min(vocab, start + int(chunk_size))
                w = weight[start:end, :]  # [C, H]
                # logits: [M, C]
                logits = x @ w.t()
                if bias is not None:
                    logits = logits + bias[start:end]
                lse_chunk = torch.logsumexp(logits.float(), dim=-1)  # [M]
                lse_total = lse_chunk if lse_total is None else torch.logaddexp(lse_total, lse_chunk)

            # target logit via indexed weight rows (no [M, V] allocation)
            w_t = weight.index_select(0, t)  # [M, H]
            target_logit = (x * w_t).sum(dim=-1)  # [M]
            if bias is not None:
                target_logit = target_logit + bias.index_select(0, t)

            loss_vec = lse_total - target_logit.float()
            return loss_vec.mean()

        def _linear_argmax_chunked(
            hidden: torch.Tensor,
            out_proj: nn.Linear,
            chunk_size: int = 4096,
        ) -> torch.Tensor:
            """
            Memory-efficient argmax over vocab for out_proj(hidden) without materializing [M, V] logits.
            - hidden: [..., H]
            Returns: argmax token ids with shape [...]
            """
            x = hidden.reshape(-1, hidden.size(-1))  # [M, H]
            weight = out_proj.weight                 # [V, H]
            bias = out_proj.bias                     # [V] or None
            vocab = int(weight.size(0))
            best_val: Optional[torch.Tensor] = None
            best_idx: Optional[torch.Tensor] = None
            for start in range(0, vocab, int(chunk_size)):
                end = min(vocab, start + int(chunk_size))
                w = weight[start:end, :]  # [C, H]
                logits = x @ w.t()
                if bias is not None:
                    logits = logits + bias[start:end]
                chunk_val, chunk_idx = torch.max(logits, dim=-1)  # [M]
                chunk_idx = chunk_idx.to(torch.long) + int(start)
                if best_val is None:
                    best_val = chunk_val
                    best_idx = chunk_idx
                else:
                    better = chunk_val > best_val
                    best_val = torch.where(better, chunk_val, best_val)
                    best_idx = torch.where(better, chunk_idx, best_idx)
            if best_idx is None:
                # Should not happen, but keep safe behavior.
                return torch.zeros(hidden.shape[:-1], device=hidden.device, dtype=torch.long)
            return best_idx.view(hidden.shape[:-1])

        # Targets with ignore_index outside mask
        targets_full = torch.where(
            mask_batch,
            tgt_batch,
            torch.full_like(tgt_batch, -100),
        )
        # Teacher CE for monitoring only (no giant logits allocation), aggregated across groups
        for b in range(batch_size):
            idx_list = by_b[b]
            if not idx_list:
                continue
            idx = torch.tensor(idx_list, device=device, dtype=torch.long)
            dec_t = dec_out_teacher.index_select(0, idx)
            tgt_t = targets_full.index_select(0, idx)
            valid = (tgt_t.reshape(-1) != -100)
            denom = valid.sum().float()
            if denom.item() <= 0:
                continue
            ce = _linear_cross_entropy_chunked(
                dec_t,
                self.local_out_proj,
                tgt_t,
                ignore_index=-100,
                chunk_size=int(getattr(self, "ce_chunk_size", 4096)),
            )
            teacher_num = teacher_num + ce.float() * denom
            teacher_den = teacher_den + denom
        teacher_ce = teacher_num / torch.clamp(teacher_den, min=1.0)

        aux_losses: Dict[str, torch.Tensor] = {}
        # Student path: predicted latent via last_hidden_for_heads (if available)
        # Always compute student CE for next-token prediction training
        kl_weight = float(getattr(self, 'kl_weight', 0.0))
        infonce_weight = float(getattr(self, 'infonce_weight', 0.0))
        if isinstance(last_hidden_for_heads, torch.Tensor):
            # Use predicted span latents derived from global hidden state at span starts
            recon_num = torch.zeros((), device=device, dtype=torch.float32)
            recon_den = torch.zeros((), device=device, dtype=torch.float32)
            # Only materialize full student outputs if KL is enabled (it needs full logits).
            dec_out_student_full = torch.empty_like(tok_emb) if float(getattr(self, "kl_weight", 0.0)) > 0.0 else None

            # Local-decoder input mode: teacher-forced vs self-conditioning (approx free-run).
            # self_condition: run a teacher-forced pass to get predicted tokens, then re-run using BOS+pred[:-1] as inputs.
            try:
                train_mode = str(getattr(self, "local_decoder_train_mode", "teacher")).lower()
            except Exception:
                train_mode = "teacher"
            try:
                p_self_cond = float(getattr(self, "local_decoder_self_condition_p", 0.0))
            except Exception:
                p_self_cond = 0.0
            use_self_cond = bool(train_mode == "self_condition") and (float(p_self_cond) > 0.0) and (torch.rand((), device=device).item() < float(p_self_cond))
            aux_losses["local_decoder_self_condition_p"] = torch.tensor(float(p_self_cond), device=device, dtype=torch.float32)
            aux_losses["local_decoder_used_self_condition"] = torch.tensor(1.0 if use_self_cond else 0.0, device=device, dtype=torch.float32)

            for b in range(batch_size):
                idx_list = by_b[b]
                if not idx_list:
                    continue
                idx = torch.tensor(idx_list, device=device, dtype=torch.long)

                tok_emb_b = tok_emb.index_select(0, idx)
                pred_lat_b = pred_latents_batch.index_select(0, idx)
                student_latent_b = self.latent_proj(pred_lat_b).unsqueeze(1)  # [N_b,1,H]
                span_mem_b = span_mem_batch.index_select(0, idx) if span_mem_batch is not None else None
                span_kpm_b = span_kpm.index_select(0, idx) if span_kpm is not None else None

                global_mem_b = last_hidden_for_heads[b:b+1, :, :]  # [1,L,H]
                global_kpm_b = (~attn_mask_bool[b:b+1, :])          # [1,L]

                dec_out_student_b = self.local_decoder(
                    tok_emb_b,
                    student_latent_b,
                    span_memory=span_mem_b,
                    span_key_padding_mask=span_kpm_b,
                    global_memory=global_mem_b,
                    global_key_padding_mask=global_kpm_b,
                )  # [N_b,L,H]

                # Residual (per-node)
                if hasattr(self, 'global_residual_gate') and hasattr(self, 'global_residual_scale'):
                    global_hidden_b = global_hiddens_batch.index_select(0, idx)
                    global_hidden_expanded = global_hidden_b.unsqueeze(1).expand_as(dec_out_student_b)
                    gate_input = torch.cat([dec_out_student_b, global_hidden_expanded], dim=-1)
                    gate = self.global_residual_gate(gate_input)
                    dec_out_student_b = (1 - gate) * dec_out_student_b + gate * self.global_residual_scale * global_hidden_expanded

                # Optional self-conditioning second pass (BOS + predicted_tokens[:-1]).
                # This approximates free-run generation while keeping training differentiable and efficient.
                if use_self_cond:
                    try:
                        # Compute predicted token ids from the first pass without materializing full logits.
                        pred_ids = _linear_argmax_chunked(
                            dec_out_student_b.detach(),
                            self.local_out_proj,
                            chunk_size=int(getattr(self, "ce_chunk_size", 4096)),
                        )  # [N_b, L]
                        # Build new decoder input ids: BOS + pred_ids[:-1] (masked by target availability).
                        tgt_b = targets_full.index_select(0, idx)  # [N_b, L]
                        mb = mask_batch.index_select(0, idx)       # [N_b, L]
                        inp2 = torch.full_like(tgt_b, fill_value=pad_id, dtype=torch.long, device=device)
                        inp2[:, 0] = int(getattr(self, "node_bos_id", 0))
                        if inp2.size(1) > 1:
                            # Only shift where we have a valid target at the previous position.
                            prev_valid = mb[:, :-1]
                            shifted = pred_ids[:, :-1].to(torch.long)
                            inp2[:, 1:] = torch.where(prev_valid, shifted, torch.full_like(shifted, pad_id))
                        tok_emb_b2 = self.local_token_embed(inp2)
                        dec_out_student_b2 = self.local_decoder(
                            tok_emb_b2,
                            student_latent_b,
                            span_memory=span_mem_b,
                            span_key_padding_mask=span_kpm_b,
                            global_memory=global_mem_b,
                            global_key_padding_mask=global_kpm_b,
                        )
                        if hasattr(self, 'global_residual_gate') and hasattr(self, 'global_residual_scale'):
                            global_hidden_b = global_hiddens_batch.index_select(0, idx)
                            global_hidden_expanded = global_hidden_b.unsqueeze(1).expand_as(dec_out_student_b2)
                            gate_input = torch.cat([dec_out_student_b2, global_hidden_expanded], dim=-1)
                            gate = self.global_residual_gate(gate_input)
                            dec_out_student_b2 = (1 - gate) * dec_out_student_b2 + gate * self.global_residual_scale * global_hidden_expanded
                        dec_out_student_b = dec_out_student_b2
                    except Exception:
                        # If anything goes wrong, keep teacher-forced student pass.
                        pass

                if dec_out_student_full is not None:
                    dec_out_student_full.index_copy_(0, idx, dec_out_student_b)

                tgt_b = targets_full.index_select(0, idx)
                valid = (tgt_b.reshape(-1) != -100)
                denom = valid.sum().float()
                if denom.item() <= 0:
                    continue
                ce = _linear_cross_entropy_chunked(
                    dec_out_student_b,
                    self.local_out_proj,
                    tgt_b,
                    ignore_index=-100,
                    chunk_size=int(getattr(self, "ce_chunk_size", 4096)),
                )
                recon_num = recon_num + ce.float() * denom
                recon_den = recon_den + denom

            recon_loss = recon_num / torch.clamp(recon_den, min=1.0)
            # KL between student and teacher on valid positions
            if kl_weight > 0:
                try:
                    # Warning: KL requires full vocab distributions and can be very memory heavy.
                    if dec_out_student_full is None:
                        raise RuntimeError("KL requested but dec_out_student_full was not materialized")
                    logits_student = self.local_out_proj(dec_out_student_full)  # [N,L,V]
                    logits_teacher = self.local_out_proj(dec_out_teacher).detach()  # [N,L,V]
                    log_p_student = F.log_softmax(logits_student[mask_batch], dim=-1)
                    p_teacher = F.softmax(logits_teacher[mask_batch], dim=-1)
                    kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean')
                    aux_losses['kl_loss'] = kl
                except Exception:
                    pass
            # InfoNCE across span latents (teacher vs student/proxy)
            if infonce_weight > 0:
                try:
                    z_q = F.normalize(latents_batch, dim=-1)  # [N,H]
                    z_k = F.normalize(latents_batch.detach(), dim=-1)  # positives (proxy)
                    sim = torch.matmul(z_q, z_k.t()) / float(getattr(self, 'infonce_tau', 0.07))
                    targets = torch.arange(sim.size(0), device=sim.device)
                    nce = F.cross_entropy(sim, targets)
                    aux_losses['infonce_loss'] = nce
                except Exception:
                    pass
        if self.node_type_head is not None and total_nodes_kept > 0:
            type_logits = self.node_type_head(latents_batch)  # [N, C]
            type_targets = torch.tensor(
                [int(sp.get('span_type_id', 0)) for sp in self._segment_non_overlapping(span_metadata.get('raw_spans', [])[0], seq_len)] if batch_size == 1 else
                [0] * latents_batch.size(0),
                dtype=torch.long, device=device
            )
            if type_targets.numel() == 0 or type_targets.size(0) != latents_batch.size(0):
                # Skip if we cannot reliably align targets
                pass
            else:
                aux_losses['node_type_loss'] = F.cross_entropy(type_logits, type_targets)

        if self.node_len_head is not None:
            # Length bin targets were not retained above; optional extension if needed
            pass

        # Type probes (encoder latent and decoder node representation)
        try:
            if total_nodes_kept > 0 and len(node_type_ids) == total_nodes_kept:
                type_targets_all = torch.tensor(node_type_ids, dtype=torch.long, device=device)
                # Encoder latent probe
                if hasattr(self, "node_type_probe_encoder") and self.node_type_probe_encoder is not None:
                    enc_logits = self.node_type_probe_encoder(latents_batch.detach())  # [N, C]
                    enc_loss = F.cross_entropy(enc_logits, type_targets_all)
                    enc_acc = (enc_logits.argmax(dim=-1) == type_targets_all).float().mean()
                    aux_losses['type_probe_encoder_loss'] = enc_loss
                    aux_losses['type_probe_encoder_acc'] = enc_acc
                # Decoder node representation probe (masked mean over valid steps)
                if hasattr(self, "node_type_probe_decoder") and self.node_type_probe_decoder is not None:
                    valid_counts = mask_batch.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [N,1]
                    masked_sum = (dec_out_teacher * mask_batch.unsqueeze(-1)).sum(dim=1)  # [N,H]
                    dec_repr = masked_sum / valid_counts  # [N,H]
                    dec_logits = self.node_type_probe_decoder(dec_repr.detach())
                    dec_loss = F.cross_entropy(dec_logits, type_targets_all)
                    dec_acc = (dec_logits.argmax(dim=-1) == type_targets_all).float().mean()
                    aux_losses['type_probe_decoder_loss'] = dec_loss
                    aux_losses['type_probe_decoder_acc'] = dec_acc
        except Exception:
            # Do not fail training if probes encounter shape/label issues
            pass

        # Scheduled sampling stats (for logging)
        try:
            aux_losses['span_ss_model_frac'] = torch.tensor(
                float(model_used_count) / float(max(1, total_nodes_kept)),
                device=device,
                dtype=torch.float32,
            )
        except Exception:
            pass

        # Expose teacher CE for monitoring
        aux_losses['teacher_ce'] = teacher_ce
        return recon_loss, (aux_losses if len(aux_losses) > 0 else None)

    @torch.no_grad()
    def generate_node_tokens(
        self, 
        span_latent: torch.Tensor, 
        span_memory: Optional[torch.Tensor] = None,
        span_key_padding_mask: Optional[torch.Tensor] = None,
        global_hidden: Optional[torch.Tensor] = None,
        global_memory: Optional[torch.Tensor] = None,
        global_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: int = 64, 
        prefix_ids: Optional[torch.Tensor] = None,
        num_new_tokens: Optional[int] = None,
        bos_id: Optional[int] = None, 
        eos_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Greedy decode one node from a span latent using the local decoder.
        
        Args:
            span_latent: [H] - the combined (encoder+global) span latent
        span_memory: [S, H] or [1, S, H] - optional per-token memory from local encoder for current node
        span_key_padding_mask: [S] or [1, S] - True to mask/ignore in span memory
            global_hidden: [H] - optional last-token global hidden for residual connection
            global_memory: [G, H] or [1, G, H] - optional full-sequence global memory for cross-attention
            global_key_padding_mask: [G] or [1, G] - True to mask/ignore in global memory
            max_len: maximum tokens to generate (total cap)
            prefix_ids: optional 1D tensor of already generated token ids to condition on
            num_new_tokens: if provided, generate at most this many new tokens beyond prefix
            bos_id: beginning of sequence token id
            eos_id: end of sequence token id
            
        Returns:
            If prefix_ids is None: token ids [<=max_len]
            If prefix_ids provided: only the newly generated token ids (excludes prefix)
        """
        self.eval()
        device = span_latent.device
        bos = bos_id if bos_id is not None else self.node_bos_id
        eos = eos_id if eos_id is not None else getattr(self.config, 'eos_token_id', None)

        # Seed tokens with BOS and optional prefix (prefix should NOT include BOS)
        tokens: List[int] = []
        prefix_list: List[int] = []
        if prefix_ids is not None and torch.numel(prefix_ids) > 0:
            if prefix_ids.dim() > 1:
                prefix_ids = prefix_ids.view(-1)
            prefix_list = [int(t) for t in prefix_ids.tolist()]
            tokens = [bos] + prefix_list
        else:
            tokens = [bos]
            prefix_list = []
        prefix_len = len(prefix_list)

        # Determine how many new tokens to generate this call
        if num_new_tokens is not None:
            steps_to_generate = max(0, int(num_new_tokens))
            steps_to_generate = min(steps_to_generate, max(0, int(max_len) - prefix_len))
        else:
            # Generate up to max_len total minus existing prefix
            steps_to_generate = max(0, int(max_len) - prefix_len)

        cond = self.latent_proj(span_latent).unsqueeze(0).unsqueeze(1)  # [1,1,H]
        
        # Prepare global hidden for residual if available
        has_residual = (
            global_hidden is not None and 
            hasattr(self, 'global_residual_gate') and 
            hasattr(self, 'global_residual_scale')
        )
        # Normalize span/global memory shapes for batch_first operations
        sm = None
        skpm = None
        if span_memory is not None:
            if span_memory.dim() == 2:
                sm = span_memory.unsqueeze(0)  # [1,S,H]
            else:
                sm = span_memory  # [1,S,H]
        if span_key_padding_mask is not None:
            if span_key_padding_mask.dim() == 1:
                skpm = span_key_padding_mask.unsqueeze(0)  # [1,S]
            else:
                skpm = span_key_padding_mask  # [1,S]
        # Normalize global memory shapes for batch_first operations
        gm = None
        gkpm = None
        if global_memory is not None:
            if global_memory.dim() == 2:
                gm = global_memory.unsqueeze(0)  # [1,G,H]
            else:
                gm = global_memory  # [1,G,H]
        if global_key_padding_mask is not None:
            if global_key_padding_mask.dim() == 1:
                gkpm = global_key_padding_mask.unsqueeze(0)  # [1,G]
            else:
                gkpm = global_key_padding_mask  # [1,G]

        for _ in range(steps_to_generate):
            inp = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)  # [1,T]
            x = self.local_token_embed(inp)  # [1,T,H]
            h = self.local_decoder(
                x,
                cond,
                span_memory=sm,
                span_key_padding_mask=skpm,
                global_memory=gm,
                global_key_padding_mask=gkpm,
            )  # [1,T,H]
            
            # Apply residual connection if global_hidden provided
            if has_residual:
                global_expanded = global_hidden.unsqueeze(0).unsqueeze(0).expand_as(h)  # [1,T,H]
                gate_input = torch.cat([h, global_expanded], dim=-1)  # [1,T,2H]
                gate = self.global_residual_gate(gate_input)  # [1,T,1]
                h = (1 - gate) * h + gate * self.global_residual_scale * global_expanded
            
            out = h[:, -1, :]  # [1,H]
            logit = self.local_out_proj(out)  # [1,V]
            next_id = int(torch.argmax(logit, dim=-1).item())
            tokens.append(next_id)
            if eos is not None and next_id == eos:
                break
        # Return newly generated tokens (exclude BOS and provided prefix)
        new_tokens = tokens[1 + prefix_len:]
        return torch.tensor(new_tokens, device=device, dtype=torch.long)


def create_blt_adapter_model(
    model_path: str = "/data/home/zhangsj/AST_decoding",
    local_num_layers: int = 2,
    local_dropout: float = 0.1,
    max_node_length: int = 64,
    num_node_types: Optional[int] = None,
    boundary_class_weight: Optional[torch.Tensor] = None,
    boundary_focal_gamma: float = 0.0,
) -> BLTAdapterModel:
    """
    Load Qwen2.5 Coder 1.5B from model_path, wrap with BLTAdapterModel,
    and copy base embeddings to the local encoder.
    """
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    config = base_model.config
    adapter = BLTAdapterModel(
        config,
        local_num_layers=local_num_layers,
        local_dropout=local_dropout,
        max_node_length=max_node_length,
        num_node_types=num_node_types,
        boundary_class_weight=boundary_class_weight,
        boundary_focal_gamma=boundary_focal_gamma,
    )
    # Persist num_node_types into config.json for robust from_pretrained() resume.
    try:
        adapter.config.num_node_types = int(getattr(adapter, "num_node_types", len(SPAN_TYPE_LIST)))
    except Exception:
        pass

    # Copy transformer weights (except embeddings which we handle separately)
    print("Copying transformer weights into adapter...")
    copied = 0
    for name, param in base_model.named_parameters():
            if name in adapter.state_dict():
                with torch.no_grad():
                    adapter.state_dict()[name].copy_(param)
                    copied += 1
            else:
                # Non-fatal; adapter has additional modules
                pass

    # Copy token embeddings to node token encoder
    adapter.copy_base_embeddings_from(base_model)  # type: ignore[arg-type]

    # Also tie lm_head if shapes match
    if "lm_head.weight" in base_model.state_dict() and "lm_head.weight" in adapter.state_dict():
        with torch.no_grad():
            adapter.state_dict()["lm_head.weight"].copy_(base_model.state_dict()["lm_head.weight"])
            print("Copied lm_head.weight")

    print(f"Copied {copied} parameters from base into adapter.")
    return adapter


# =========================
# Simple Python-only training entrypoint
# =========================
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from ast_parsing_folder.AST_parsing import parse_to_ast, get_ast_leaf_nodes_for_spans  # type: ignore
from torch.optim import AdamW
import datetime
import math
from torch.utils.tensorboard import SummaryWriter
# Optional PEFT imports (not required for inference unless a PEFT adapter is used)
try:
    from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    PeftModel = None  # type: ignore


class PythonASTSpanDataset(Dataset):
    def __init__(self, parquet_file_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Span type vocabulary (fixed mapping provided above)
        self.span_type_to_id: Dict[str, int] = SPAN_TYPE_TO_ID
        self.id_to_span_type: Dict[int, str] = ID_TO_SPAN_TYPE
        self.textual_span_types = TEXTUAL_SPAN_TYPES
        self.num_node_types: int = len(self.span_type_to_id)
        if not os.path.exists(parquet_file_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_file_path}")
        self.df = pd.read_parquet(parquet_file_path)
        # Filter
        content_filter = (self.df['content'].notna()) & (self.df['content'].str.strip() != '')
        if 'error' in self.df.columns:
            self.df = self.df[content_filter & (~self.df['error'].notna())]
        else:
            self.df = self.df[content_filter]
        ast_span_filter = (self.df['AST_span'].notna()) & (self.df['AST_span'].str.len() > 2)
        self.df = self.df[ast_span_filter]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        content = row['content']
        enc = self.tokenizer(
            content,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        # Build span metadata similarly to K_distillation_test1
        try:
            ast_spans = row['AST_span']
            import json
            spans = json.loads(ast_spans) if ast_spans else []
        except Exception:
            spans = []
        span_meta = self._build_span_meta(input_ids, spans)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'span_metadata': span_meta,
            'original_content': content
        }

    def _build_span_meta(self, input_ids: torch.Tensor, ast_spans: List[Dict]) -> Dict[str, torch.Tensor]:
        seq_len = int(input_ids.shape[0])
        span_types = np.zeros(seq_len, dtype=np.int64)  # token-level type id (best-effort; last-wins on overlaps)
        positions = np.zeros(seq_len, dtype=np.int64)
        boundaries = np.zeros(seq_len, dtype=np.int64)
        processed = []
        for sp in ast_spans:
            if not isinstance(sp, dict):
                continue
            token_indices = sp.get('token_indices', [])
            if not token_indices:
                continue
            token_indices = np.array(token_indices, dtype=np.int64)
            valid = token_indices[(token_indices >= 0) & (token_indices < seq_len)]
            if valid.size == 0:
                continue
            span_type_str = str(sp.get('type', 'unknown'))
            span_type_id = int(self.span_type_to_id.get(span_type_str, self.span_type_to_id['unknown']))
            # Textual spans => split into single-token spans
            if span_type_str in self.textual_span_types:
                for t in valid.tolist():
                    span_types[t] = span_type_id
                    positions[t] = 0
                    boundaries[t] = 3  # single
                    processed.append({'token_indices': np.array([t], dtype=np.int64), 'span_type_id': span_type_id})
            else:
                for pos, t in enumerate(valid):
                    span_types[t] = span_type_id
                    positions[t] = min(pos, 31)
                    if valid.size == 1:
                        boundaries[t] = 3
                    elif pos == 0:
                        boundaries[t] = 1
                    elif pos == valid.size - 1:
                        boundaries[t] = 2
                    else:
                        boundaries[t] = 0
                processed.append({'token_indices': valid, 'span_type_id': span_type_id})
        return {
            'span_types': torch.tensor(span_types, dtype=torch.long),
            'positions': torch.tensor(positions, dtype=torch.long),
            'boundaries': torch.tensor(boundaries, dtype=torch.long),
            'raw_spans': processed
        }


def train_main():
    """
    Minimal Python-only training loop for BLTAdapterModel.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding")
    parser.add_argument("--parquet", type=str, default="/data/home/zhangsj/Data/more_big_code_language/python/python_ast_parsed.parquet", help="Path to python parquet with AST_span")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save checkpoints; default uses trail_name under checkpoints/blt_adapter")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log dir; default under output_dir with timestamp")
    parser.add_argument("--trail_name", type=str, default="11_25_blt_adapter_unfreeze_most_local_decoder_cos_decay", help="Trail name")
    # Probe controls
    parser.add_argument("--probe_only", action="store_true", help="Train only node-type probe heads using probe losses")
    parser.add_argument("--num_node_types", type=int, default=113, help="Number of node type classes for probes")
    # LoRA controls
    parser.add_argument("--lora_r", type=int, default=512, help="LoRA rank (increase to use more GPU memory/compute)")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha; defaults to 2 * lora_r if unset")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lora_include_global", dest="lora_include_global", action="store_true", help="Also apply LoRA to global transformer (q/k/v/o and MLP)")
    group.add_argument("--no_lora_include_global", dest="lora_include_global", action="store_false", help="Do not apply LoRA to global transformer")
    parser.set_defaults(lora_include_global=True)
    parser.add_argument("--lora_include_local", dest="lora_include_local", action="store_true", help="Apply LoRA to local decoder modules (disabled by default)")
    parser.set_defaults(lora_include_local=False)
    parser.add_argument("--lora_targets", type=str, default=None, help="Comma-separated module substrings to LoRA; overrides defaults")
    # Loss schedule controls
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lm_warm", type=float, default=0.0)
    parser.add_argument("--node_warm", type=float, default=0.5)
    parser.add_argument("--bnd_warm", type=float, default=0.5)
    parser.add_argument("--mse_warm", type=float, default=0.5)
    parser.add_argument("--kl_warm", type=float, default=0.0)
    parser.add_argument("--nce_warm", type=float, default=0.0)
    parser.add_argument("--lm_main", type=float, default=0.05)
    parser.add_argument("--node_main", type=float, default=1.0)
    parser.add_argument("--bnd_main", type=float, default=0.3)
    parser.add_argument("--mse_main", type=float, default=0.2)
    parser.add_argument("--kl_main", type=float, default=0.3)
    parser.add_argument("--nce_main", type=float, default=0.1)
    parser.add_argument("--infonce_tau", type=float, default=0.07)
    # Probe loss control
    parser.add_argument("--probe_loss_weight", type=float, default=0.1, help="Weight for probe losses when included with main loss")
    # Separate per-loss gradient clipping/backprop
    parser.add_argument("--separate_loss_clipping", action="store_true", help="Clip and accumulate grads per loss component separately")
    parser.add_argument("--loss_clip_norm", type=float, default=1.0, help="Max grad norm per loss component when --separate_loss_clipping is enabled")
    # Span sampling
    parser.add_argument("--max_nodes_per_sample", type=int, default=12)
    parser.add_argument("--min_span_len", type=int, default=3)
    # LM CE weight schedule
    parser.add_argument("--lm_weight_schedule", type=str, default="cosine", choices=["none", "linear", "cosine", "exp"], help="Per-step schedule for LM CE weight")
    parser.add_argument("--lm_weight_start", type=float, default=0.8, help="LM CE weight at the start (step 0)")
    parser.add_argument("--lm_weight_end", type=float, default=0.1, help="LM CE weight at the end (last step)")
    args = parser.parse_args()
    trail_name = args.trail_name
    if not args.output_dir:
        args.output_dir = f"/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/{trail_name}"
    if not args.log_dir:
        args.log_dir = f"/data/home/zhangsj/AST_decoding/tensorboard_logs/{trail_name}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Build dataset first to derive node-type space size from vocab
    dataset = PythonASTSpanDataset(args.parquet, tokenizer, max_length=args.max_length)
    derived_num_node_types = getattr(dataset, "num_node_types", args.num_node_types)
    # Allow configuring local model size
    if not hasattr(args, "local_num_layers"):
        args.local_num_layers = 2
    if not hasattr(args, "max_node_length"):
        args.max_node_length = 64
    adapter = create_blt_adapter_model(
        args.model_path,
        local_num_layers=args.local_num_layers,
        max_node_length=args.max_node_length,
        num_node_types=int(derived_num_node_types)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and args.dtype in ("auto", "bf16") else (
                torch.float16 if args.dtype in ("auto", "fp16") else
                torch.float32
            )
        except Exception:
            dtype = torch.float16 if args.dtype in ("auto", "fp16") else torch.float32
        adapter = adapter.to(device=device, dtype=dtype)
    else:
        adapter = adapter.to(device=device, dtype=torch.float32)

    # Memory optimizations for training
    try:
        adapter.config.use_cache = False
    except Exception:
        pass
    try:
        adapter.gradient_checkpointing_enable()
    except Exception:
        pass
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Freeze/train selection
    trainable_params = []
    if getattr(args, "probe_only", False):
        # Freeze everything first
        for p in adapter.parameters():
            p.requires_grad = False
        # Enable only probe heads
        if hasattr(adapter, 'node_type_probe_encoder') and adapter.node_type_probe_encoder is not None:
            for p in adapter.node_type_probe_encoder.parameters():
                p.requires_grad = True
                trainable_params.append(p)
        if hasattr(adapter, 'node_type_probe_decoder') and adapter.node_type_probe_decoder is not None:
            for p in adapter.node_type_probe_decoder.parameters():
                p.requires_grad = True
                trainable_params.append(p)
        # Ensure lm_head and tied mats remain frozen
        if hasattr(adapter, 'lm_head'):
            for p in adapter.lm_head.parameters():
                p.requires_grad = False
        # Mark model to use probe-only loss composition
        adapter.probe_only = True
    else:
        # Freeze the global transformer (latent transformer); keep local modules trainable
        if hasattr(adapter, 'model') and hasattr(adapter.model, 'layers'):
            for p in adapter.model.layers.parameters():
                p.requires_grad = False
            # Unfreeze the last transformer layer as requested
            try:
                for p in adapter.model.layers[-1].parameters():
                    p.requires_grad = True
                    trainable_params.append(p)
            except Exception:
                pass
        # Keep embed encoder partly trainable: freeze token embeddings, train adapter + ln
        if hasattr(adapter.model, 'embed_tokens'):
            et = adapter.model.embed_tokens
            if hasattr(et, 'token_embeddings'):
                for p in et.token_embeddings.parameters():
                    p.requires_grad = False
            for mod_name in ['token_adapter', 'layer_norm']:
                if hasattr(et, mod_name):
                    for p in getattr(et, mod_name).parameters():
                        p.requires_grad = True
                        trainable_params.append(p)
        # Local decoder and projection modules
        for name in ['latent_proj', 'local_transformer', 'boundary_head', 'latent_from_global']:
            if hasattr(adapter, name):
                for p in getattr(adapter, name).parameters():
                    p.requires_grad = True
                    trainable_params.append(p)
        # Ensure probe heads are optimized by default (their inputs are detached, so only heads update)
        for name in ['node_type_probe_encoder', 'node_type_probe_decoder']:
            if hasattr(adapter, name) and getattr(adapter, name) is not None:
                for p in getattr(adapter, name).parameters():
                    p.requires_grad = True
                    trainable_params.append(p)
        # Freeze tied large matrices explicitly (local_token_embed/local_out_proj already tied & frozen)
        if hasattr(adapter, 'local_token_embed'):
            for p in adapter.local_token_embed.parameters():
                p.requires_grad = False
        if hasattr(adapter, 'local_out_proj'):
            for p in adapter.local_out_proj.parameters():
                p.requires_grad = False
        # Apply PEFT LoRA to selected modules to reduce trainable footprint
        try:
            # Build LoRA target modules
            if args.lora_targets:
                target_modules = [s.strip() for s in str(args.lora_targets).split(",") if s.strip()]
            else:
                target_modules = []
                # Optionally include LOCAL decoder/adapters (off by default)
                if args.lora_include_local:
                    if hasattr(adapter, 'latent_proj'):
                        target_modules.append('latent_proj')
                    if hasattr(adapter, 'local_out_proj'):
                        target_modules.append('local_out_proj')
                    # FF layers inside local transformer blocks
                    target_modules.extend(['ff.0', 'ff.2'])
                    # Local attention out projections (self/cross)
                    target_modules.extend(['self_attn.out_proj', 'cross_attn.out_proj'])
                # Include GLOBAL transformer attention and MLP (on by default)
                if args.lora_include_global:
                    target_modules.extend(['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
            if len(target_modules) > 0:
                lora_alpha = int(args.lora_alpha) if args.lora_alpha is not None else int(args.lora_r) * 2
                lora_config = LoraConfig(
                    task_type="CAUSAL_LM",
                    inference_mode=False,
                    r=int(args.lora_r),
                    lora_alpha=lora_alpha,
                    lora_dropout=float(args.lora_dropout),
                    target_modules=target_modules,
                    bias="none"
                )
                adapter = get_peft_model(adapter, lora_config)
                # Rebuild trainable params to include LoRA adapters + small heads
                trainable_params = [p for p in adapter.parameters() if p.requires_grad]
        except Exception as e:
            print(f"[warn] Failed to apply PEFT LoRA: {e}")
        # Optional legacy node heads if set
        for name in ['node_type_head', 'node_len_head']:
            if hasattr(adapter, name) and getattr(adapter, name) is not None:
                for p in getattr(adapter, name).parameters():
                    p.requires_grad = True
                    trainable_params.append(p)
        # Freeze lm_head by default
        if hasattr(adapter, 'lm_head'):
            for p in adapter.lm_head.parameters():
                p.requires_grad = False

    # Custom collate to handle variable-length raw_spans lists
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
        attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
        span_types = torch.stack([item['span_metadata']['span_types'] for item in batch], dim=0)
        positions = torch.stack([item['span_metadata']['positions'] for item in batch], dim=0)
        boundaries = torch.stack([item['span_metadata']['boundaries'] for item in batch], dim=0)
        raw_spans = [item['span_metadata']['raw_spans'] for item in batch]  # keep as list (variable lengths)
        span_metadata = {
            'span_types': span_types,
            'positions': positions,
            'boundaries': boundaries,
            'raw_spans': raw_spans
        }
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'span_metadata': span_metadata
        }
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    # Only optimize trainable params
    if len(trainable_params) == 0:
        # Fallback: train all params that require grad (in case structure is different)
        trainable_params = [p for p in adapter.parameters() if p.requires_grad]
    opt = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # ===== Logging setup =====
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir if args.log_dir else os.path.join(args.output_dir, f"logs_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Param counts
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_count = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    frozen_count = total_params - trainable_count
    print(f"[setup] total_params={total_params:,} trainable={trainable_count:,} frozen={frozen_count:,}")
    writer.add_text("setup/total_params", str(total_params))
    writer.add_text("setup/trainable_params", str(trainable_count))
    writer.add_text("setup/frozen_params", str(frozen_count))
    writer.add_text("setup/model_path", args.model_path)
    writer.add_text("setup/output_dir", args.output_dir)
    writer.add_text("setup/log_dir", log_dir)
    writer.add_text("setup/dtype", str(dtype))
    writer.add_text("setup/device", str(device))
    writer.add_text("train/hparams", f"lr={args.lr}, batch_size={args.batch_size}, max_length={args.max_length}")
    # Adapter config
    writer.add_text("adapter/boundary_loss_weight", str(getattr(adapter, "boundary_loss_weight", "n/a")))
    writer.add_text("adapter/latent_mse_weight", str(getattr(adapter, "latent_mse_weight", "n/a")))
    writer.add_text("adapter/max_node_length", str(getattr(adapter, "max_node_length", "n/a")))
    writer.add_text("adapter/num_node_types", str(getattr(adapter, "num_node_types", "n/a")))

    os.makedirs(args.output_dir, exist_ok=True)
    adapter.train()
    global_step = 0
    global_last_layer_unfrozen = False
    steps_per_epoch = len(dataloader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    def compute_lm_weight(step: int) -> float:
        if args.lm_weight_schedule == "none":
            return float(adapter.lm_loss_weight)
        start = float(args.lm_weight_start)
        end = float(args.lm_weight_end)
        t = min(max(step / max(1, total_steps - 1), 0.0), 1.0)
        if args.lm_weight_schedule == "linear":
            return (1.0 - t) * start + t * end
        if args.lm_weight_schedule == "cosine":
            # Cosine decay from start -> end
            return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))
        if args.lm_weight_schedule == "exp":
            # Exponential decay
            if start <= 0 or end <= 0:
                return (1.0 - t) * start + t * end
            ratio = end / start
            return float(start * (ratio ** t))
        return (1.0 - t) * start + t * end
    for epoch in range(args.epochs):
        # Schedule weights
        if epoch < int(args.warmup_epochs):
            # LM weight will be overridden per-step if a schedule is set
            if args.lm_weight_schedule == "none":
                adapter.lm_loss_weight = float(args.lm_warm)
            adapter.node_recon_loss_weight = float(args.node_warm)
            adapter.boundary_loss_weight = float(args.bnd_warm)
            adapter.latent_mse_weight = float(args.mse_warm)
            adapter.kl_weight = float(args.kl_warm)
            adapter.infonce_weight = float(args.nce_warm)
        else:
            # Transition to Stage 2: optionally unfreeze LAST global transformer layer once
            if (not global_last_layer_unfrozen) and hasattr(adapter, 'model') and hasattr(adapter.model, 'layers'):
                try:
                    for p in adapter.model.layers[-1].parameters():
                        p.requires_grad = True
                    # Add newly unfrozen params to optimizer
                    newly_trainable = [p for p in adapter.model.layers[-1].parameters() if p.requires_grad]
                    if len(newly_trainable) > 0:
                        opt.add_param_group({"params": newly_trainable})
                        writer.add_text("stage/unfreeze", "Unfroze last global transformer layer", global_step)
                    global_last_layer_unfrozen = True
                except Exception:
                    pass
            # LM weight will be overridden per-step if a schedule is set
            if args.lm_weight_schedule == "none":
                adapter.lm_loss_weight = float(args.lm_main)
            adapter.node_recon_loss_weight = float(args.node_main)
            adapter.boundary_loss_weight = float(args.bnd_main)
            adapter.latent_mse_weight = float(args.mse_main)
            adapter.kl_weight = float(args.kl_main)
            adapter.infonce_weight = float(args.nce_main)
        adapter.infonce_tau = float(args.infonce_tau)
        # Set probe loss weight
        try:
            adapter.probe_loss_weight = float(args.probe_loss_weight)
        except Exception:
            pass
        adapter.max_nodes_per_sample = int(args.max_nodes_per_sample)
        for batch in dataloader:
            # Per-step LM CE weight schedule (decay over training)
            if args.lm_weight_schedule != "none":
                adapter.lm_loss_weight = compute_lm_weight(global_step)
            input_ids = batch['input_ids'].to(adapter.device)
            attention_mask = batch['attention_mask'].to(adapter.device)
            span_metadata = {k: (v.to(adapter.device) if isinstance(v, torch.Tensor) else v) for k, v in batch['span_metadata'].items()}
            outputs = adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_metadata=span_metadata,
                labels=input_ids
            )
            loss = outputs.loss if outputs.loss is not None else outputs.node_recon_loss
            # Loss component logging
            try:
                from math import isnan
                writer.add_scalar("loss/total", float(loss.item()), global_step)
                if hasattr(outputs, "lm_ce"):
                    writer.add_scalar("loss/lm_ce", float(outputs.lm_ce.item()), global_step)
                if hasattr(outputs, "boundary_loss"):
                    writer.add_scalar("loss/boundary", float(outputs.boundary_loss.item()), global_step)
                if hasattr(outputs, "latent_mse"):
                    writer.add_scalar("loss/latent_mse", float(outputs.latent_mse.item()), global_step)
                if hasattr(outputs, "node_recon_loss"):
                    writer.add_scalar("loss/node_recon", float(outputs.node_recon_loss.item()), global_step)
                if hasattr(outputs, "node_type_loss") and outputs.node_type_loss is not None:
                    writer.add_scalar("loss/node_type", float(outputs.node_type_loss.item()), global_step)
                if hasattr(outputs, "node_len_loss") and outputs.node_len_loss is not None:
                    writer.add_scalar("loss/node_len", float(outputs.node_len_loss.item()), global_step)
                if hasattr(outputs, "kl_loss"):
                    writer.add_scalar("loss/kl", float(outputs.kl_loss.item()), global_step)
                if hasattr(outputs, "infonce_loss"):
                    writer.add_scalar("loss/infonce", float(outputs.infonce_loss.item()), global_step)
                if hasattr(outputs, "type_probe_encoder_loss"):
                    writer.add_scalar("loss/type_probe_encoder", float(outputs.type_probe_encoder_loss.item()), global_step)
                if hasattr(outputs, "type_probe_encoder_acc"):
                    writer.add_scalar("acc/type_probe_encoder", float(outputs.type_probe_encoder_acc.item()), global_step)
                if hasattr(outputs, "type_probe_decoder_loss"):
                    writer.add_scalar("loss/type_probe_decoder", float(outputs.type_probe_decoder_loss.item()), global_step)
                if hasattr(outputs, "type_probe_decoder_acc"):
                    writer.add_scalar("acc/type_probe_decoder", float(outputs.type_probe_decoder_acc.item()), global_step)
            except Exception:
                pass
            # GPU memory logging
            if torch.cuda.is_available():
                try:
                    writer.add_scalar("mem/alloc_MB", torch.cuda.memory_allocated() / (1024**2), global_step)
                    writer.add_scalar("mem/reserved_MB", torch.cuda.memory_reserved() / (1024**2), global_step)
                except Exception:
                    pass
            opt.zero_grad()
            if getattr(args, "separate_loss_clipping", False):
                # Build parameter subsets
                def params_of(module_list):
                    ps = []
                    for m in module_list:
                        if m is None:
                            continue
                        try:
                            for p in m.parameters():
                                if getattr(p, "requires_grad", False):
                                    ps.append(p)
                        except Exception:
                            continue
                    return ps
                probe_params: List[torch.nn.Parameter] = []
                if hasattr(adapter, 'node_type_probe_encoder') and adapter.node_type_probe_encoder is not None:
                    probe_params.extend(list(adapter.node_type_probe_encoder.parameters()))
                if hasattr(adapter, 'node_type_probe_decoder') and adapter.node_type_probe_decoder is not None:
                    probe_params.extend(list(adapter.node_type_probe_decoder.parameters()))
                probe_param_ids = {id(p) for p in probe_params}
                # Global LM params (last layer + embed adapters)
                global_modules = []
                try:
                    if hasattr(adapter, 'model') and hasattr(adapter.model, 'layers'):
                        global_modules.append(adapter.model.layers[-1])
                except Exception:
                    pass
                try:
                    if hasattr(adapter.model, 'embed_tokens'):
                        et = adapter.model.embed_tokens
                        for name in ['token_adapter', 'layer_norm']:
                            if hasattr(et, name):
                                global_modules.append(getattr(et, name))
                except Exception:
                    pass
                global_params = params_of(global_modules)
                # Boundary head
                boundary_params = params_of([getattr(adapter, 'boundary_head', None)])
                # Latent-from-global projector
                latent_pred_params = params_of([getattr(adapter, 'latent_from_global', None)])
                # Local decoder path (latent_proj + local_transformer)
                local_params = params_of([getattr(adapter, 'latent_proj', None), getattr(adapter, 'local_transformer', None)])
                # Remove overlaps explicitly
                def unique_list(lst):
                    seen = set()
                    out = []
                    for p in lst:
                        if id(p) not in seen:
                            seen.add(id(p))
                            out.append(p)
                    return out
                global_params = unique_list([p for p in global_params if id(p) not in probe_param_ids])
                boundary_params = unique_list([p for p in boundary_params if id(p) not in probe_param_ids])
                latent_pred_params = unique_list([p for p in latent_pred_params if id(p) not in probe_param_ids])
                local_params = unique_list([p for p in local_params if id(p) not in probe_param_ids])
                # Components: (name, loss, params)
                comps: List[Tuple[str, Optional[torch.Tensor], List[torch.nn.Parameter]]] = []
                # Use already-weighted losses from adapter weights
                try:
                    if hasattr(outputs, "lm_ce") and outputs.lm_ce is not None and float(getattr(adapter, "lm_loss_weight", 0.0)) > 0.0 and len(global_params) > 0:
                        comps.append(("lm", float(getattr(adapter, "lm_loss_weight", 0.0)) * outputs.lm_ce, global_params))
                except Exception:
                    pass
                try:
                    if hasattr(outputs, "boundary_loss") and outputs.boundary_loss is not None and float(getattr(adapter, "boundary_loss_weight", 0.0)) > 0.0 and len(boundary_params) > 0:
                        comps.append(("boundary", float(getattr(adapter, "boundary_loss_weight", 0.0)) * outputs.boundary_loss, boundary_params))
                except Exception:
                    pass
                try:
                    if hasattr(outputs, "latent_mse") and outputs.latent_mse is not None and float(getattr(adapter, "latent_mse_weight", 0.0)) > 0.0 and len(latent_pred_params) > 0:
                        comps.append(("latent_mse", float(getattr(adapter, "latent_mse_weight", 0.0)) * outputs.latent_mse, latent_pred_params))
                except Exception:
                    pass
                try:
                    if hasattr(outputs, "node_recon_loss") and outputs.node_recon_loss is not None and float(getattr(adapter, "node_recon_loss_weight", 0.0)) > 0.0 and len(local_params) > 0:
                        comps.append(("node_recon", float(getattr(adapter, "node_recon_loss_weight", 0.0)) * outputs.node_recon_loss, local_params))
                except Exception:
                    pass
                try:
                    if hasattr(outputs, "kl_loss") and outputs.kl_loss is not None and float(getattr(adapter, "kl_weight", 0.0)) > 0.0 and len(local_params) > 0:
                        comps.append(("kl", float(getattr(adapter, "kl_weight", 0.0)) * outputs.kl_loss, local_params))
                except Exception:
                    pass
                try:
                    if hasattr(outputs, "infonce_loss") and outputs.infonce_loss is not None and float(getattr(adapter, "infonce_weight", 0.0)) > 0.0 and len(local_params) > 0:
                        comps.append(("infonce", float(getattr(adapter, "infonce_weight", 0.0)) * outputs.infonce_loss, local_params))
                except Exception:
                    pass
                # Accumulate per-component grads with per-component clipping
                eps = 1e-12
                max_norm = float(getattr(args, "loss_clip_norm", 1.0))
                for _, comp_loss, comp_params in comps:
                    if comp_loss is None or len(comp_params) == 0:
                        continue
                    try:
                        grads = torch.autograd.grad(comp_loss, comp_params, retain_graph=True, allow_unused=True)
                    except Exception:
                        continue
                    # Compute norm over available grads
                    sq = 0.0
                    for g in grads:
                        if g is not None:
                            sq += float(g.detach().data.norm(2).item() ** 2)
                    if sq == 0.0:
                        scale = 1.0
                    else:
                        total = sq ** 0.5
                        scale = min(1.0, max_norm / (total + eps))
                    # Accumulate into .grad
                    for p, g in zip(comp_params, grads):
                        if g is None:
                            continue
                        if p.grad is None:
                            p.grad = scale * g
                        else:
                            p.grad = p.grad + scale * g
                # Train probes separately (no coupling)
                if not getattr(adapter, "probe_only", False):
                    probe_total = None
                    try:
                        if hasattr(outputs, "type_probe_encoder_loss") and outputs.type_probe_encoder_loss is not None:
                            probe_total = outputs.type_probe_encoder_loss if probe_total is None else probe_total + outputs.type_probe_encoder_loss
                        if hasattr(outputs, "type_probe_decoder_loss") and outputs.type_probe_decoder_loss is not None:
                            probe_total = outputs.type_probe_decoder_loss if probe_total is None else probe_total + outputs.type_probe_decoder_loss
                    except Exception:
                        probe_total = None
                    if probe_total is not None and len(probe_params) > 0:
                        try:
                            grads = torch.autograd.grad(probe_total, probe_params, retain_graph=True, allow_unused=True)
                            sq = 0.0
                            for g in grads:
                                if g is not None:
                                    sq += float(g.detach().data.norm(2).item() ** 2)
                            if sq == 0.0:
                                scale = 1.0
                            else:
                                total = sq ** 0.5
                                scale = min(1.0, max_norm / (total + 1e-12))
                            for p, g in zip(probe_params, grads):
                                if g is None:
                                    continue
                                if p.grad is None:
                                    p.grad = scale * g
                                else:
                                    p.grad = p.grad + scale * g
                        except Exception:
                            pass
                # Log total grad norm after accumulation
                try:
                    total_grad_norm = 0.0
                    for p in trainable_params:
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_grad_norm += float(param_norm.item() ** 2)
                    total_grad_norm = total_grad_norm ** 0.5
                    writer.add_scalar("grad/total_norm", total_grad_norm, global_step)
                except Exception:
                    pass
                opt.step()
            else:
                # Standard single backward path with separate probe training and clipping
                loss.backward()
                # Backprop probe heads separately (do not include in total_loss); inputs are detached so only probes update
                if not getattr(adapter, "probe_only", False):
                    probe_total = None
                    try:
                        if hasattr(outputs, "type_probe_encoder_loss") and outputs.type_probe_encoder_loss is not None:
                            probe_total = outputs.type_probe_encoder_loss if probe_total is None else probe_total + outputs.type_probe_encoder_loss
                        if hasattr(outputs, "type_probe_decoder_loss") and outputs.type_probe_decoder_loss is not None:
                            probe_total = outputs.type_probe_decoder_loss if probe_total is None else probe_total + outputs.type_probe_decoder_loss
                    except Exception:
                        probe_total = None
                    if probe_total is not None:
                        try:
                            probe_total.backward()
                        except Exception:
                            pass
                # Grad norms before clipping
                try:
                    total_grad_norm = 0.0
                    for p in trainable_params:
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_grad_norm += float(param_norm.item() ** 2)
                    total_grad_norm = total_grad_norm ** 0.5
                    writer.add_scalar("grad/total_norm", total_grad_norm, global_step)
                except Exception:
                    pass
                # Clip grads separately for main vs probe params to avoid coupling
                try:
                    probe_params2: List[torch.nn.Parameter] = []
                    if hasattr(adapter, 'node_type_probe_encoder') and adapter.node_type_probe_encoder is not None:
                        probe_params2.extend(list(adapter.node_type_probe_encoder.parameters()))
                    if hasattr(adapter, 'node_type_probe_decoder') and adapter.node_type_probe_decoder is not None:
                        probe_params2.extend(list(adapter.node_type_probe_decoder.parameters()))
                    probe_param_ids2 = {id(p) for p in probe_params2}
                    main_params2 = [p for p in adapter.parameters() if getattr(p, "requires_grad", False) and id(p) not in probe_param_ids2]
                    if len(main_params2) > 0:
                        torch.nn.utils.clip_grad_norm_(main_params2, 1.0)
                    if len(probe_params2) > 0:
                        torch.nn.utils.clip_grad_norm_(probe_params2, 1.0)
                except Exception:
                    # Fallback to clipping all if separation fails
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                opt.step()
            global_step += 1
            # Current LR
            try:
                current_lr = opt.param_groups[0]["lr"]
                writer.add_scalar("opt/lr", float(current_lr), global_step)
                if args.lm_weight_schedule != "none":
                    writer.add_scalar("weight/lm_ce", float(adapter.lm_loss_weight), global_step)
                # Log other composite loss weights
                writer.add_scalar("weight/node_recon", float(getattr(adapter, "node_recon_loss_weight", 0.0)), global_step)
                writer.add_scalar("weight/boundary", float(getattr(adapter, "boundary_loss_weight", 0.0)), global_step)
                writer.add_scalar("weight/latent_mse", float(getattr(adapter, "latent_mse_weight", 0.0)), global_step)
                writer.add_scalar("weight/kl", float(getattr(adapter, "kl_weight", 0.0)), global_step)
                writer.add_scalar("weight/infonce", float(getattr(adapter, "infonce_weight", 0.0)), global_step)
            except Exception:
                pass
            if global_step % 50 == 0:
                msg = f"epoch {epoch+1} step {global_step} | total_loss {float(loss.item()):.4f}"
                if hasattr(outputs, "lm_ce"):
                    msg += f" | lm_ce {float(outputs.lm_ce.item()):.4f}"
                if hasattr(outputs, "boundary_loss"):
                    msg += f" | boundary_ce {float(outputs.boundary_loss.item()):.4f}"
                if hasattr(outputs, "latent_mse"):
                    msg += f" | latent_mse {float(outputs.latent_mse.item()):.4f}"
                if hasattr(outputs, "node_recon_loss"):
                    msg += f" | node_recon_ce {float(outputs.node_recon_loss.item()):.4f}"
                if hasattr(outputs, "type_probe_encoder_loss"):
                    msg += f" | type_probe_encoder_ce {float(outputs.type_probe_encoder_loss.item()):.4f}"
                if hasattr(outputs, "type_probe_encoder_acc"):
                    msg += f" (type_probe_encoder_acc {float(outputs.type_probe_encoder_acc.item()):.3f})"
                if hasattr(outputs, "type_probe_decoder_loss"):
                    msg += f" | type_probe_decoder_ce {float(outputs.type_probe_decoder_loss.item()):.4f}"
                if hasattr(outputs, "type_probe_decoder_acc"):
                    msg += f" (type_probe_decoder_acc {float(outputs.type_probe_decoder_acc.item()):.3f})"
                if hasattr(outputs, "kl_loss"):
                    msg += f" | kl_div {float(outputs.kl_loss.item()):.4f}"
                if hasattr(outputs, "infonce_loss"):
                    msg += f" | info_nce {float(outputs.infonce_loss.item()):.4f}"
                print(msg)
        # Save per epoch
        save_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        # If PEFT is active, save base model and adapter separately to preserve trained local decoder weights
        try:
            is_peft = (PeftModel is not None) and isinstance(adapter, PeftModel)  # type: ignore[arg-type]
        except Exception:
            is_peft = False
        if is_peft:
            base_dir = os.path.join(save_dir, "base_model")
            lora_dir = os.path.join(save_dir, "lora_adapter")
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(lora_dir, exist_ok=True)
            # Save underlying base (with trained local decoder etc.)
            try:
                adapter.get_base_model().save_pretrained(base_dir)  # type: ignore[attr-defined]
            except Exception:
                try:
                    adapter.base_model.save_pretrained(base_dir)  # type: ignore[attr-defined]
                except Exception:
                    # Fallback: attempt to save the whole model
                    adapter.save_pretrained(base_dir)
            # Save LoRA adapter separately
            adapter.save_pretrained(lora_dir)
            print(f"Saved base model to {base_dir} and LoRA adapter to {lora_dir}")
        else:
            adapter.save_pretrained(save_dir)
            print(f"Saved checkpoint to {save_dir}")
        tokenizer.save_pretrained(save_dir)
        try:
            writer.add_text("checkpoints/epoch", f"Saved checkpoint: {save_dir}", global_step)
        except Exception:
            pass

    try:
        writer.add_text("training/status", "COMPLETED", global_step)
        writer.close()
    except Exception:
        pass

if __name__ == "__main__":
    # Run training main if executed directly
    train_main()


