from typing import Dict, List, Optional, Tuple
import os
if 'LOCAL_RANK' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
        num_node_types: int = 128,
    ):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_node_length = int(max_node_length)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.latent_mse_weight = float(latent_mse_weight)
        self.num_node_types = int(num_node_types)
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

        # Replace base embedding with mean-pooled span encoder (token-level outputs)
        self.model.embed_tokens = MeanPooledSpanEncoder(config)

        # Local decoder components
        nhead = max(1, self.hidden_size // 64)
        dim_ff = max(self.hidden_size * 4, 512)
        self.local_token_embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.latent_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.local_transformer = LocalCausalTransformer(
            hidden_size=self.hidden_size,
            nhead=nhead,
            dim_ff=dim_ff,
            num_layers=local_num_layers,
            dropout=local_dropout,
            max_len=self.max_node_length + 1,
        )
        self.local_out_proj = nn.Linear(self.hidden_size, self.vocab_size)

        # Tie and freeze large vocab projections/embeddings to avoid duplicating VxH params
        try:
            # Tie local token embed to base token embeddings
            self.local_token_embed.weight = self.model.embed_tokens.token_embeddings.weight  # type: ignore[attr-defined]
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

    def copy_base_embeddings_from(self, base: Qwen2ForCausalLM) -> None:
        """
        Copy token embeddings from a base Qwen model into our local encoder's token embeddings.
        """
        with torch.no_grad():
            self.model.embed_tokens.token_embeddings.weight.copy_(base.model.embed_tokens.weight)

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
        # Compute token embeddings (local encoder)
        inputs_embeds = self.model.embed_tokens(input_ids, span_metadata)  # [B, L, H]

        # Forward through global transformer using inputs_embeds
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ['input_ids', 'inputs_embeds', 'labels', 'output_hidden_states', 'return_dict']
        }
        # We need last hidden states for auxiliary heads only during training with span metadata
        need_hidden_states = bool(self.training and span_metadata is not None and attention_mask is not None)
        # Respect caller request while ensuring we have hidden states when needed
        want_hidden_states = bool(kwargs.get('output_hidden_states', False) or need_hidden_states)
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=want_hidden_states,
            return_dict=True,  # force dict to access fields reliably
            **filtered_kwargs,
        )

        # Preserve raw LM cross-entropy before composing total loss
        total_loss = None
        base_lm_ce = outputs.loss if hasattr(outputs, "loss") else None
        if base_lm_ce is not None:
            try:
                outputs.lm_ce = base_lm_ce.detach()
            except Exception:
                outputs.lm_ce = base_lm_ce
            total_loss = self.lm_loss_weight * base_lm_ce

        # Add learned boundary + latent regression losses if training and spans are present
        if self.training and span_metadata is not None and attention_mask is not None:
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
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    return_dict=True,
                )
                last_hidden = base_out.last_hidden_state  # [B, L, H]
            # Do not backprop into the global transformer for auxiliary heads to save memory
            last_hidden_for_heads = last_hidden.detach() if isinstance(last_hidden, torch.Tensor) else last_hidden

            # Boundary supervision from span_metadata['boundaries'] (1=start, 3=single => positive)
            if isinstance(last_hidden_for_heads, torch.Tensor) and 'boundaries' in span_metadata:
                boundaries = span_metadata['boundaries'].to(last_hidden_for_heads.device)  # [B, L]
                boundary_targets = ((boundaries == 1) | (boundaries == 3)).long()  # [B, L]
                # Mask to valid positions
                mask = attention_mask.to(torch.bool) if attention_mask is not None else torch.ones_like(boundary_targets, dtype=torch.bool, device=boundary_targets.device)
                logits = self.boundary_head(last_hidden_for_heads)  # [B, L, 2]
                ce_loss = F.cross_entropy(
                    logits[mask].view(-1, 2),
                    boundary_targets[mask].view(-1),
                )
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
                        # Target latent = mean of inputs_embeds on span indices
                        idxs_t = torch.tensor(idxs, device=inputs_embeds.device, dtype=torch.long)
                        target_latent = inputs_embeds[b, idxs_t, :].mean(dim=0)  # [H]
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

        # Add local node reconstruction loss only during training and when labels/spans present
        if self.training and labels is not None and span_metadata is not None and 'raw_spans' in span_metadata:
            node_losses = self._compute_local_node_recon_loss(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                span_metadata=span_metadata,
                last_hidden_for_heads=last_hidden_for_heads,
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
                    # If probe-only, replace/compose total loss from probes
                    if getattr(self, "probe_only", False):
                        probe_total = None
                        if 'type_probe_encoder_loss' in aux:
                            probe_total = aux['type_probe_encoder_loss'] if probe_total is None else probe_total + aux['type_probe_encoder_loss']
                        if 'type_probe_decoder_loss' in aux:
                            probe_total = aux['type_probe_decoder_loss'] if probe_total is None else probe_total + aux['type_probe_decoder_loss']
                        # Fallback to zero if no probe loss available
                        if probe_total is None:
                            probe_total = torch.zeros((), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        total_loss = probe_total
        outputs.loss = total_loss
        return outputs

    def _segment_non_overlapping(self, raw_spans: List[Dict], seq_len: int) -> List[Dict]:
        """
        Build non-overlapping, left-to-right spans from possibly overlapping spans.
        Strategy: sort by (start, -length), greedily select spans that don't intersect.
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
        # sort
        spans.sort(key=lambda x: (x['start'], -x['length']))
        used = np.zeros(seq_len, dtype=bool)
        selected = []
        for sp in spans:
            idxs = sp['indices']
            if not used[idxs].any():
                used[idxs] = True
                selected.append(sp)
        return selected

    def _compute_local_node_recon_loss(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        span_metadata: Dict,
        last_hidden_for_heads: Optional[torch.Tensor] = None
    ) -> Optional[Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]]:
        """
        Teacher-forced local node reconstruction with cross-attention to span latent.
        Returns (recon_loss, aux_losses_dict) or None if no nodes.
        """
        device = inputs_embeds.device
        batch_size, seq_len = input_ids.shape

        node_input_seqs = []
        node_target_seqs = []
        node_encoder_latents = []  # Raw encoder latents (for combining later)
        node_pred_latents = []
        node_type_ids: List[int] = []

        total_nodes_kept = 0

        # Also collect global hidden at span start for residual connection
        node_global_hiddens: List[torch.Tensor] = []
        
        # Track batch/position indices for gradient-enabled indexing later
        node_batch_indices: List[int] = []
        node_start_indices: List[int] = []
        node_span_indices: List[List[int]] = []  # Token indices for each span
        
        for b in range(batch_size):
            raw_list = span_metadata.get('raw_spans', [])
            if not raw_list or b >= len(raw_list):
                continue
            item_spans = raw_list[b]
            sel = self._segment_non_overlapping(item_spans, seq_len)
            if not sel:
                continue
            # Limit number of nodes per sample to bound memory
            if len(sel) > getattr(self, "max_nodes_per_sample", 16):
                sel = sel[:getattr(self, "max_nodes_per_sample", 16)]
            for sp in sel:
                idxs = sp['indices']
                if len(idxs) == 0:
                    continue
                token_seq = input_ids[b, torch.tensor(idxs, device=device)].detach()
                L = int(token_seq.shape[0])
                if L <= 0:
                    continue
                # Encoder latent: mean-pooled input embeddings over span
                encoder_latent = inputs_embeds[b, torch.tensor(idxs, device=inputs_embeds.device), :].mean(dim=0)
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
                node_encoder_latents.append(encoder_latent.detach())  # Detached for now
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
        
        # === BATCHED GRADIENT-ENABLED OPERATIONS ===
        # Build encoder latents with gradient flow through inputs_embeds
        encoder_latents_list = []
        for i in range(total_nodes_kept):
            b = node_batch_indices[i]
            idxs = node_span_indices[i]
            # Mean-pool with gradient flow
            span_embeds = inputs_embeds[b, idxs, :]  # [span_len, H]
            encoder_latents_list.append(span_embeds.mean(dim=0))  # [H]
        encoder_latents_batch = torch.stack(encoder_latents_list, dim=0)  # [N, H]
        
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

        # Local decoder forward (cross-attn to span latent)
        tok_emb = self.local_token_embed(inp_batch)  # [N, L, H]
        teacher_span_latent = self.latent_proj(latents_batch).unsqueeze(1)  # [N, 1, H]
        dec_out_teacher = self.local_transformer(tok_emb, teacher_span_latent)  # [N, L, H]
        
        # === NEW: Add residual from global hidden (for single-token shortcut) ===
        if hasattr(self, 'global_residual_gate') and hasattr(self, 'global_residual_scale'):
            # Expand global hidden to match sequence length: [N, H] -> [N, L, H]
            global_hidden_expanded = global_hiddens_batch.unsqueeze(1).expand_as(dec_out_teacher)
            
            # Compute gate: how much to blend global vs local
            # Gate input: concat of local decoder output and global hidden
            gate_input = torch.cat([dec_out_teacher, global_hidden_expanded], dim=-1)  # [N, L, 2H]
            gate = self.global_residual_gate(gate_input)  # [N, L, 1]
            
            # Blend: gate * global + (1-gate) * local
            # When gate -> 1: output is dominated by global (good for single-token)
            # When gate -> 0: output is dominated by local decoder (good for multi-token)
            dec_out_teacher = (1 - gate) * dec_out_teacher + gate * self.global_residual_scale * global_hidden_expanded
        
        logits_teacher = self.local_out_proj(dec_out_teacher)  # [N, L, V]

        vocab = logits_teacher.size(-1)
        recon_loss = F.cross_entropy(
            logits_teacher.view(-1, vocab),
            torch.where(mask_batch.view(-1), tgt_batch.view(-1), torch.full_like(tgt_batch.view(-1), -100)),
            ignore_index=-100
        )

        aux_losses: Dict[str, torch.Tensor] = {}
        # Student path: predicted latent via last_hidden_for_heads (if available)
        # SKIP if both KL and InfoNCE weights are 0 to save memory
        kl_weight = float(getattr(self, 'kl_weight', 0.0))
        infonce_weight = float(getattr(self, 'infonce_weight', 0.0))
        if isinstance(last_hidden_for_heads, torch.Tensor) and (kl_weight > 0 or infonce_weight > 0):
            # Use predicted span latents derived from global hidden state at span starts
            student_span_latent = self.latent_proj(pred_latents_batch).unsqueeze(1)  # [N,1,H]
            dec_out_student = self.local_transformer(tok_emb, student_span_latent)  # [N,L,H]
            
            # Apply same residual connection for student path
            if hasattr(self, 'global_residual_gate') and hasattr(self, 'global_residual_scale'):
                global_hidden_expanded = global_hiddens_batch.unsqueeze(1).expand_as(dec_out_student)
                gate_input = torch.cat([dec_out_student, global_hidden_expanded], dim=-1)
                gate = self.global_residual_gate(gate_input)
                dec_out_student = (1 - gate) * dec_out_student + gate * self.global_residual_scale * global_hidden_expanded
            
            logits_student = self.local_out_proj(dec_out_student)  # [N,L,V]
            # KL between student and teacher on valid positions
            if kl_weight > 0:
                try:
                    log_p_student = F.log_softmax(logits_student[mask_batch], dim=-1)
                    p_teacher = F.softmax(logits_teacher[mask_batch].detach(), dim=-1)
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

        return recon_loss, (aux_losses if len(aux_losses) > 0 else None)

    @torch.no_grad()
    def generate_node_tokens(
        self, 
        span_latent: torch.Tensor, 
        global_hidden: Optional[torch.Tensor] = None,
        max_len: int = 64, 
        bos_id: Optional[int] = None, 
        eos_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Greedy decode one node from a span latent using the local decoder.
        
        Args:
            span_latent: [H] - the combined (encoder+global) span latent
            global_hidden: [H] - optional global hidden for residual connection
            max_len: maximum tokens to generate
            bos_id: beginning of sequence token id
            eos_id: end of sequence token id
            
        Returns:
            token ids [<=max_len]
        """
        self.eval()
        device = span_latent.device
        bos = bos_id if bos_id is not None else self.node_bos_id
        eos = eos_id if eos_id is not None else getattr(self.config, 'eos_token_id', None)

        tokens: List[int] = [bos]
        cond = self.latent_proj(span_latent).unsqueeze(0).unsqueeze(1)  # [1,1,H]
        
        # Prepare global hidden for residual if available
        has_residual = (
            global_hidden is not None and 
            hasattr(self, 'global_residual_gate') and 
            hasattr(self, 'global_residual_scale')
        )

        for _ in range(max_len):
            inp = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)  # [1,T]
            x = self.local_token_embed(inp)  # [1,T,H]
            h = self.local_transformer(x, cond)  # [1,T,H]
            
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
        return torch.tensor(tokens[1:], device=device, dtype=torch.long)


def create_blt_adapter_model(
    model_path: str = "/data/home/zhangsj/AST_decoding",
    local_num_layers: int = 2,
    local_dropout: float = 0.1,
    max_node_length: int = 64,
    num_node_types: int = 128,
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
        num_node_types=num_node_types
    )

    # Copy transformer weights (except embeddings which we handle separately)
    print("Copying transformer weights into adapter...")
    copied = 0
    for name, param in base_model.named_parameters():
        if name != "model.embed_tokens.weight":
            if name in adapter.state_dict():
                with torch.no_grad():
                    adapter.state_dict()[name].copy_(param)
                    copied += 1
            else:
                # Non-fatal; adapter has additional modules
                pass

    # Copy token embeddings
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


