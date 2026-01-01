import os
import sys
import argparse
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Set, DefaultDict
from collections import defaultdict

# Make project root importable
PROJECT_ROOT = "/data/home/zhangsj"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Avoid HuggingFace tokenizer's "forked after parallelism" spam if any caller later forks.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer
from AST_decoding.blt_adapter_model import create_blt_adapter_model, BLTAdapterModel  # type: ignore
# Optional PEFT import (only needed if --peft_adapter is provided)
try:
    from peft import PeftModel  # type: ignore
except Exception:  # pragma: no cover
    PeftModel = None  # type: ignore

_BANNED_TOKEN_IDS_CACHE: Dict[Tuple[str, int], List[int]] = {}


def _get_banned_token_ids_for_textual_guard(tokenizer) -> List[int]:
    """
    Cache-heavy helper. Building the banned-id list by decoding the full vocab is expensive,
    and `incremental_generate()` is called once per task in EvalPlus.
    """
    try:
        key = (str(getattr(tokenizer, "name_or_path", "tokenizer")), int(len(tokenizer)))
    except Exception:
        key = ("tokenizer", 0)
    cached = _BANNED_TOKEN_IDS_CACHE.get(key)
    if cached is not None:
        return cached

    banned_ids: List[int] = []
    for tok_str in ["#", '"""', "'''"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0:
                banned_ids.append(tid)
        except Exception:
            pass
    try:
        vocab_n = int(len(tokenizer))
        for tid in range(vocab_n):
            if tid in banned_ids:
                continue
            txt = tokenizer.decode([tid], skip_special_tokens=True)
            if ("#") in txt or '"""' in txt or "'''" in txt:
                banned_ids.append(tid)
    except Exception:
        pass

    banned_ids = sorted(set(int(x) for x in banned_ids))
    _BANNED_TOKEN_IDS_CACHE[key] = banned_ids
    return banned_ids


def select_device(preferred_device: str = "auto") -> str:
    """
    Select device. GPU selection is handled by CUDA_VISIBLE_DEVICES environment variable.
    This function only selects between "cuda" (GPU 0 from PyTorch's perspective) and "cpu".
    """
    if preferred_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred_device == "cuda":
        if not torch.cuda.is_available():
            return "cpu"
        return "cuda"
    if preferred_device == "cpu":
        return "cpu"
    # Default fallback
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(device: str, preferred_dtype: str = "auto") -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if preferred_dtype == "bf16":
        return torch.bfloat16
    if preferred_dtype == "fp16":
        return torch.float16
    if preferred_dtype == "fp32":
        return torch.float32
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        return torch.float16


def load_adapter_and_tokenizer(checkpoint_path: Optional[str], model_path: str, device: str, dtype: torch.dtype, peft_adapter: Optional[str] = None) -> (BLTAdapterModel, Any):
    if checkpoint_path and os.path.isdir(checkpoint_path):
        # Prefer loading tokenizer from checkpoint, fallback to model_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Try standard HF load; if it fails, fallback to state_dict on a freshly created adapter
        adapter = None
        load_error: Optional[Exception] = None
        try:
            print(f"[load] Attempting to load checkpoint from {checkpoint_path} using from_pretrained...")
            # Capture loading info so we can detect partially-initialized modules (which can silently tank quality).
            base, loading_info = BLTAdapterModel.from_pretrained(  # type: ignore[misc]
                checkpoint_path,
                torch_dtype=dtype,  # type: ignore[arg-type]
                output_loading_info=True,
            )
            missing_keys = list((loading_info or {}).get("missing_keys", []))
            unexpected_keys = list((loading_info or {}).get("unexpected_keys", []))
            if missing_keys:
                print(f"[load] WARNING: from_pretrained missing {len(missing_keys)} key(s). Example: {missing_keys[:8]}")
            if unexpected_keys:
                print(f"[load] WARNING: from_pretrained unexpected {len(unexpected_keys)} key(s). Example: {unexpected_keys[:8]}")

            # Backward-compat: older checkpoints may only have `cross_attn_global.*` (nn.MultiheadAttention)
            # while newer code defaults to using the SDPA module `cross_attn_global_sdpa.*` at runtime.
            # If SDPA weights are missing, we MUST disable the SDPA path, otherwise inference uses random weights.
            if any("cross_attn_global_sdpa" in k for k in missing_keys):
                try:
                    num_flipped = 0
                    if hasattr(base, "local_decoder") and hasattr(base.local_decoder, "layers"):
                        for layer in base.local_decoder.layers:
                            if hasattr(layer, "use_sdpa_global_attn"):
                                layer.use_sdpa_global_attn = False
                                num_flipped += 1
                    print(
                        f"[load] Detected checkpoint without SDPA global-attn weights; "
                        f"disabled SDPA path for {num_flipped} local decoder layer(s) to use loaded MultiheadAttention weights."
                    )
                except Exception as _e:
                    print(f"[load] WARNING: Failed to disable SDPA global-attn path automatically: {_e}")
            adapter = PeftModel.from_pretrained(base, peft_adapter) if (peft_adapter and os.path.isdir(peft_adapter)) else base  # type: ignore[assignment]
            print(f"[load] Successfully loaded checkpoint using from_pretrained")
        except Exception as e:
            load_error = e
            print(f"[load] from_pretrained failed: {e}")
            print(f"[load] Falling back to state_dict loading...")
            # Robust fallback: instantiate adapter from base path then load state_dict with strict=False
            base = create_blt_adapter_model(model_path)
            state_path_bin = os.path.join(checkpoint_path, "pytorch_model.bin")
            state_path_safetensors = os.path.join(checkpoint_path, "model.safetensors")
            state_dict = None
            if os.path.isfile(state_path_bin):
                print(f"[load] Loading state_dict from {state_path_bin}")
                state_dict = torch.load(state_path_bin, map_location="cpu")
            elif os.path.isfile(state_path_safetensors):
                try:
                    from safetensors.torch import load_file as load_safetensors  # type: ignore
                    print(f"[load] Loading state_dict from {state_path_safetensors}")
                    state_dict = load_safetensors(state_path_safetensors)
                except Exception as e2:
                    print(f"[load] Failed to load safetensors: {e2}")
                    state_dict = None
            else:
                print(f"[load] WARNING: No state_dict file found in {checkpoint_path}")
            
            if state_dict is not None:
                try:
                    model_sd = base.state_dict()
                    filtered_state = {}
                    skipped = []
                    for k, v in state_dict.items():
                        if k in model_sd and model_sd[k].shape == v.shape:
                            filtered_state[k] = v
                        else:
                            skipped.append(k)
                    
                    if len(skipped) > 0:
                        print(f"[load] Skipped {len(skipped)} keys (shape mismatch or not in model): {skipped[:10]}...")
                    
                    if len(filtered_state) > 0:
                        missing_keys, unexpected_keys = base.load_state_dict(filtered_state, strict=False)
                        print(f"[load] Loaded {len(filtered_state)} parameters from checkpoint")
                        if missing_keys:
                            print(f"[load] WARNING: {len(missing_keys)} missing keys: {list(missing_keys)[:10]}...")
                        if unexpected_keys:
                            print(f"[load] WARNING: {len(unexpected_keys)} unexpected keys: {list(unexpected_keys)[:10]}...")
                    else:
                        print(f"[load] ERROR: No parameters could be loaded from checkpoint!")
                        raise RuntimeError(f"Failed to load any parameters from {checkpoint_path}")
                except Exception as e2:
                    print(f"[load] ERROR: Failed to load state_dict: {e2}")
                    raise
            else:
                print(f"[load] ERROR: No state_dict available to load!")
                raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}: no state_dict file found")
            
            adapter = PeftModel.from_pretrained(base, peft_adapter) if (peft_adapter and os.path.isdir(peft_adapter)) else base  # type: ignore[assignment]

            # Same backward-compat runtime safety for the fallback path:
            # if the checkpoint doesn't contain SDPA weights, don't route inference through them.
            try:
                if state_dict is not None:
                    sd_keys = list(state_dict.keys())
                    has_sdpa = any("cross_attn_global_sdpa" in k for k in sd_keys)
                    has_mha = any("cross_attn_global." in k for k in sd_keys)
                    if (not has_sdpa) and has_mha:
                        num_flipped = 0
                        if hasattr(base, "local_decoder") and hasattr(base.local_decoder, "layers"):
                            for layer in base.local_decoder.layers:
                                if hasattr(layer, "use_sdpa_global_attn"):
                                    layer.use_sdpa_global_attn = False
                                    num_flipped += 1
                        print(
                            f"[load] Checkpoint provides MultiheadAttention global-attn weights but not SDPA; "
                            f"disabled SDPA path for {num_flipped} local decoder layer(s)."
                        )
            except Exception as _e:
                print(f"[load] WARNING: Failed to apply SDPA/MHA compatibility toggle in fallback load: {_e}")
        # Move to device with graceful OOM fallback
        try:
            adapter = adapter.to(device=device, dtype=dtype)  # type: ignore[union-attr]
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                adapter = adapter.to(device="cpu", dtype=torch.float32)  # type: ignore[union-attr]
            else:
                # If both HF and fallback failed earlier, raise the root error; else raise this
                raise load_error or e
    else:
        # Fresh adapter wrapping base model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base = create_blt_adapter_model(model_path)
        adapter = PeftModel.from_pretrained(base, peft_adapter) if (peft_adapter and os.path.isdir(peft_adapter)) else base
        try:
            adapter = adapter.to(device=device, dtype=dtype)
        except RuntimeError as e:
            # Graceful fallback to CPU on CUDA OOM
            if "out of memory" in str(e).lower() and device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                adapter = adapter.to(device="cpu", dtype=torch.float32)
            else:
                raise
    # Reduce memory during inference
    try:
        # For EvalPlus-style decoding, KV cache is a large speed win.
        # We'll still be able to force full recomputation when a span rewrite happens.
        adapter.config.use_cache = True
    except Exception:
        pass
    adapter.eval()
    
    # Verify checkpoint was loaded by checking trainable parameters
    # This helps catch cases where the checkpoint wasn't actually loaded
    if checkpoint_path and os.path.isdir(checkpoint_path):
        try:
            # Get checksums of key trainable components to verify they're different
            checksums = {}
            if hasattr(adapter, 'boundary_head') and adapter.boundary_head is not None:
                boundary_params = list(adapter.boundary_head.parameters())
                if boundary_params:
                    # Compute a simple checksum: sum of all parameter values
                    boundary_sum = sum(p.sum().item() for p in boundary_params)
                    checksums['boundary_head'] = boundary_sum
                    print(f"[load] Verification: boundary_head param sum = {boundary_sum:.6f}")
            
            if hasattr(adapter, 'latent_from_global') and adapter.latent_from_global is not None:
                latent_params = list(adapter.latent_from_global.parameters())
                if latent_params:
                    latent_sum = sum(p.sum().item() for p in latent_params)
                    checksums['latent_from_global'] = latent_sum
                    print(f"[load] Verification: latent_from_global param sum = {latent_sum:.6f}")
            
            if hasattr(adapter, 'local_transformer') and adapter.local_transformer is not None:
                local_params = list(adapter.local_transformer.parameters())
                if local_params:
                    local_sum = sum(p.sum().item() for p in local_params[:5])  # First 5 params as sample
                    checksums['local_transformer_sample'] = local_sum
                    print(f"[load] Verification: local_transformer sample param sum = {local_sum:.6f}")
            
            # Print checkpoint path for debugging
            print(f"[load] Checkpoint loaded from: {checkpoint_path}")
            print(f"[load] Parameter checksums: {checksums}")
            
        except Exception as e:
            print(f"[load] Verification check failed: {e}")
            import traceback
            traceback.print_exc()
    
    return adapter, tokenizer


def is_boundary_heuristic(tokenizer, token_id: int) -> bool:
    """
    Simple heuristic: boundary if decoded token ends with whitespace or punctuation.
    """
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=True)
    except Exception:
        return False
    if len(text) == 0:
        return False
    ch = text[-1]
    return ch.isspace() or ch in {':', ';', ',', '.', '(', ')', '{', '}', '[', ']', '-', '=', '+', '*', '/', '\\'}


def compute_entropy(logits: torch.Tensor) -> float:
    """
    logits: [V]
    """
    probs = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    ent = -torch.sum(probs * logp).item()
    return ent


def _build_no_repeat_ngram_index(sequence: List[int], n: int) -> DefaultDict[Tuple[int, ...], Set[int]]:
    """
    Build mapping: (n-1)-gram prefix -> set(next_token) observed in the sequence.
    """
    index: DefaultDict[Tuple[int, ...], Set[int]] = defaultdict(set)
    if n <= 0 or len(sequence) < n:
        return index
    for i in range(len(sequence) - n + 1):
        prefix = tuple(sequence[i:i + n - 1])
        next_tok = sequence[i + n - 1]
        index[prefix].add(next_tok)
    return index


def _apply_repetition_penalty_and_ngram_blocking(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """
    Apply CTRL-style repetition penalty and HF-style no-repeat-ngram blocking in-place on logits.
    logits: [1, V]
    generated_ids: [1, T]
    """
    if logits.dim() != 2 or logits.size(0) != 1:
        return logits
    if generated_ids is None or generated_ids.numel() == 0:
        return logits

    logits_view = logits[0]

    # 1) Repetition penalty (CTRL): penalize tokens that already appeared
    if repetition_penalty and repetition_penalty > 1.0:
        unique_ids = set(int(t) for t in generated_ids[0].tolist())
        if len(unique_ids) > 0:
            token_logits = logits_view[torch.tensor(list(unique_ids), device=logits.device, dtype=torch.long)]
            neg_mask = token_logits < 0
            token_logits[neg_mask] = token_logits[neg_mask] * repetition_penalty
            token_logits[~neg_mask] = token_logits[~neg_mask] / repetition_penalty
            logits_view.scatter_(0, torch.tensor(list(unique_ids), device=logits.device, dtype=torch.long), token_logits)

    # 2) No-repeat-ngram blocking (HF): forbid tokens that would close an already seen n-gram
    if no_repeat_ngram_size and no_repeat_ngram_size > 0 and generated_ids.size(1) >= no_repeat_ngram_size - 1:
        seq_list = generated_ids[0].tolist()
        n = int(no_repeat_ngram_size)
        index = _build_no_repeat_ngram_index(seq_list, n)
        prefix = tuple(seq_list[-(n - 1):]) if n > 1 else tuple()
        banned: Set[int] = index.get(prefix, set())
        if banned:
            banned_idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
            logits_view.index_fill_(0, banned_idx, float("-inf"))

    return logits


def is_inside_docstring(tokenizer, input_ids: torch.Tensor) -> bool:
    """
    Heuristic: we're inside a triple-quoted docstring if the decoded text has an odd
    number of triple quotes (either \"\"\" or ''').
    """
    try:
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    except Exception:
        return False
    dq = text.count('\"\"\"')
    sq = text.count("'''")
    return (dq % 2 == 1) or (sq % 2 == 1)


def is_inside_comment(tokenizer, input_ids: torch.Tensor) -> bool:
    """
    Simple heuristic treating the current line as a comment if it starts with '#'.
    """
    try:
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    except Exception:
        return False
    lines = text.split("\n")
    if not lines:
        return False
    last_line = lines[-1]
    stripped = last_line.lstrip()
    return stripped.startswith("#")


@torch.no_grad()
def incremental_generate(
    model: BLTAdapterModel,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 128,
    patcher: str = "learned",  # none|heuristic|entropy|learned
    entropy_threshold: float = 4.0,
    max_patch_len: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    boundary_threshold: float = 0.7,
    min_steps_between_patches: int = 8,
    disable_patching_in_docstring: bool = True,
    collect_stats: bool = False,
    use_local_decoder: bool = True,  # Enable local decoder for span refinement
    local_decoder_mode: str = "refine",  # "generate" = from scratch, "refine" = teacher-forced denoise using current span tokens
    disable_local_encoder_only: bool = False,  # keep local decoder but skip local encoder span_memory
    min_rewrite_span_len: int = 8,  # minimum span length to actually apply a rewrite (still finalizes boundary either way)
) -> str:
    # `@torch.no_grad()` disables autograd, but `inference_mode()` can be slightly faster
    # (skips version counter updates). Keep it local to this function.
    with torch.inference_mode():
        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=4096,
        )
        device = next(model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
        prompt_len = int(input_ids.size(1))  # never rewrite prompt tokens

        # Helper to append tokens
        def append_tokens(toks: List[int]):
            nonlocal input_ids, attention_mask
            add = torch.tensor([toks], device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, add], dim=1)
            add_mask = torch.ones_like(add)
            attention_mask = torch.cat([attention_mask, add_mask], dim=1)

        # Simple sampler
        def sample_from_logits(logits: torch.Tensor) -> int:
            if temperature > 0.0:
                logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = (cumprobs > top_p).nonzero(as_tuple=False)
                if cutoff.numel() > 0:
                    last_idx = cutoff[0, 1]
                    sorted_logits = sorted_logits[:, :last_idx+1]
                    sorted_indices = sorted_indices[:, :last_idx+1]
                    probs = F.softmax(sorted_logits, dim=-1)
                    sampled_idx = torch.multinomial(probs, num_samples=1)
                    return int(sorted_indices.gather(1, sampled_idx).item())
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())

        # Build banned token ids (comments/docstrings) once per tokenizer (cached)
        banned_ids = _get_banned_token_ids_for_textual_guard(tokenizer)

        new_tokens = 0
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else getattr(model.config, "eos_token_id", None)
        steps_since_patch = 1_000_000
        node_buffer: List[int] = []
        current_node_start_idx = 0

        finalized_any = False
        boundary_triggered = False

        past_key_values = None
        global_hidden_prefix: Optional[torch.Tensor] = None  # [1, L, H]
        global_hidden_new: List[torch.Tensor] = []  # list of [1,1,H]
        last_hidden: Optional[torch.Tensor] = None  # [1,H]
        logits: Optional[torch.Tensor] = None  # [1,V]
        hidden_states_list_lastpos: Optional[List[torch.Tensor]] = None  # each [1,1,H]

        def _full_recompute_states(want_hidden_states: bool = True):
            nonlocal past_key_values, global_hidden_prefix, global_hidden_new, last_hidden, logits, hidden_states_list_lastpos
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=bool(want_hidden_states),
                use_cache=True,
                return_dict=True,
            )
            past_key_values = getattr(out, "past_key_values", None)
            hs_list = list(out.hidden_states) if getattr(out, "hidden_states", None) is not None else None
            if hs_list is not None and len(hs_list) > 0:
                global_hidden_prefix = hs_list[-1]  # [1,L,H]
                global_hidden_new = []
                last_hidden = global_hidden_prefix[:, -1, :]  # [1,H]
                # Keep last-position hidden-states-per-layer for boundary head (all layers, last position only)
                hidden_states_list_lastpos = [h[:, -1:, :] for h in hs_list]
            else:
                # Fallback: some models may not return hidden states if disabled
                global_hidden_prefix = None
                global_hidden_new = []
                last_hidden = None
                hidden_states_list_lastpos = None
            logits = out.logits[:, -1, :]  # [1,V]

        def _get_global_hidden_seq() -> Optional[torch.Tensor]:
            if global_hidden_prefix is None:
                return None
            if not global_hidden_new:
                return global_hidden_prefix
            return torch.cat([global_hidden_prefix] + global_hidden_new, dim=1)

        def _forward_one_token_and_update():
            """
            Advance the global transformer by exactly one token using KV cache.
            Assumes `input_ids` already includes the appended token.
            """
            nonlocal past_key_values, last_hidden, logits, hidden_states_list_lastpos, global_hidden_new
            if past_key_values is None:
                _full_recompute_states(want_hidden_states=True)
                return
            out = model(
                input_ids=input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = getattr(out, "past_key_values", past_key_values)
            hs_list = list(out.hidden_states) if getattr(out, "hidden_states", None) is not None else None
            if hs_list is not None and len(hs_list) > 0:
                # Each layer hidden state is [1,1,H] because we fed a single token with cache.
                last_hidden = hs_list[-1][:, -1, :]  # [1,H]
                hidden_states_list_lastpos = [h[:, -1:, :] for h in hs_list]
                global_hidden_new.append(hs_list[-1])
            logits = out.logits[:, -1, :]  # [1,V]

        def finalize_completed_node(global_hidden_seq: torch.Tensor, include_last_token: bool = False):
            nonlocal input_ids, attention_mask, node_buffer, current_node_start_idx, new_tokens, finalized_any
            span_len = len(node_buffer) if include_last_token else len(node_buffer) - 1
            if span_len <= 0 or current_node_start_idx + span_len > input_ids.size(1):
                if not include_last_token and node_buffer:
                    node_buffer[:] = node_buffer[-1:]
                    current_node_start_idx = input_ids.size(1) - len(node_buffer)
                else:
                    node_buffer.clear()
                    current_node_start_idx = input_ids.size(1)
                return
            if span_len > max_patch_len:
                current_node_start_idx = current_node_start_idx + span_len
                node_buffer.clear()
                return
            start = current_node_start_idx
            end = start + span_len
            span_ids = input_ids[:, start:end]

            # Safety: do not rewrite the prompt portion; only rewrite generated completion.
            # Also avoid rewriting very small spans which tend to be unstable.
            allow_rewrite = bool(use_local_decoder) and (start >= prompt_len) and (span_len >= int(min_rewrite_span_len))

            if allow_rewrite:
                # === LOCAL DECODER path ===
                span_memory = None
                span_mask = None
                if not bool(disable_local_encoder_only):
                    try:
                        span_memory = model.node_token_encoder(span_ids)
                    except Exception:
                        span_memory = None
                    if span_memory is not None:
                        span_mask = torch.zeros(span_memory.size(1), dtype=torch.bool, device=device)

                global_hidden_at_start = global_hidden_seq[0, start, :]  # [H]
                span_latent = model.latent_from_global(global_hidden_at_start.unsqueeze(0)).squeeze(0)  # [H]

                global_memory = global_hidden_seq[:, :end, :]  # [1, end, H]
                global_hidden_last = global_hidden_seq[0, end - 1, :]  # [H]
                global_kpm = None
                try:
                    global_kpm = ~attention_mask[:, :end].squeeze(0).to(torch.bool)
                except Exception:
                    global_kpm = None

                mode = str(local_decoder_mode or "refine").lower()
                decoded_ids = None
                if mode == "refine":
                    try:
                        bos = int(getattr(model, "node_bos_id", getattr(model.config, "bos_token_id", 0)))
                    except Exception:
                        bos = 0
                    prefix = span_ids[0].to(device=device, dtype=torch.long)  # [span_len]
                    if int(prefix.numel()) != int(span_len):
                        decoded_ids = prefix
                    else:
                        if span_len == 1:
                            inp_ids = torch.tensor([bos], device=device, dtype=torch.long).unsqueeze(0)  # [1,1]
                        else:
                            inp_ids = torch.cat(
                                [
                                    torch.tensor([bos], device=device, dtype=torch.long),
                                    prefix[:-1],
                                ],
                                dim=0,
                            ).unsqueeze(0)  # [1, span_len]
                        x = model.local_token_embed(inp_ids)  # [1, span_len, H]
                        cond = model.latent_proj(span_latent).unsqueeze(0).unsqueeze(1)  # [1,1,H]
                        sm = None
                        if span_memory is not None:
                            sm = span_memory if span_memory.dim() == 3 else span_memory.unsqueeze(0)
                        skpm = None
                        if span_mask is not None:
                            skpm = span_mask if span_mask.dim() == 2 else span_mask.unsqueeze(0)
                        gm = global_memory if global_memory.dim() == 3 else global_memory.unsqueeze(0)
                        gkpm = global_kpm
                        if gkpm is not None and gkpm.dim() == 1:
                            gkpm = gkpm.unsqueeze(0)
                        h = model.local_decoder(
                            x,
                            cond,
                            span_memory=sm,
                            span_key_padding_mask=skpm,
                            global_memory=gm,
                            global_key_padding_mask=gkpm,
                        )  # [1, span_len, H]
                        if hasattr(model, "global_residual_gate") and hasattr(model, "global_residual_scale"):
                            global_expanded = global_hidden_last.unsqueeze(0).unsqueeze(0).expand_as(h)  # [1,T,H]
                            gate_input = torch.cat([h, global_expanded], dim=-1)  # [1,T,2H]
                            gate = model.global_residual_gate(gate_input)  # [1,T,1]
                            h = (1 - gate) * h + gate * model.global_residual_scale * global_expanded
                        logits_local = model.local_out_proj(h)  # [1, span_len, V]
                        decoded_ids = torch.argmax(logits_local, dim=-1).view(-1).to(torch.long)  # [span_len]
                else:
                    decoded_ids = model.generate_node_tokens(
                        span_latent=span_latent,
                        span_memory=span_memory.squeeze(0) if span_memory is not None else None,
                        span_key_padding_mask=span_mask if span_mask is not None else None,
                        global_hidden=global_hidden_last,
                        global_memory=global_memory.squeeze(0),
                        global_key_padding_mask=global_kpm,
                        max_len=int(span_len),
                        prefix_ids=None,
                        num_new_tokens=int(span_len),
                        bos_id=None,
                        eos_id=eos_id,
                    )

                if decoded_ids is not None and decoded_ids.numel() == span_len:
                    new_node_tensor = decoded_ids.view(1, -1).to(device=device, dtype=input_ids.dtype)
                else:
                    new_node_tensor = span_ids.to(device=device, dtype=input_ids.dtype)
            else:
                new_node_tensor = span_ids.to(device=device, dtype=input_ids.dtype)

            before = input_ids[:, :start]
            after = input_ids[:, end:]
            input_ids = torch.cat([before, new_node_tensor, after], dim=1)
            attention_mask = torch.ones_like(input_ids)
            length_delta = new_node_tensor.size(1) - span_len
            if length_delta > 0:
                new_tokens += length_delta
            if include_last_token:
                node_buffer.clear()
                current_node_start_idx = before.size(1) + new_node_tensor.size(1)
            else:
                node_buffer[:] = node_buffer[-1:]
                current_node_start_idx = before.size(1) + new_node_tensor.size(1)
            finalized_any = True
            boundary_triggered = True

            # After any rewrite, KV cache and stored hidden prefixes are invalid.
            _full_recompute_states(want_hidden_states=True)

        # Initial full forward on the prompt to seed logits + cache.
        _full_recompute_states(want_hidden_states=True)
        fired_boundaries = 0
        total_tokens = 0
        boundary_events = []  # Per-boundary trigger events for debugging/analysis

        while new_tokens < max_new_tokens:
            if logits is None:
                _full_recompute_states(want_hidden_states=True)
                if logits is None:
                    break

            boundary_confidence = None
            entropy_score = None
            if patcher == "entropy":
                entropy_score = compute_entropy(logits[0])
            if patcher == "learned":
                # Use same boundary feature path as training if available (mid+last concatenation).
                try:
                    last_seq = None
                    if last_hidden is not None:
                        last_seq = last_hidden.unsqueeze(1)  # [1,1,H]
                    if last_seq is not None:
                        boundary_logits = model.compute_boundary_logits(
                            last_hidden=last_seq,
                            hidden_states=hidden_states_list_lastpos,
                        )[:, 0, :]
                    else:
                        boundary_logits = model.boundary_head(last_hidden)  # type: ignore[arg-type]
                    probs = torch.softmax(boundary_logits, dim=-1)
                    boundary_confidence = float(probs[0, 1].item())
                except Exception:
                    boundary_confidence = None

            if banned_ids:
                logits[:, banned_ids] = float("-inf")

            logits = _apply_repetition_penalty_and_ngram_blocking(
                logits=logits,
                generated_ids=input_ids,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            if temperature <= 0.0 and top_p >= 1.0:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                next_id = sample_from_logits(logits)
            append_tokens([next_id])
            node_buffer.append(next_id)
            current_node_start_idx = input_ids.size(1) - len(node_buffer)
            new_tokens += 1
            steps_since_patch += 1
            total_tokens += 1

            # Advance cache/logits for the next step (cheap 1-token forward).
            _forward_one_token_and_update()

            # Only compute docstring/comment guard when we are actually considering a patch.
            textual_guard = False
            if disable_patching_in_docstring:
                # Learned patcher only matters when threshold is met; heuristic/entropy only when boundary condition is met.
                need_guard = False
                if patcher == "learned" and boundary_confidence is not None:
                    need_guard = (boundary_confidence >= boundary_threshold) and (steps_since_patch >= int(min_steps_between_patches))
                elif patcher == "heuristic":
                    need_guard = is_boundary_heuristic(tokenizer, next_id)
                elif patcher == "entropy" and entropy_score is not None:
                    need_guard = entropy_score > entropy_threshold
                if need_guard:
                    try:
                        inside_doc = is_inside_docstring(tokenizer, input_ids)
                    except Exception:
                        inside_doc = False
                    try:
                        inside_comment = is_inside_comment(tokenizer, input_ids)
                    except Exception:
                        inside_comment = False
                    textual_guard = inside_doc or inside_comment

            if eos_id is not None and next_id == eos_id:
                if not textual_guard and finalized_any:
                    # Ensure we have full hidden seq for finalize.
                    _full_recompute_states(want_hidden_states=True)
                    ghs = _get_global_hidden_seq()
                    if ghs is not None:
                        finalize_completed_node(ghs, include_last_token=True)
                node_buffer.clear()
                current_node_start_idx = input_ids.size(1)
                break

            is_new_node = False
            if not textual_guard:
                if patcher == "heuristic":
                    is_new_node = is_boundary_heuristic(tokenizer, next_id)
                elif patcher == "entropy" and entropy_score is not None:
                    is_new_node = entropy_score > entropy_threshold
                elif patcher == "learned" and boundary_confidence is not None:
                    is_new_node = (boundary_confidence >= boundary_threshold) and (steps_since_patch >= int(min_steps_between_patches))

            if is_new_node:
                # Need full hidden seq to perform finalize/rewrite.
                _full_recompute_states(want_hidden_states=True)
                ghs = _get_global_hidden_seq()
                if ghs is not None:
                    try:
                        tok_txt = tokenizer.decode([next_id], skip_special_tokens=True)
                    except Exception:
                        tok_txt = ""
                    boundary_events.append(
                        {
                            "pos": int(input_ids.size(1) - 1),
                            "token_id": int(next_id),
                            "token_text": tok_txt,
                            "boundary_confidence": (float(boundary_confidence) if boundary_confidence is not None else None),
                            "steps_since_patch": int(steps_since_patch),
                        }
                    )
                    finalize_completed_node(ghs, include_last_token=False)
                steps_since_patch = 0
                fired_boundaries += 1
                continue

            if textual_guard:
                node_buffer.clear()
                current_node_start_idx = input_ids.size(1)
                continue

        if node_buffer:
            guard_final = False
            if disable_patching_in_docstring:
                guard_final = is_inside_docstring(tokenizer, input_ids) or is_inside_comment(tokenizer, input_ids)
            if not guard_final and finalized_any and boundary_triggered:
                _full_recompute_states(want_hidden_states=True)
                ghs = _get_global_hidden_seq()
                if ghs is not None:
                    finalize_completed_node(ghs, include_last_token=True)
            else:
                node_buffer.clear()
                current_node_start_idx = input_ids.size(1)
        output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if collect_stats:
            boundary_rate = (float(fired_boundaries) / float(total_tokens)) if total_tokens > 0 else 0.0
            return output, {
                "fired_boundaries": int(fired_boundaries),
                "total_tokens": int(total_tokens),
                "boundary_rate": float(boundary_rate),
                "boundary_trigger_token_ids": [int(e.get("token_id", -1)) for e in boundary_events],
                "boundary_trigger_tokens": [str(e.get("token_text", "")) for e in boundary_events],
                "boundary_trigger_confidences": [e.get("boundary_confidence", None) for e in boundary_events],
                "boundary_events": boundary_events,
            }
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLT Adapter Inference with patchers")
    parser.add_argument("--checkpoint", type=str, default="/data/home/zhangsj/AST_decoding/checkpoints/blt_adapter/focused_sep_embedding_global_kv_residual_LM_NTP/epoch_10", help="Path to saved adapter checkpoint (optional)")
    parser.add_argument("--model_path", type=str, default="/data/home/zhangsj/AST_decoding", help="Base Qwen2.5 path if no checkpoint provided")
    parser.add_argument("--input_file", type=str, required=True, help="Source code file (python)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--patcher", type=str, default="learned", choices=["none", "global_only", "heuristic", "entropy", "learned"])
    parser.add_argument("--entropy_threshold", type=float, default=4.0)
    parser.add_argument("--max_patch_len", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help=">1.0 discourages reusing seen tokens")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Block repeating n-grams of this size (0=disabled)")
    parser.add_argument("--peft_adapter", type=str, default="", help="Path to PEFT LoRA adapter directory (optional)")
    parser.add_argument("--boundary_threshold", type=float, default=0.7, help="Probability threshold for learned patcher to trigger a patch")
    parser.add_argument("--min_steps_between_patches", type=int, default=1, help="Minimum global steps between two patches for learned patcher")
    parser.add_argument("--disable_patching_in_docstring", action="store_true", help="If set, prevents patching while inside triple-quoted docstrings")
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    dtype = select_dtype(device, args.dtype)

    adapter, tokenizer = load_adapter_and_tokenizer(
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        peft_adapter=args.peft_adapter if args.peft_adapter else None
    )

    with open(args.input_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    output = incremental_generate(
        model=adapter,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=args.max_new_tokens,
        patcher=args.patcher,
        entropy_threshold=args.entropy_threshold,
        max_patch_len=args.max_patch_len,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        boundary_threshold=args.boundary_threshold,
        min_steps_between_patches=args.min_steps_between_patches,
        disable_patching_in_docstring=args.disable_patching_in_docstring
    )
    print(output)


if __name__ == "__main__":
    main()


