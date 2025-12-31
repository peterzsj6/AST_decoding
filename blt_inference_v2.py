import os
import sys
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F

# Make project root importable
PROJECT_ROOT = "/data/home/zhangsj"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer  # noqa: E402
from AST_decoding.blt_adapter_model_v2 import BLTAdapterModel  # noqa: E402


def select_device(preferred_device: str = "auto") -> str:
    if preferred_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred_device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred_device == "cpu":
        return "cpu"
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


def load_adapter_and_tokenizer(checkpoint_path: str, model_path: str, device: str, dtype: torch.dtype, peft_adapter: Optional[str] = None, low_cpu_mem_usage: bool = False) -> Tuple[BLTAdapterModel, Any]:
    """Load v2 BLTAdapterModel checkpoint + tokenizer."""
    # NOTE: peft_adapter is accepted for CLI parity but is currently ignored for v2.
    if os.path.isdir(checkpoint_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = BLTAdapterModel.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,  # forwarded to base model
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = BLTAdapterModel.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=low_cpu_mem_usage)

    if getattr(tokenizer, "pad_token", None) is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    # Clear GPU cache before moving model to GPU to reduce fragmentation
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Move model to device and dtype
    # Use no_sync to avoid unnecessary synchronization overhead
    model = model.to(device=device, dtype=dtype)
    
    # Clear cache again after loading to free any temporary allocations
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    return model, tokenizer


def _build_no_repeat_ngram_index(sequence: List[int], n: int) -> DefaultDict[Tuple[int, ...], Set[int]]:
    index: DefaultDict[Tuple[int, ...], Set[int]] = defaultdict(set)
    if n <= 0 or len(sequence) < n:
        return index
    for i in range(len(sequence) - n + 1):
        prefix = tuple(sequence[i : i + n - 1])
        next_tok = sequence[i + n - 1]
        index[prefix].add(next_tok)
    return index


def _apply_repetition_penalty_and_ngram_blocking(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    if logits.dim() != 2 or logits.size(0) != 1:
        return logits
    if generated_ids is None or generated_ids.numel() == 0:
        return logits

    logits_view = logits[0]

    if repetition_penalty and repetition_penalty > 1.0:
        unique_ids = set(int(t) for t in generated_ids[0].tolist())
        if unique_ids:
            idx = torch.tensor(list(unique_ids), device=logits.device, dtype=torch.long)
            token_logits = logits_view.index_select(0, idx)
            neg_mask = token_logits < 0
            token_logits[neg_mask] = token_logits[neg_mask] * repetition_penalty
            token_logits[~neg_mask] = token_logits[~neg_mask] / repetition_penalty
            logits_view.scatter_(0, idx, token_logits)

    if no_repeat_ngram_size and no_repeat_ngram_size > 0 and generated_ids.size(1) >= no_repeat_ngram_size - 1:
        seq_list = generated_ids[0].tolist()
        n = int(no_repeat_ngram_size)
        index = _build_no_repeat_ngram_index(seq_list, n)
        prefix = tuple(seq_list[-(n - 1) :]) if n > 1 else tuple()
        banned: Set[int] = index.get(prefix, set())
        if banned:
            banned_idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
            logits_view.index_fill_(0, banned_idx, float("-inf"))

    return logits


@torch.no_grad()
def incremental_generate(
    model: BLTAdapterModel,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 512,
    patcher: str = "learned",  # none|learned
    boundary_threshold: float = 0.65,
    min_steps_between_patches: int = 4,
    max_patch_len: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    disable_patching_in_docstring: bool = True,  # unused for now
    collect_stats: bool = True,
    use_local_decoder: bool = True,
    local_decoder_mode: str = "generate",  # generate|refine (refine not fully supported in v2)
    disable_local_encoder_only: bool = True,
    min_rewrite_span_len: int = 8,
) -> Tuple[str, Dict[str, Any]]:
    """Incremental generation with optional learned boundary patching for v2 model."""
    
    # For code completion, use direct prompt (no chat template)
    # Chat templates are typically for conversational tasks, not code completion
    # The prompt from EvalPlus is already formatted for direct completion
    
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,  # Don't add special tokens for code completion
        truncation=True,
        max_length=4096,
    )
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    prompt_len = int(input_ids.size(1))

    def _sample_from_logits(logits_1v: torch.Tensor) -> int:
        logits = logits_1v
        if temperature > 0.0:
            logits = logits / float(temperature)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = (cumprobs > float(top_p)).nonzero(as_tuple=False)
            if cutoff.numel() > 0:
                last_idx = int(cutoff[0, 1].item())
                sorted_logits = sorted_logits[:, : last_idx + 1]
                sorted_indices = sorted_indices[:, : last_idx + 1]
                probs = F.softmax(sorted_logits, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1)
                return int(sorted_indices.gather(1, sampled_idx).item())
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    fired_boundaries = 0
    total_tokens = 0
    steps_since_patch = 1_000_000

    node_buffer: List[int] = []
    current_node_start_idx = int(input_ids.size(1))

    boundary_events: List[Dict[str, Any]] = []
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else getattr(model.config, "eos_token_id", None)
    
    # Repetition detection: track recent tokens to detect loops
    recent_tokens: List[int] = []
    max_recent_window = 50  # Check last 50 tokens for repetition
    repetition_threshold = 0.8  # If 80% of recent tokens are repeats, stop

    for _ in range(int(max_new_tokens)):
        out = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]  # [1,V]
        # Only apply repetition penalty to generated tokens (not the prompt)
        generated_ids_only = input_ids[:, prompt_len:] if input_ids.size(1) > prompt_len else input_ids
        logits = _apply_repetition_penalty_and_ngram_blocking(
            logits,
            generated_ids_only,
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )

        next_id = int(torch.argmax(logits, dim=-1).item()) if float(temperature) == 0.0 else _sample_from_logits(logits)

        add = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, add], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(add)], dim=1)

        total_tokens += 1
        if input_ids.size(1) > prompt_len:
            node_buffer.append(next_id)
        steps_since_patch += 1
        
        # Repetition detection: check if we're stuck in a loop
        if total_tokens > 20:  # Only check after generating some tokens
            recent_tokens.append(next_id)
            if len(recent_tokens) > max_recent_window:
                recent_tokens.pop(0)
            
            # Simple repetition check: decode recent tokens and look for repeated phrases
            if len(recent_tokens) >= 30 and total_tokens % 10 == 0:  # Check every 10 tokens after 30
                recent_text = tokenizer.decode(recent_tokens[-30:], skip_special_tokens=False)
                # Check for obvious repetition patterns (same line/phrase repeating)
                lines = recent_text.split('\n')
                if len(lines) >= 3:
                    # Check if last few lines are identical
                    if len(set(lines[-3:])) == 1 and len(lines[-1].strip()) > 10:
                        print(f"[WARNING] Detected repetition loop (same line repeating), stopping generation early (token {total_tokens})")
                        break
                # Check for repeated words/phrases
                words = recent_text.split()
                if len(words) >= 10:
                    # Check if we see the same 5-word sequence multiple times
                    for i in range(len(words) - 5):
                        phrase = ' '.join(words[i:i+5])
                        if recent_text.count(phrase) >= 3:  # Same phrase appears 3+ times
                            print(f"[WARNING] Detected repetition loop (repeated phrase), stopping generation early (token {total_tokens})")
                            break
                    else:
                        continue
                    break

        if eos_id is not None and next_id == int(eos_id):
            break
        
        # Additional stopping criteria for code completion: stop if we see triple backticks (markdown code blocks)
        # This helps prevent the model from generating explanations after code
        if total_tokens > 10:  # Only check after generating some tokens
            recent_text = tokenizer.decode(input_ids[0, max(0, input_ids.size(1) - 20):], skip_special_tokens=False)
            if "```" in recent_text and recent_text.count("```") >= 2:
                # Found closing code block, likely done with code generation
                break

        if patcher == "learned" and use_local_decoder and steps_since_patch >= int(min_steps_between_patches):
            hs = out.hidden_states[-1]  # [1,L,H] for sequence BEFORE append
            b_logits = model.boundary_head(hs[:, -1, :])
            b_prob = float(torch.softmax(b_logits, dim=-1)[0, 1].item())

            if b_prob >= float(boundary_threshold):
                fired_boundaries += 1
                boundary_events.append({"pos": int(input_ids.size(1) - 1), "token_id": int(next_id), "boundary_prob": float(b_prob)})

                span_len = int(len(node_buffer))
                if span_len >= int(min_rewrite_span_len) and span_len <= int(max_patch_len):
                    # refresh hiddens on current sequence
                    out2 = model.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    hs2 = out2.hidden_states[-1][0]  # [L,H]
                    start = int(current_node_start_idx)
                    end = int(start + span_len)
                    if end <= int(input_ids.size(1)):
                        global_hidden_start = hs2[start]  # [H]
                        pred = model.latent_from_global(global_hidden_start.unsqueeze(0)).squeeze(0)
                        span_latent = model.latent_proj(pred)

                        gmem = hs2
                        gkpm = ~attention_mask[0].to(torch.bool)

                        # v2 only supports a generation-style rewrite here (fixed length)
                        new_tokens = model.generate_node_tokens(
                            span_latent=span_latent,
                            span_memory=None,
                            span_key_padding_mask=None,
                            global_hidden=global_hidden_start,
                            global_memory=gmem,
                            global_key_padding_mask=gkpm,
                            max_len=int(max_patch_len),
                            prefix_ids=None,
                            num_new_tokens=int(span_len),
                        )
                        new_ids = [int(t) for t in new_tokens.tolist()]
                        if len(new_ids) < span_len:
                            new_ids = new_ids + node_buffer[len(new_ids) :]
                        new_ids = new_ids[:span_len]

                        input_ids[0, start:end] = torch.tensor(new_ids, device=device, dtype=input_ids.dtype)

                        node_buffer.clear()
                        current_node_start_idx = int(input_ids.size(1))
                        steps_since_patch = 0

    # Decode the full sequence (prompt + completion)
    # EvalPlus expects the full code in the "solution" field (see evaluate.py line 220-224)
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    stats: Dict[str, Any] = {
        "fired_boundaries": int(fired_boundaries),
        "total_tokens": int(total_tokens),
        "boundary_rate": float(fired_boundaries / max(1, total_tokens)),
        "boundary_events": boundary_events,
    }
    return text, stats
