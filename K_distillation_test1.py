import os
# If running single-process (no DDP), you may set a specific GPU here.
# When using torchrun with LOCAL_RANK, prefer setting CUDA_VISIBLE_DEVICES in the shell.
if 'LOCAL_RANK' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'





from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import Qwen2ForCausalLM, Qwen2Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from ast_parsing_folder.AST_parsing import parse_to_ast,get_ast_leaf_nodes_for_spans
from typing import List, Dict, Optional, Tuple
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


model_path = "/data/home/zhangsj/AST_decoding"
config = Qwen2Config.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
embedding_approach = "mean_pooling"
trail_name="11_19_unfreez_2_LM_layers"
training_languages = ["python"]#, "cpp", "java", "javascript", "php", "rust



# Local decoder configuration (BLT-style node latent + local token decoder)
use_local_decoder = True
local_decoder_type = "transformer"  # Options: "gru", "transformer"
lm_loss_weight = 1.0
node_reconstruction_loss_weight = 1.0
node_type_cls_loss_weight = 0.2
node_length_cls_loss_weight = 0.1
max_node_length = 32
# PEFT LoRA flags (applied via peft to specific custom modules)
use_peft_lora_local_decoder = False
use_peft_lora_span_embeddings = False
# PEFT LoRA config for custom modules
peft_lora_rank = 8
peft_lora_alpha = 16
peft_lora_dropout = 0.05
# DDP configuration (no KD/student-teacher)
use_ddp = False  # set True to enable DistributedDataParallel
# Configuration options for span-aware embeddings:
# - embedding_approach: 
#   * "placeholder": Standard embeddings only
#   * "multi_component": Additive combination of token + span type + position + boundary embeddings
#   * "mean_pooling": Mean pooling within spans + weighted combination with token embeddings
#   * "mean_pooling_only": Mean pooling within spans only (no weighted combination)

class ASTSpanDataset(Dataset):
    """
    Dataset class for loading parquet files with AST span information.
    Handles both content and AST_span columns from the processed parquet files.
    """
    
    def __init__(self, parquet_file_path, tokenizer, max_length=1024, language="python"):
        """
        Initialize the dataset.
        
        Args:
            parquet_file_path: Path to the processed parquet file
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
            language: Programming language for the data
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language = language
        
        # Load the parquet file
        print(f"Loading dataset from: {parquet_file_path}")
        self.df = pd.read_parquet(parquet_file_path)
        
        # Filter out rows with errors or empty content
        initial_count = len(self.df)
        
        # Basic content filtering
        content_filter = (self.df['content'].notna()) & (self.df['content'].str.strip() != '')
        
        # Error filtering (only if error column exists)
        if 'error' in self.df.columns:
            error_filter = ~self.df['error'].notna()
            self.df = self.df[content_filter & error_filter]
        else:
            self.df = self.df[content_filter]
        
        # Additional filtering for valid AST spans
        ast_span_filter = (self.df['AST_span'].notna()) & (self.df['AST_span'].str.len() > 2)  # More than just "[]"
        self.df = self.df[ast_span_filter]
        
        print(f"Dataset loaded: {len(self.df)} valid samples (filtered from {initial_count})")
        
        # Span type mapping (consistent with embedding layers)
        self.span_type_vocab = {
                'unknown': 0, 'keyword': 1, 'identifier': 2, 'string': 3,
                'number': 4, 'comment': 5, 'operator': 6, 'punctuation': 7,
                'module': 8, 'class': 9, 'function': 10, 'text': 11, 
                '(': 12, ')': 13, '$': 14, '@': 15,
                '=': 16, 'expression_statement': 17, 'type_identifier': 18,
                'field_identifier': 19, '::': 20, 'function_definition': 21,
                'block': 22, '"': 23, 'program': 24, 'property_identifier': 25,
                "'": 26, 'string_start': 27, 'call_expression': 28,
                'namespace_identifier': 29, 'class_declaration': 30,
                'number_literal': 31, 'string_fragment': 32, 'if_statement': 33,
                'primitive_type': 34, 'string_end': 35, 'const': 36,
                'namespace_definition': 37, 'impl_item': 38, 'source_file': 39,
                'translation_unit': 40, 'integer_literal': 41, 'let': 42,
                'else_clause': 43, 'function_item': 44, 'method_declaration': 45,
                'this': 46, 'decorated_definition': 47, 'doc_comment': 48,
                'compound_statement': 49, 'self': 50, 'public': 51,
                'line_comment': 52, 'fn': 53, 'class_definition': 54
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample with both content and AST span metadata.
        
        Returns:
            dict with keys:
                - input_ids: Tokenized input
                - attention_mask: Attention mask
                - span_metadata: AST span information for span-aware models
                - original_content: Original source code
                - language: Programming language
        """
        row = self.df.iloc[idx]
        content = row['content']
        ast_span_json = row['AST_span']
        
        # Tokenize the content
        encoding = self.tokenizer(
            content,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Parse AST spans from JSON
        try:
            ast_spans = json.loads(ast_span_json) if ast_span_json else []
        except (json.JSONDecodeError, TypeError):
            ast_spans = []
        
        # Process span metadata
        span_metadata = self._process_ast_spans(encoding['input_ids'].squeeze(), ast_spans)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'span_metadata': span_metadata,
            'original_content': content,
            'language': self.language,
            'repo_path': row.get('max_stars_repo_path', ''),
            'repo_name': row.get('max_stars_repo_name', ''),
            'stars_count': row.get('max_stars_count', 0),
            'coverage_percentage': row.get('coverage_percentage', 0)
        }
    
    def _process_ast_spans(self, input_ids, ast_spans):
        """
        Convert AST spans to metadata tensors for the model.
        
        Args:
            input_ids: Tokenized input tensor [seq_len]
            ast_spans: List of AST span dictionaries
            
        Returns:
            Dictionary with span metadata tensors
        """
        seq_len = len(input_ids)
        
        # Initialize metadata arrays as numpy arrays
        span_types = np.zeros(seq_len, dtype=np.int64)  # 0 = unknown
        positions = np.zeros(seq_len, dtype=np.int64)   # position within span
        boundaries = np.zeros(seq_len, dtype=np.int64)  # 0 = middle, 1 = start, 2 = end, 3 = single
        
        # Process each AST span
        processed_spans = []
        for span in ast_spans:
            if not isinstance(span, dict):
                continue
                
            token_indices = span.get('token_indices', [])
            span_type = span.get('type', 'unknown')
            
            if not token_indices:
                continue
            
            # Convert token_indices to numpy array and filter valid indices
            token_indices = np.array(token_indices, dtype=np.int64)
            valid_mask = (token_indices >= 0) & (token_indices < seq_len)
            valid_indices = token_indices[valid_mask]
            
            if len(valid_indices) == 0:
                continue
            
            # Map span type to ID
            span_type_id = self.span_type_vocab.get(span_type, 0)
            
            # Fill metadata for each token in span
            for pos, token_idx in enumerate(valid_indices):
                span_types[token_idx] = span_type_id
                positions[token_idx] = min(pos, 31)  # Cap at 31 (max 32 positions)
                
                # Determine boundary type
                if len(valid_indices) == 1:
                    boundaries[token_idx] = 3  # single token span
                elif pos == 0:
                    boundaries[token_idx] = 1  # start
                elif pos == len(valid_indices) - 1:
                    boundaries[token_idx] = 2  # end
                else:
                    boundaries[token_idx] = 0  # middle
            
            # Store processed span for mean pooling approach with numpy arrays
            processed_spans.append({
                'token_indices': valid_indices,
                'type': span_type,
                'span_type_id': span_type_id
            })
        
        return {
            'span_types': torch.tensor(span_types, dtype=torch.long),
            'positions': torch.tensor(positions, dtype=torch.long),
            'boundaries': torch.tensor(boundaries, dtype=torch.long),
            'raw_spans': processed_spans  # For mean pooling approach
        }


def create_dataloader(parquet_files, tokenizer, batch_size=4, max_length=1024, 
                     language="python", num_workers=0, shuffle=True):
    """
    Create a DataLoader from multiple parquet files.
    
    Args:
        parquet_files: List of parquet file paths or single path
        tokenizer: Tokenizer to use
        batch_size: Batch size for training
        max_length: Maximum sequence length
        language: Programming language
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    if isinstance(parquet_files, str):
        parquet_files = [parquet_files]
    
    # Combine multiple parquet files if needed
    all_datasets = []
    for parquet_file in parquet_files:
        if os.path.exists(parquet_file):
            dataset = ASTSpanDataset(parquet_file, tokenizer, max_length, language)
            all_datasets.append(dataset)
            print(f"Loaded dataset from {parquet_file}: {len(dataset)} samples")
        else:
            print(f"Warning: File not found: {parquet_file}")
    
    if not all_datasets:
        raise ValueError("No valid parquet files found!")
    
    # Combine datasets if multiple files
    if len(all_datasets) == 1:
        combined_dataset = all_datasets[0]
    else:
        # Concatenate datasets
        combined_df = pd.concat([ds.df for ds in all_datasets], ignore_index=True)
        combined_dataset = ASTSpanDataset.__new__(ASTSpanDataset)
        combined_dataset.tokenizer = tokenizer
        combined_dataset.max_length = max_length
        combined_dataset.language = language
        combined_dataset.df = combined_df
        combined_dataset.span_type_vocab = all_datasets[0].span_type_vocab
        print(f"Combined dataset: {len(combined_dataset)} total samples")
    
    # Check if dataset is empty
    if len(combined_dataset) == 0:
        raise ValueError("Dataset is empty after filtering! Check your data and filtering criteria.")
    
    # Create DataLoader
    def collate_fn(batch):
        """Custom collate function to handle span metadata."""
        # Stack regular tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Collect span metadata
        batch_span_metadata = {
            'span_types': torch.stack([item['span_metadata']['span_types'] for item in batch]),
            'positions': torch.stack([item['span_metadata']['positions'] for item in batch]),
            'boundaries': torch.stack([item['span_metadata']['boundaries'] for item in batch]),
            'raw_spans': [item['span_metadata']['raw_spans'] for item in batch]
        }
        
        # Collect other metadata
        batch_metadata = {
            'original_content': [item['original_content'] for item in batch],
            'language': [item['language'] for item in batch],
            'repo_path': [item['repo_path'] for item in batch],
            'repo_name': [item['repo_name'] for item in batch],
            'stars_count': [item['stars_count'] for item in batch],
            'coverage_percentage': [item['coverage_percentage'] for item in batch]
        }
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'span_metadata': batch_span_metadata,
            'metadata': batch_metadata
        }
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def load_language_datasets(base_path="/data/home/zhangsj/Data/more_big_code_language", 
                          languages=None, tokenizer=None, batch_size=4, max_length=1024):
    """
    Load datasets for multiple programming languages.
    
    Args:
        base_path: Base directory containing language subdirectories
        languages: List of languages to load (default: all available)
        tokenizer: Tokenizer to use
        batch_size: Batch size for DataLoaders
        max_length: Maximum sequence length
        
    Returns:
        Dictionary mapping language names to DataLoader instances
    """
    if languages is None:
        languages = ["python"]
    # available languages "cpp", "java", "javascript", "php", "python", "rust"
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("/data/home/zhangsj/qwen_coder_1.5b")
    
    dataloaders = {}
    
    for language in languages:
        parquet_file = os.path.join(base_path, language, f"{language}_ast_parsed.parquet")
        
        if os.path.exists(parquet_file):
            try:
                dataloader = create_dataloader(
                    parquet_file, 
                    tokenizer, 
                    batch_size=batch_size,
                    max_length=max_length,
                    language=language,
                    shuffle=True
                )
                dataloaders[language] = dataloader
                print(f"✅ Created DataLoader for {language}: {len(dataloader.dataset)} samples")
            except Exception as e:
                print(f"❌ Error creating DataLoader for {language}: {e}")
        else:
            print(f"⚠️  Parquet file not found for {language}: {parquet_file}")
    
    return dataloaders





class MeanPooledSpanEmbeddingLayer(nn.Module):
    """
    ENHANCED: Mean Pooling + Weighted Combination Approach
    
    Strategy:
    1. Get original token embeddings
    2. For each AST span, mean-pool token embeddings to create span embedding
    3. Add span type information to pooled embeddings  
    4. Assign the span embedding to all tokens in that span
    5. Combine: weighted_sum(token_emb, span_emb) with more weight on tokens initially
    
    Benefits:
    - Span embeddings directly derived from token context
    - Easier training (starts closer to original model)
    - Natural regularization through span consistency
    """
    
    def __init__(self, config, span_dropout_prob=0.0):
        super().__init__()
        self.config = config
        
        # Traditional token embeddings  
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        
        # ✨ NEW: Trainable adapter on frozen token embeddings
        # This allows token embeddings to adapt without modifying the base weights
        # Residual design: starts near identity for stable training
        self.token_adapter = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        # Initialize adapter to near-identity (small residual)
        nn.init.normal_(self.token_adapter[0].weight, std=0.02)
        nn.init.zeros_(self.token_adapter[0].bias)
        nn.init.normal_(self.token_adapter[2].weight, std=0.02)
        nn.init.zeros_(self.token_adapter[2].bias)
        
        # Span type embeddings (to enhance pooled representations)
        self.span_type_vocab = {
                'unknown': 0, 'keyword': 1, 'identifier': 2, 'string': 3,
                'number': 4, 'comment': 5, 'operator': 6, 'punctuation': 7,
                'module': 8, 'class': 9, 'function': 10, 'text': 11, 
                '(': 12, ')': 13, '$': 14, '@': 15,
                '=': 16, 'expression_statement': 17, 'type_identifier': 18,
                'field_identifier': 19, '::': 20, 'function_definition': 21,
                'block': 22, '"': 23, 'program': 24, 'property_identifier': 25,
                "'": 26, 'string_start': 27, 'call_expression': 28,
                'namespace_identifier': 29, 'class_declaration': 30,
                'number_literal': 31, 'string_fragment': 32, 'if_statement': 33,
                'primitive_type': 34, 'string_end': 35, 'const': 36,
                'namespace_definition': 37, 'impl_item': 38, 'source_file': 39,
                'translation_unit': 40, 'integer_literal': 41, 'let': 42,
                'else_clause': 43, 'function_item': 44, 'method_declaration': 45,
                'this': 46, 'decorated_definition': 47, 'doc_comment': 48,
                'compound_statement': 49, 'self': 50, 'public': 51,
                'line_comment': 52, 'fn': 53, 'class_definition': 54
        }
        self.span_type_embeddings = nn.Embedding(
            len(self.span_type_vocab), config.hidden_size
        )
        self.num_span_types = len(self.span_type_vocab)
        
        
        # Span enhancement projection (optional refinement of pooled embeddings)
        self.span_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        # Normalize only the span branch; keep token branch raw to preserve base distribution
        self.span_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Span dropout probability (drop span features for full samples during training)
        self.span_dropout_prob = float(span_dropout_prob)

        # Gating network to combine token and span embeddings adaptively
        self.gate_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.gate_activation = nn.Sigmoid()
        
    @property
    def weight(self):
        """
        Expose a weight parameter compatible with HF tie_weights logic.
        Returns the underlying token embedding weight so that
        lm_head.weight can be tied/cloned to it.
        """
        return self.token_embeddings.weight
    
    @property
    def num_embeddings(self):
        return self.token_embeddings.num_embeddings
    
    @property
    def embedding_dim(self):
        return self.token_embeddings.embedding_dim
        
    def forward(self, input_ids, span_metadata=None):
        """
        Enhanced forward with mean pooling over AST spans and token adapter.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            span_metadata: Dict containing:
                - raw_spans: List of span dicts with token_indices and type info
                - span_types: [batch_size, seq_len] (for fallback)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get original token embeddings (frozen)
        token_emb_frozen = self.token_embeddings(input_ids)  # [B, L, H]
        
        # ✨ NEW: Apply trainable adapter with residual connection
        # This allows adaptation while keeping base embeddings frozen
        token_emb_adapted = token_emb_frozen + self.token_adapter(token_emb_frozen)
        
        # Use adapted embeddings for all downstream operations
        token_emb = token_emb_adapted
        
        if span_metadata is None or 'raw_spans' not in span_metadata:
            # Fallback to adapted token embeddings (works for natural language too!)
            combined_emb = self.dropout(token_emb)
            return {
                'embeddings': combined_emb
            }
        
        # Initialize span embeddings as copy of token embeddings
        span_emb = token_emb.clone()  # [B, L, H]
        span_mask = torch.zeros(batch_size, seq_len, device=device, dtype=token_emb.dtype)  # [B, L]
        
        # Process each batch
        for batch_idx in range(batch_size):
            batch_spans = span_metadata.get('raw_spans', [])
            if not batch_spans or batch_idx >= len(batch_spans):
                continue
                
            # Get spans for this specific batch item
            item_spans = batch_spans[batch_idx]
            if not item_spans:
                continue
                
            # Process each span in this batch item
            for span in item_spans:
                if not isinstance(span, dict):
                    continue
                    
                token_indices = span.get('token_indices', [])
                span_type = span.get('type', 'unknown')
                
                # Convert to numpy array if it's not already
                if not isinstance(token_indices, np.ndarray):
                    token_indices = np.array(token_indices, dtype=np.int64)
                
                if len(token_indices) <= 1:
                    continue  # Skip single-token spans
                
                # Filter valid token indices
                valid_mask = (token_indices >= 0) & (token_indices < seq_len)
                valid_indices = token_indices[valid_mask]
                if len(valid_indices) <= 1:
                    continue
                
                # Mean pool token embeddings within this span
                span_token_embs = token_emb[batch_idx, valid_indices]  # [span_len, H]
                pooled_span_emb = span_token_embs.mean(dim=0, keepdim=True)  # [1, H]
                
                # Enhance with span type information
                span_type_id = self.span_type_vocab.get(span_type, 0)
                type_emb = self.span_type_embeddings(
                    torch.tensor(span_type_id, device=device)
                )  # [H]
                
                # Combine pooled embedding with type information
                enhanced_span_emb = pooled_span_emb.squeeze(0) + type_emb  # [H]
                
                # Optional: Project through linear layer for refinement
                enhanced_span_emb = self.span_projection(enhanced_span_emb)  # [H]
                
                # Assign this span embedding to all tokens in the span
                for idx in valid_indices:
                    span_emb[batch_idx, idx] = enhanced_span_emb
                    span_mask[batch_idx, idx] = 1.0
        
        # Optional span dropout per-sample during training
        if self.training and self.span_dropout_prob > 0.0:
            # Drop spans for a random subset of batch items (entire sample)
            drop_vec = (torch.rand(batch_size, device=device) < self.span_dropout_prob).to(span_mask.dtype)  # 1 => drop
            # Apply to mask: if dropped, zero out all span tokens for that sample
            ones_mask = torch.ones_like(span_mask, dtype=span_mask.dtype)
            span_mask = span_mask * (ones_mask - drop_vec.view(-1, 1))

        # Adaptive gating between token_emb and span_emb
        # Compute gate on normalized span branch while keeping token branch raw
        span_emb_norm = self.span_norm(span_emb.to(self.span_norm.weight.dtype)).to(token_emb.dtype)
        gate_input = torch.cat([token_emb, span_emb_norm], dim=-1)  # [B, L, 2H]
        gate = self.gate_activation(self.gate_linear(gate_input)).to(token_emb.dtype)  # [B, L, H] in [0,1]
        # Ensure no-span positions (or dropped) fall back to token-only
        gate = gate * span_mask.unsqueeze(-1)
        # Store for logging
        self.last_gate = gate.detach()
        self.last_span_mask = span_mask.detach()
        # Gating-only combination (no global token/span weights)
        ones_gate = torch.ones_like(gate, dtype=gate.dtype)
        combined_emb = (ones_gate - gate) * token_emb + gate * span_emb
        
        # Return combined without a post-combine normalization to avoid shifting base distribution
        combined_emb = self.dropout(combined_emb)
        
        return {
            'embeddings': combined_emb,
            'gate': gate,
            'span_mask': span_mask
        }

    


class CustomQwen2Model(Qwen2ForCausalLM):
    def __init__(self, config, embedding_type="placeholder"):
        super().__init__(config)
        
        # Force mean-pooling span-aware embedding as the only embedding type
        self.embedding_type = "mean_pooling"
        self.model.embed_tokens = MeanPooledSpanEmbeddingLayer(config)
        print("Using Mean Pooling AST span-aware embedding layer")
        
        # =========================
        # Node-level local decoders
        # =========================
        self.local_decoder_enabled = bool(use_local_decoder)
        self.local_decoder_type = local_decoder_type
        self.node_bos_id = getattr(config, 'bos_token_id', getattr(config, 'eos_token_id', 0))
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_node_length_bins = max_node_length
        
        # Span type vocabulary for node-level supervision (align with embedding layers)
        self.span_type_vocab = {
                'unknown': 0, 'keyword': 1, 'identifier': 2, 'string': 3,
                'number': 4, 'comment': 5, 'operator': 6, 'punctuation': 7,
                'module': 8, 'class': 9, 'function': 10, 'text': 11, 
                '(': 12, ')': 13, '$': 14, '@': 15,
                '=': 16, 'expression_statement': 17, 'type_identifier': 18,
                'field_identifier': 19, '::': 20, 'function_definition': 21,
                'block': 22, '"': 23, 'program': 24, 'property_identifier': 25,
                "'": 26, 'string_start': 27, 'call_expression': 28,
                'namespace_identifier': 29, 'class_declaration': 30,
                'number_literal': 31, 'string_fragment': 32, 'if_statement': 33,
                'primitive_type': 34, 'string_end': 35, 'const': 36,
                'namespace_definition': 37, 'impl_item': 38, 'source_file': 39,
                'translation_unit': 40, 'integer_literal': 41, 'let': 42,
                'else_clause': 43, 'function_item': 44, 'method_declaration': 45,
                'this': 46, 'decorated_definition': 47, 'doc_comment': 48,
                'compound_statement': 49, 'self': 50, 'public': 51,
                'line_comment': 52, 'fn': 53, 'class_definition': 54
        }
        self.num_span_types = len(self.span_type_vocab)
        
        if self.local_decoder_enabled:
            # Node-level inputs and heads
            self.local_token_embed = nn.Embedding(self.vocab_size, self.hidden_size)
            self.latent_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.node_type_head = nn.Linear(self.hidden_size, self.num_span_types)
            self.node_len_head = nn.Linear(self.hidden_size, self.max_node_length_bins)
            
            if self.local_decoder_type == "gru":
                self.local_rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=2, batch_first=True)
                self.local_out_proj = nn.Linear(self.hidden_size, self.vocab_size)
            else:
                # Transformer-style local decoder (causal)
                nhead = max(1, self.hidden_size // 64)
                dim_ff = max(self.hidden_size * 4, 512)
                encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
                self.local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.pos_embed = nn.Embedding(max_node_length + 1, self.hidden_size)
                self.local_out_proj = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, input_ids, attention_mask=None, span_metadata=None, **kwargs):
        """
        Enhanced forward pass that can handle span metadata and span type prediction.
        """
        # If inputs_embeds is provided (e.g., during span-aware inference), bypass custom embedding computation
        if 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
            # Do not attempt to call embed_tokens when inputs_embeds is already given
            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                **kwargs
            )

        # Compute embeddings (handle dict outputs)
        emb_out = self.model.embed_tokens(input_ids, span_metadata)
        if isinstance(emb_out, dict):
            inputs_embeds = emb_out.get('embeddings', None)
        else:
            inputs_embeds = emb_out
        
        # Forward through base model with embeddings
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['input_ids', 'inputs_embeds']}
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **filtered_kwargs
        )
        # Extract labels (if any) for gating node losses to training-only
        labels = kwargs.get('labels', None)
        
        # Node-level local decoder losses (BLT-style) - TRAINING ONLY
        if self.training and self.local_decoder_enabled and (labels is not None) and span_metadata is not None and 'raw_spans' in span_metadata:
            node_losses = self._compute_node_losses(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                span_metadata=span_metadata
            )
            if node_losses is not None:
                node_recon_loss, node_type_loss, node_len_loss, stats = node_losses
                # Combine losses
                total = 0.0
                if outputs.loss is not None:
                    total = lm_loss_weight * outputs.loss
                total = total + node_reconstruction_loss_weight * node_recon_loss
                if node_type_loss is not None:
                    total = total + node_type_cls_loss_weight * node_type_loss
                if node_len_loss is not None:
                    total = total + node_length_cls_loss_weight * node_len_loss
                outputs.loss = total
                # Expose components for logging
                outputs.node_recon_loss = node_recon_loss
                outputs.node_type_loss = node_type_loss
                outputs.node_len_loss = node_len_loss
                outputs.node_stats = stats
        
        return outputs
    
    def _segment_non_overlapping(self, raw_spans: List[Dict], seq_len: int) -> List[Dict]:
        """
        Build non-overlapping, left-to-right node spans from possibly overlapping spans.
        Strategy: sort by (start, -length), greedily select spans that don't intersect.
        """
        spans = []
        for sp in raw_spans:
            # Robustness to malformed span entries (e.g., strings)
            if not isinstance(sp, dict):
                continue
            token_indices = sp.get('token_indices', [])
            if isinstance(token_indices, list):
                token_indices = np.array(token_indices, dtype=np.int64)
            if not isinstance(token_indices, np.ndarray):
                continue
            if len(token_indices) == 0:
                continue
            # sanitize
            token_indices = np.unique(token_indices[(token_indices >= 0) & (token_indices < seq_len)])
            if len(token_indices) == 0:
                continue
            start = int(token_indices.min())
            length = int(len(token_indices))
            spans.append({
                'start': start,
                'length': length,
                'indices': token_indices,
                'span_type_id': int(sp.get('span_type_id', 0))
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
    
    def _compute_node_losses(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        span_metadata: Dict
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict]]:
        """
        Compute node-level reconstruction loss (teacher-forced), node type loss, and node length loss.
        Returns (recon_loss, type_loss, len_loss, stats_dict) or None if no nodes.
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        node_input_seqs = []   # with BOS and shifted
        node_target_seqs = []  # targets
        node_latents = []      # [H]
        node_type_ids = []     # scalar
        node_len_bins = []     # scalar (clipped)
        total_nodes_kept = 0
        total_nodes_skipped = 0
        
        for b in range(batch_size):
            raw_list = span_metadata.get('raw_spans', [])
            if not raw_list or b >= len(raw_list):
                continue
            item_spans = raw_list[b]
            sel = self._segment_non_overlapping(item_spans, seq_len)
            if not sel:
                continue
            for sp in sel:
                idxs = sp['indices']
                if len(idxs) == 0:
                    total_nodes_skipped += 1
                    continue
                # gather token ids for this node
                token_seq = input_ids[b, torch.tensor(idxs, device=device)].detach()
                L = int(token_seq.shape[0])
                if L <= 0:
                    total_nodes_skipped += 1
                    continue
                # latent by mean pooling of embeddings at node positions
                latent = inputs_embeds[b, torch.tensor(idxs, device=inputs_embeds.device), :].mean(dim=0)
                
                # truncate for reconstruction
                Lc = min(L, max_node_length)
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
                node_latents.append(latent)
                node_type_ids.append(int(sp.get('span_type_id', 0)))
                node_len_bins.append(min(L, self.max_node_length_bins - 1))
                total_nodes_kept += 1
        
        if total_nodes_kept == 0:
            return None
        
        # Pad to batch for local decoding
        maxL = max(seq.shape[0] for seq in node_input_seqs)
        inp_batch = torch.full((total_nodes_kept, maxL), fill_value=self.config.pad_token_id if self.config.pad_token_id is not None else 0, dtype=torch.long, device=device)
        tgt_batch = torch.full((total_nodes_kept, maxL), fill_value=-100, dtype=torch.long, device=device)
        mask_batch = torch.zeros((total_nodes_kept, maxL), dtype=torch.bool, device=device)
        for i in range(total_nodes_kept):
            L = node_input_seqs[i].shape[0]
            inp_batch[i, :L] = node_input_seqs[i]
            tgtL = node_target_seqs[i].shape[0]
            tgt_batch[i, :tgtL] = node_target_seqs[i]
            mask_batch[i, :tgtL] = True
        latents_batch = torch.stack(node_latents, dim=0)  # [N, H]
        
        # Local decoder forward
        tok_emb = self.local_token_embed(inp_batch)  # [N, L, H]
        cond = self.latent_proj(latents_batch).unsqueeze(1)  # [N,1,H]
        tok_emb = tok_emb + cond  # simple FiLM-like conditioning
        
        if self.local_decoder_type == "gru":
            # initial hidden from latent
            h0 = self.latent_proj(latents_batch).unsqueeze(0).repeat(2, 1, 1)  # [layers, N, H]
            dec_out, _ = self.local_rnn(tok_emb, h0)  # [N, L, H]
        else:
            # Causal mask
            L = inp_batch.shape[1]
            causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
            # add simple learned positions
            pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(inp_batch.size(0), -1)
            pos_emb = self.pos_embed(torch.clamp(pos_ids, max=self.pos_embed.num_embeddings - 1))
            x = tok_emb + pos_emb
            dec_out = self.local_transformer(x, mask=causal_mask)  # [N, L, H]
        
        logits = self.local_out_proj(dec_out)  # [N, L, V]
        
        # Reconstruction CE with mask
        vocab = logits.size(-1)
        recon_loss = F.cross_entropy(
            logits.view(-1, vocab),
            torch.where(mask_batch.view(-1), tgt_batch.view(-1), torch.full_like(tgt_batch.view(-1), -100)),
            ignore_index=-100
        )
        
        # Node type loss
        type_logits = self.node_type_head(latents_batch)  # [N, C]
        type_targets = torch.tensor(node_type_ids, dtype=torch.long, device=device)
        node_type_loss = F.cross_entropy(type_logits, type_targets) if self.num_span_types > 0 else None
        
        # Node length (binned) loss
        len_logits = self.node_len_head(latents_batch)  # [N, B]
        len_targets = torch.tensor(node_len_bins, dtype=torch.long, device=device)
        node_len_loss = F.cross_entropy(len_logits, len_targets)
        
        stats = {
            'nodes_kept': total_nodes_kept,
            'nodes_skipped': total_nodes_skipped,
            'avg_node_len': float(torch.tensor([t.shape[0] for t in node_target_seqs], dtype=torch.float32).mean().item())
        }
        return recon_loss, node_type_loss, node_len_loss, stats
    
    @torch.no_grad()
    def generate_node_tokens(self, latent: torch.Tensor, max_len: int = 64, bos_id: Optional[int] = None, eos_id: Optional[int] = None) -> torch.Tensor:
        """
        Greedy-generate a node's token sequence from a node latent using the local decoder.
        latent: [H]
        Returns tensor of token ids [L]
        """
        self.eval()
        device = latent.device
        bos = bos_id if bos_id is not None else self.node_bos_id
        eos = eos_id if eos_id is not None else getattr(self.config, 'eos_token_id', None)
        
        tokens = [bos]
        hidden = None
        cond = self.latent_proj(latent).unsqueeze(0)  # [1,H]
        
        if self.local_decoder_type == "gru":
            h = self.latent_proj(latent).unsqueeze(0).repeat(2, 1, 1)  # [layers, 1, H]
            for _ in range(max_len):
                inp = torch.tensor([tokens[-1]], device=device, dtype=torch.long).unsqueeze(0)  # [1,1]
                x = self.local_token_embed(inp) + cond.unsqueeze(1)  # [1,1,H]
                out, h = self.local_rnn(x, h)  # [1,1,H]
                logit = self.local_out_proj(out[:, -1])  # [1,V]
                next_id = int(torch.argmax(logit, dim=-1).item())
                tokens.append(next_id)
                if eos is not None and next_id == eos:
                    break
        else:
            # Transformer incremental generation (re-encode prefix each step)
            for _ in range(max_len):
                inp = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)  # [1,T]
                pos_ids = torch.arange(inp.size(1), device=device).unsqueeze(0)
                x = self.local_token_embed(inp) + self.pos_embed(torch.clamp(pos_ids, max=self.pos_embed.num_embeddings - 1)) + cond.unsqueeze(1)
                L = x.size(1)
                mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
                h = self.local_transformer(x, mask=mask)  # [1,T,H]
                out = h[:, -1, :]  # [1,H]
                logit = self.local_out_proj(out)  # [1,V]
                next_id = int(torch.argmax(logit, dim=-1).item())
                tokens.append(next_id)
                if eos is not None and next_id == eos:
                    break
        
        # drop BOS
        return torch.tensor(tokens[1:], device=device, dtype=torch.long)
    
    @torch.no_grad()
    def decode_node_from_span(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor, span_token_indices: torch.Tensor, max_len: int = 64) -> torch.Tensor:
        """
        Convenience method: compute latent by pooling embeddings on given span indices, then generate node tokens.
        input_ids: [B,L], inputs_embeds: [B,L,H], span_token_indices: [K] (single sample context)
        Returns token ids [<=max_len]
        """
        device = inputs_embeds.device
        if input_ids.size(0) != 1 or inputs_embeds.size(0) != 1:
            raise ValueError("decode_node_from_span expects batch size 1.")
        latent = inputs_embeds[0, span_token_indices.to(device)].mean(dim=0)
        return self.generate_node_tokens(latent, max_len=max_len)


# Step 1: Load the original model (only when run as main)
def create_span_aware_model():
    """Create and initialize the span-aware model."""
    original_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Step 2: Create your custom model with the same config (with span awareness)
    config = original_model.config
    
    custom_model = CustomQwen2Model(config, embedding_type=embedding_approach)
    
    # Step 3: Copy weights from original to custom model (except embedding)
    print("Copying weights...")
    copied_count = 0
    for name, param in original_model.named_parameters():
        if name != "model.embed_tokens.weight":  # Skip the embedding layer - handle separately
            if name in custom_model.state_dict():
                with torch.no_grad():
                    custom_model.state_dict()[name].copy_(param)
                    copied_count += 1
            else:
                print(f"MISSING: {name} not found in custom model")
    
    # Also copy lm_head.weight if it exists in original model
    if "lm_head.weight" in original_model.state_dict():
        if "lm_head.weight" in custom_model.state_dict():
            with torch.no_grad():
                custom_model.state_dict()["lm_head.weight"].copy_(original_model.state_dict()["lm_head.weight"])
                print(f"Copied lm_head.weight")
                copied_count += 1
    
    # Step 4: Handle the embedding layer specially
    original_embedding = original_model.model.embed_tokens.weight
    # Always copy into mean-pooling token embeddings
    with torch.no_grad():
        custom_model.model.embed_tokens.token_embeddings.weight.copy_(original_embedding)
        print(f"Copied token embeddings to mean_pooling layer: {original_embedding.shape}")
    
    # Proactively free the original model before moving custom_model to GPU
    try:
        del original_model
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    
    # Move to CUDA in reduced precision to save memory
    if torch.cuda.is_available():
        try:
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            target_dtype = torch.float16
        device_index = torch.cuda.current_device()
        device = torch.device(f"cuda:{device_index}")
        custom_model = custom_model.to(device=device, dtype=target_dtype)
        return custom_model
    
    # Optional: apply PEFT LoRA to selected custom modules (not the base transformer)
    try:
        target_modules = []
        if use_peft_lora_local_decoder:
            target_modules.extend([
                "latent_proj", "local_out_proj", "node_type_head", "node_len_head"
            ])
        if use_peft_lora_span_embeddings:
            # Safe linear targets inside span embedding layer
            target_modules.extend([
                "span_projection", "gate_linear"
            ])
        # Deduplicate
        target_modules = list(dict.fromkeys(target_modules))
        if len(target_modules) > 0:
            print(f"🔧 Applying PEFT LoRA to modules: {target_modules}")
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=peft_lora_rank,
                lora_alpha=peft_lora_alpha,
                lora_dropout=peft_lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            custom_model = get_peft_model(custom_model, peft_config)
            # Print a short summary of trainable params (LoRA adapters)
            try:
                custom_model.print_trainable_parameters()
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️  Failed to apply PEFT LoRA to custom modules: {e}")
    
    return custom_model

def process_code_for_spans(code_text, language="python"):
    """
    Process code text to extract AST span metadata for the model.
    
    Args:
        code_text: Source code string
        language: Programming language (python, java, etc.)
    
    Returns:
        Dictionary with span metadata tensors
    """
    try:
        # Parse AST and get spans using our improved parsing
        ast_root = parse_to_ast(code_text, ps_language=language)
        if ast_root is None:
            return None
        
        ast_spans = get_ast_leaf_nodes_for_spans(ast_root)
        
        # Tokenize the code to get sequence length
        tokens = tokenizer.encode(code_text, add_special_tokens=False)
        seq_len = len(tokens)
        
        # Initialize metadata arrays as numpy arrays
        span_types = np.zeros(seq_len, dtype=np.int64)  # 0 = unknown
        positions = np.zeros(seq_len, dtype=np.int64)   # position within span
        boundaries = np.zeros(seq_len, dtype=np.int64)  # 0 = middle, 1 = start, 2 = end, 3 = single
        
        # Span type mapping (from our SpanAwareEmbeddingLayer)
        span_type_map = {
                'unknown': 0, 'keyword': 1, 'identifier': 2, 'string': 3,
                'number': 4, 'comment': 5, 'operator': 6, 'punctuation': 7,
                'module': 8, 'class': 9, 'function': 10, 'text': 11, 
                '(': 12, ')': 13, '$': 14, '@': 15,
                '=': 16, 'expression_statement': 17, 'type_identifier': 18,
                'field_identifier': 19, '::': 20, 'function_definition': 21,
                'block': 22, '"': 23, 'program': 24, 'property_identifier': 25,
                "'": 26, 'string_start': 27, 'call_expression': 28,
                'namespace_identifier': 29, 'class_declaration': 30,
                'number_literal': 31, 'string_fragment': 32, 'if_statement': 33,
                'primitive_type': 34, 'string_end': 35, 'const': 36,
                'namespace_definition': 37, 'impl_item': 38, 'source_file': 39,
                'translation_unit': 40, 'integer_literal': 41, 'let': 42,
                'else_clause': 43, 'function_item': 44, 'method_declaration': 45,
                'this': 46, 'decorated_definition': 47, 'doc_comment': 48,
                'compound_statement': 49, 'self': 50, 'public': 51,
                'line_comment': 52, 'fn': 53, 'class_definition': 54
        }
        
        # Process each AST span
        for span in ast_spans:
            token_indices = span.get('token_indices', [])
            span_type = span.get('type', 'unknown')
            
            if not token_indices:
                continue
            
            # Convert token_indices to numpy array
            token_indices = np.array(token_indices, dtype=np.int64)
            
            # Map span type to ID
            span_type_id = span_type_map.get(span_type, 0)
            
            # Filter valid token indices
            valid_mask = (token_indices >= 0) & (token_indices < seq_len)
            valid_indices = token_indices[valid_mask]
            
            if len(valid_indices) == 0:
                continue
            
            # Fill metadata for each token in span
            for pos, token_idx in enumerate(valid_indices):
                span_types[token_idx] = span_type_id
                positions[token_idx] = min(pos, 31)  # Cap at 31 (0-indexed, max 32)
                
                # Determine boundary type
                if len(valid_indices) == 1:
                    boundaries[token_idx] = 3  # single token span
                elif pos == 0:
                    boundaries[token_idx] = 1  # start
                elif pos == len(valid_indices) - 1:
                    boundaries[token_idx] = 2  # end
                else:
                    boundaries[token_idx] = 0  # middle
        
        return {
            'span_types': torch.from_numpy(span_types).unsqueeze(0).long(),  # Add batch dim, faster than torch.tensor
            'positions': torch.from_numpy(positions).unsqueeze(0).long(),
            'boundaries': torch.from_numpy(boundaries).unsqueeze(0).long(),
            'raw_spans': ast_spans  # Keep original spans for mean pooling approach
        }
        
    except Exception as e:
        print(f"Error processing spans: {e}")
        return None

# Model creation moved to function above and main block below




# ============================================================================
# FINE-TUNING SETUP WITH MEAN POOLING EMBEDDINGS
# ============================================================================

def setup_finetuning(
    model,
    learning_rate=5e-5,
    batch_size=4,
    max_length=1024,
    num_epochs=5,
    log_dir=None,
    train_languages=training_languages,
    unfreeze_last_k_layers=2,
    unfreeze_lm_head=True,
    span_dropout_prob=0.0,
):
    """
    Set up fine-tuning process with mean pooling embeddings for Python data.
    
    Args:
        model: The model to train
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_epochs: Number of training epochs
        log_dir: Directory for TensorBoard logs (auto-generated if None)
    """
    print("\n" + "="*60)
    print("🚀 SETTING UP FINE-TUNING WITH MEAN POOLING EMBEDDINGS")
    print("="*60)

    # Load all available language datasets
    print(f"📊 Loading all available language datasets...")
    dataloaders = load_language_datasets(
        base_path="/data/home/zhangsj/Data/more_big_code_language",
        languages=train_languages,
        #avabliable languages "cpp", "java", "javascript", "php", "python", "rust"
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )
    
    if not dataloaders:
        print("❌ No datasets found. Please run test_parquet_processing.py first to generate the data.")
        return None
    
    # Combine all dataloaders into one
    all_datasets = []
    total_samples = 0
    for lang, loader in dataloaders.items():
        all_datasets.append(loader.dataset)
        total_samples += len(loader.dataset)
        print(f"   {lang}: {len(loader.dataset):,} samples")
    
    # Create combined dataset
    combined_df = pd.concat([ds.df for ds in all_datasets], ignore_index=True)
    combined_dataset = ASTSpanDataset.__new__(ASTSpanDataset)
    combined_dataset.tokenizer = tokenizer
    combined_dataset.max_length = max_length
    combined_dataset.language = "multi"  # Multi-language
    combined_dataset.df = combined_df
    combined_dataset.span_type_vocab = all_datasets[0].span_type_vocab
    
    # Create combined dataloader
    sampler = None
    use_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if use_dist else 1
    rank = dist.get_rank() if use_dist else 0
    if use_dist:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(combined_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        shuffle = False
    else:
        shuffle = True
    def collate_fn(batch):
        """Custom collate function to handle span metadata."""
        # Stack regular tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Collect span metadata
        batch_span_metadata = {
            'span_types': torch.stack([item['span_metadata']['span_types'] for item in batch]),
            'positions': torch.stack([item['span_metadata']['positions'] for item in batch]),
            'boundaries': torch.stack([item['span_metadata']['boundaries'] for item in batch]),
            'raw_spans': [item['span_metadata']['raw_spans'] for item in batch]
        }
        
        # Collect other metadata
        batch_metadata = {
            'original_content': [item['original_content'] for item in batch],
            'language': [item['language'] for item in batch],
            'repo_path': [item['repo_path'] for item in batch],
            'repo_name': [item['repo_name'] for item in batch],
            'stars_count': [item['stars_count'] for item in batch],
            'coverage_percentage': [item['coverage_percentage'] for item in batch]
        }
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'span_metadata': batch_span_metadata,
            'metadata': batch_metadata
        }
    
    train_dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        sampler=sampler
    )
    
    print(f"✅ Multi-language training dataloader created:")
    print(f"   Total dataset size: {len(train_dataloader.dataset):,} samples")
    print(f"   Languages: {list(dataloaders.keys())}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches: {len(train_dataloader):,}")
    print(f"   Max sequence length: {max_length}")
    
    # Set up optimizer (only train the new embedding components)
    print(f"\n🔧 Setting up optimizer...")
    
    # Parameters to train
    trainable_params = []
    # Support DDP-wrapped and/or PEFT-wrapped models
    base_model = model.module if hasattr(model, 'module') else model
    # Detect if model is PEFT-wrapped (LoRA)
    is_peft_wrapped = hasattr(base_model, "peft_config") or base_model.__class__.__name__.lower().startswith("peft")
    
    if is_peft_wrapped:
        # With PEFT, train only LoRA adapters (already requires_grad=True)
        print("   🔧 PEFT detected - training LoRA adapters only")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        # Train span-specific components
        trainable_params.extend(list(base_model.model.embed_tokens.span_type_embeddings.parameters()))
        trainable_params.extend(list(base_model.model.embed_tokens.span_projection.parameters()))
        
        # ✨ NEW: Add token adapter parameters
        trainable_params.extend(list(base_model.model.embed_tokens.token_adapter.parameters()))
        
        trainable_params.extend(list(base_model.model.embed_tokens.layer_norm.parameters()))
        # Also train the span-only normalization used for gating
        if hasattr(base_model.model.embed_tokens, 'span_norm'):
            trainable_params.extend(list(base_model.model.embed_tokens.span_norm.parameters()))
        trainable_params.extend(list(base_model.model.embed_tokens.gate_linear.parameters()))
        
        print("   Training parameters:")
        print("   ✅ Token adapter (NEW - allows adaptation of frozen embeddings)")
        print("   ✅ Span type embeddings")
        print("   ✅ Span projection layer")
        print("   ✅ Gating network")
        print("   ✅ Layer normalization")
        print("   ❄️  Token embeddings (frozen)")
        print("   ❄️  Transformer layers (frozen)")

        # Configure span dropout probability for training robustness if provided
        try:
            if hasattr(base_model.model, 'embed_tokens') and hasattr(base_model.model.embed_tokens, 'span_dropout_prob'):
                base_model.model.embed_tokens.span_dropout_prob = float(span_dropout_prob)
                print(f"   ✅ Set span_dropout_prob to {base_model.model.embed_tokens.span_dropout_prob}")
        except Exception as e:
            print(f"   ⚠️  Could not set span_dropout_prob: {e}")
        

    # Include local decoder parameters if enabled
    try:
        if hasattr(base_model, 'local_decoder_enabled') and base_model.local_decoder_enabled:
            if hasattr(base_model, 'local_token_embed'):
                trainable_params.extend(list(base_model.local_token_embed.parameters()))
            if hasattr(base_model, 'latent_proj'):
                trainable_params.extend(list(base_model.latent_proj.parameters()))
            if hasattr(base_model, 'node_type_head'):
                trainable_params.extend(list(base_model.node_type_head.parameters()))
            if hasattr(base_model, 'node_len_head'):
                trainable_params.extend(list(base_model.node_len_head.parameters()))
            if hasattr(base_model, 'local_rnn'):
                trainable_params.extend(list(base_model.local_rnn.parameters()))
            if hasattr(base_model, 'local_transformer'):
                trainable_params.extend(list(base_model.local_transformer.parameters()))
            if hasattr(base_model, 'pos_embed'):
                trainable_params.extend(list(base_model.pos_embed.parameters()))
            if hasattr(base_model, 'local_out_proj'):
                trainable_params.extend(list(base_model.local_out_proj.parameters()))
            # LoRA residuals for local decoder out proj
            if hasattr(base_model, 'lora_A'):
                trainable_params.extend(list(base_model.lora_A.parameters()))
            if hasattr(base_model, 'lora_B'):
                trainable_params.extend(list(base_model.lora_B.parameters()))
            print("   ✅ Local decoder parameters added to optimizer")
    except Exception as e:
        print(f"   ⚠️  Could not add local decoder params: {e}")

    # Additionally unfreeze last K transformer blocks and lm_head to ensure generation is learnable
    try:
        if unfreeze_lm_head and hasattr(base_model, 'lm_head'):
            trainable_params.extend(list(base_model.lm_head.parameters()))
            print("   ✅ Unfreezing lm_head")
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
            k = max(0, min(int(unfreeze_last_k_layers), len(layers)))
            if k > 0:
                for layer in layers[-k:]:
                    trainable_params.extend(list(layer.parameters()))
                print(f"   ✅ Unfreezing last {k} transformer layer(s)")
            else:
                print("   ℹ️  unfreeze_last_k_layers set to 0; no transformer layers unfrozen")
        else:
            print("   ⚠️  Could not access model.model.layers; skipping layer unfreeze")
    except Exception as e:
        print(f"   ⚠️  Error while unfreezing layers: {e}")
    
    # Freeze/unfreeze only when not using PEFT
    if not is_peft_wrapped:
        # Freeze non-trainable parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Enable gradients only for trainable parameters
        for param in trainable_params:
            param.requires_grad = True
    
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    
    print(f"   Trainable parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.1f}%)")
    
    # Create optimizer
    from torch.optim import AdamW
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataloader))
    
    print(f"   Optimizer: AdamW (lr={learning_rate}, weight_decay=0.01)")
    print(f"   Scheduler: CosineAnnealingLR")
    
    # Set up TensorBoard logging
    if log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"/data/home/zhangsj/AST_decoding/tensorboard_logs/{trail_name}_{embedding_approach}_{timestamp}"
    
    # Only rank 0 writes logs
    if use_dist and rank != 0:
        writer = None
    else:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    
    if writer:
        print(f"\n📊 TensorBoard logging setup:")
        print(f"   Log directory: {log_dir}")
        print(f"   View logs with: tensorboard --logdir {log_dir}")
    
    # Get model config for logging (unwrap DDP/PEFT if needed)
    model_config = base_model.config
    
    # Log initial model configuration
    if writer:
        writer.add_text("config/embedding_approach", embedding_approach)
        writer.add_text("config/learning_rate", str(learning_rate))
        writer.add_text("config/batch_size", str(batch_size))
        writer.add_text("config/max_length", str(max_length))
        writer.add_text("config/num_epochs", str(num_epochs))
        writer.add_text("config/dataset_size", str(len(train_dataloader.dataset)))
    
    # Log important setup parameters from lines 15-21
    if writer:
        writer.add_text("setup/model_path", model_path)
        writer.add_text("setup/trail_name", trail_name)
        writer.add_text("setup/training_languages", str(training_languages))
        writer.add_text("setup/embedding_approach", embedding_approach)
        writer.add_text("setup/unfreeze_last_k_layers", str(unfreeze_last_k_layers))
        writer.add_text("setup/unfreeze_lm_head", str(unfreeze_lm_head))
        writer.add_text("setup/span_dropout_prob", str(span_dropout_prob))
    # Local decoder setup logs
    try:
        if writer:
            writer.add_text("setup/use_local_decoder", str(getattr(base_model, 'local_decoder_enabled', False)))
            writer.add_text("setup/local_decoder_type", str(getattr(base_model, 'local_decoder_type', 'none')))
            writer.add_text("setup/node_reconstruction_loss_weight", str(node_reconstruction_loss_weight))
            writer.add_text("setup/node_type_cls_loss_weight", str(node_type_cls_loss_weight))
            writer.add_text("setup/node_length_cls_loss_weight", str(node_length_cls_loss_weight))
            writer.add_text("setup/max_node_length", str(max_node_length))
            writer.add_text("setup/use_peft_lora_local_decoder", str(use_peft_lora_local_decoder))
            writer.add_text("setup/use_peft_lora_span_embeddings", str(use_peft_lora_span_embeddings))
    except Exception:
        pass
    
    # Log model configuration details
    if writer:
        writer.add_text("model/config/vocab_size", str(model_config.vocab_size))
        writer.add_text("model/config/hidden_size", str(model_config.hidden_size))
        writer.add_text("model/config/num_layers", str(model_config.num_hidden_layers))
        writer.add_text("model/config/num_attention_heads", str(model_config.num_attention_heads))
        writer.add_text("model/config/max_position_embeddings", str(model_config.max_position_embeddings))
    
    return {
        'model': model,
        'train_dataloader': train_dataloader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'tokenizer': tokenizer,
        'writer': writer,
        'log_dir': log_dir,
    }


def train_model(training_setup, num_epochs=5, save_steps=5000, log_steps=50,trail_name=trail_name):
    """
    Fine-tune the model with AST span awareness.
    
    Args:
        training_setup: Dict returned from setup_finetuning()
        num_epochs: Number of training epochs
        save_steps: Save checkpoint every N steps
        log_steps: Log progress every N steps
    """
    print("\n" + "="*60)
    print("🏋️ STARTING FINE-TUNING")
    print("="*60)

    model = training_setup['model']
    train_dataloader = training_setup['train_dataloader']
    optimizer = training_setup['optimizer']
    scheduler = training_setup['scheduler']
    tokenizer = training_setup['tokenizer']
    writer = training_setup['writer']
    use_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if use_dist else 1
    rank = dist.get_rank() if use_dist else 0
    
    model.train()
    
    # Enable gradient checkpointing to save memory
    base_for_checkpoint = model.module if hasattr(model, 'module') else model
    if hasattr(base_for_checkpoint, "gradient_checkpointing_enable"):
        base_for_checkpoint.gradient_checkpointing_enable()
    
    total_steps = len(train_dataloader) * num_epochs
    step = 0
    
    print(f"Training configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Save every: {save_steps} steps")
    print(f"   Log every: {log_steps} steps")
    
    # Initialize TensorBoard logging
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        epoch_steps = 0
        
        iterator = enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")) if (not use_dist or rank == 0) else enumerate(train_dataloader)
        # Resolve device once (works with DDP-wrapped model)
        device = next(model.parameters()).device
        for batch_idx, batch in iterator:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Move span metadata to device
            span_metadata = {}
            for key, value in batch['span_metadata'].items():
                if isinstance(value, torch.Tensor):
                    span_metadata[key] = value.to(device)
                else:
                    span_metadata[key] = value
            
            # Plain teacher-only forward (no KD)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_metadata=span_metadata,
                labels=input_ids  # Causal LM training
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1
            
            # TensorBoard logging
            current_lr = scheduler.get_last_lr()[0]
            if writer:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/learning_rate', current_lr, step)
            # Log node-level losses if available
            if writer and 'outputs' in locals():
                if hasattr(outputs, 'node_recon_loss'):
                    writer.add_scalar('train/node_recon_loss', outputs.node_recon_loss.item(), step)
                if hasattr(outputs, 'node_type_loss') and outputs.node_type_loss is not None:
                    writer.add_scalar('train/node_type_loss', outputs.node_type_loss.item(), step)
                if hasattr(outputs, 'node_len_loss') and outputs.node_len_loss is not None:
                    writer.add_scalar('train/node_len_loss', outputs.node_len_loss.item(), step)
            
            # Log embedding stats periodically
            if writer and step % (log_steps * 2) == 0:
                if embedding_approach == "mean_pooling":
                    # Log span type embedding norms
                    span_type_norms = torch.norm(model.model.embed_tokens.span_type_embeddings.weight, dim=1)
                    writer.add_histogram('embeddings/span_type_norms', span_type_norms, step)
                    
                    # ✨ NEW: Log adapter statistics
                    adapter_weight_norm = torch.norm(model.model.embed_tokens.token_adapter[0].weight)
                    adapter_output_norm = torch.norm(model.model.embed_tokens.token_adapter[2].weight)
                    writer.add_scalar('embeddings/adapter_weight_norm', adapter_weight_norm.item(), step)
                    writer.add_scalar('embeddings/adapter_output_norm', adapter_output_norm.item(), step)
                    
                    # Log how much the adapter changes embeddings
                    with torch.no_grad():
                        sample_ids = input_ids[0:1, :min(100, input_ids.size(1))]  # Sample first 100 tokens
                        frozen_emb = model.model.embed_tokens.token_embeddings(sample_ids)
                        adapted_emb = frozen_emb + model.model.embed_tokens.token_adapter(frozen_emb)
                        adaptation_magnitude = torch.norm(adapted_emb - frozen_emb) / torch.norm(frozen_emb)
                        writer.add_scalar('embeddings/adapter_change_ratio', adaptation_magnitude.item(), step)
                
                
                # Log gradients if available (for all embedding approaches)
                for name, param in model.model.embed_tokens.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'gradients/{name}', param.grad, step)
                        writer.add_scalar(f'gradients/{name}_norm', torch.norm(param.grad), step)
            
            # Console logging
            if writer and step % log_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                log_msg = f"   Step {step:,}/{total_steps:,} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                
                
                print(log_msg)
                
                # Log span metadata statistics for current batch
                if embedding_approach == "mean_pooling" and span_metadata and writer:
                    num_spans = sum(len(spans) for spans in span_metadata['raw_spans'])
                    writer.add_scalar('batch/num_spans', num_spans / len(span_metadata['raw_spans']), step)
                    
                    # Log span type distribution
                    span_types_flat = span_metadata['span_types'].flatten()
                    unique_types, counts = torch.unique(span_types_flat, return_counts=True)
                    for type_id, count in zip(unique_types.tolist(), counts.tolist()):
                        # Handle both tensor and int types
                        count_value = count.item() if hasattr(count, 'item') else count
                        writer.add_scalar(f'span_types/type_{type_id}', count_value, step)
            
            # Save checkpoint
            if writer and step % save_steps == 0:
                checkpoint_path = f"/data/home/zhangsj/AST_decoding/checkpoints/{trail_name}/checkpoint_step_{step}"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                # Save training state
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'embedding_approach': embedding_approach,
                    'best_loss': best_loss
                }, f"{checkpoint_path}/training_state.pt")
                
                print(f"   💾 Checkpoint saved: {checkpoint_path}")
                writer.add_text('checkpoints/saved', f"Step {step}: {checkpoint_path}", step)
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if writer:
            writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
        
        print(f"   Epoch {epoch + 1} completed | Average Loss: {avg_epoch_loss:.4f}")
        
        # Save best model
        if writer and avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = f"/data/home/zhangsj/AST_decoding/checkpoints/{trail_name}/best_model"
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"   🏆 New best model saved: {best_model_path} (loss: {best_loss:.4f})")
            writer.add_text('checkpoints/best_model', f"Epoch {epoch+1}: {best_model_path} (loss: {best_loss:.4f})", step)
    
    # Save final model
    if writer:
        final_path = f"/data/home/zhangsj/AST_decoding/checkpoints/{trail_name}/final_model"
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
    
    # Final TensorBoard logging
    if writer:
        writer.add_text('training/status', 'COMPLETED', step)
        writer.add_scalar('training/final_loss', loss.item(), step)
        writer.add_scalar('training/best_loss', best_loss, step)
    
    if writer:
        writer.close()
    
    if rank == 0:
        print(f"\n✅ FINE-TUNING COMPLETED!")
        if writer:
            print(f"   Final model saved to: {final_path}")
            print(f"   Best model saved under: /data/home/zhangsj/AST_decoding/checkpoints/{trail_name}/best_model")
            print(f"   📊 TensorBoard logs: {training_setup['log_dir']}")
            print(f"   🔍 View with: tensorboard --logdir {training_setup['log_dir']}")
        print(f"   Total steps: {step:,}")
        print(f"   Final loss: {loss.item():.4f}")
        print(f"   Best loss: {best_loss:.4f}")
    
    return model


# ============================================================================
# MAIN FINE-TUNING EXECUTION
# ============================================================================

if __name__ == "__main__":
    use_dist = False
    local_rank = 0
    if use_ddp:
        if not dist.is_available():
            raise RuntimeError("DDP requested but torch.distributed not available")
        dist.init_process_group(backend="nccl", init_method="env://")
        use_dist = dist.is_initialized()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print("🎯 AST SPAN-AWARE MODEL FINE-TUNING")
        print(f"Embedding approach: {embedding_approach}")
        print(f"Target languages: {training_languages}")
    
    # Create the span-aware model (uses current cuda device)
    model = create_span_aware_model()
    if use_dist:
        # Sanity check before wrapping with DDP
        try:
            rank = dist.get_rank()
        except Exception:
            rank = -1
        try:
            num_tensors = sum(1 for _ in model.parameters())
            total_numel = sum(p.numel() for p in model.parameters())
            device_str = str(next(model.parameters()).device) if num_tensors > 0 else "unknown"
            current_cuda = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        except Exception:
            num_tensors, total_numel, device_str, current_cuda = -1, -1, "error", "error"
        print(f"[rank{rank}] Model sanity: param_tensors={num_tensors}, total_params={total_numel}, device={device_str}, cuda_current_device={current_cuda}")
        # Ensure all ranks finished model construction before DDP init
        dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Set up fine-tuning
    training_setup = setup_finetuning(
        model,
        learning_rate=5e-5,
        batch_size=4,  # Reduced from 4 to 2
        max_length=384,  # Reduced from 1024 to 512
        num_epochs=5,
        unfreeze_last_k_layers=2,  # keep base frozen
        unfreeze_lm_head=True,    # keep lm_head frozen
        span_dropout_prob=0.3,
        )
    
    if training_setup is not None:
        # Start fine-tuning
        trained_model = train_model(
            training_setup,
            num_epochs=5,
            save_steps=22000,
            log_steps=50
        )
        if not use_dist or dist.get_rank() == 0:
            print("🎉 Training completed successfully!")
    else:
        if not use_dist or dist.get_rank() == 0:
            print("❌ Failed to set up training. Check data availability.")
    if use_dist:
        dist.barrier()
        dist.destroy_process_group()