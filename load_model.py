from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import Qwen2ForCausalLM, Qwen2Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from ast_parsing_folder.AST_parsing import parse_to_ast,get_ast_leaf_nodes_for_spans

model_path = "/data/home/zhangsj/qwen_coder_1.5b"
config = Qwen2Config.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
embedding_approach = "mean_pooling"
trail_name="9_16_mean_pooling"
freeze_token_span_weights = True  # Change this to True to freeze the weights alpha
training_languages = ["python"]#, "cpp", "java", "javascript", "php", "rust
span_type_loss_weight = 0.5  # Weight for span type loss in combined loss (span_type_only approach)
# Configuration options for span-aware embeddings:
# - embedding_approach: 
#   * "placeholder": Standard embeddings only
#   * "multi_component": Additive combination of token + span type + position + boundary embeddings
#   * "mean_pooling": Mean pooling within spans + weighted combination with token embeddings
#   * "mean_pooling_only": Mean pooling within spans only (no weighted combination)
#   * "span_type_only": Focus on span type prediction task (classification head on top of embeddings)
# - freeze_token_span_weights: True/False - whether to freeze the token-span combination weights

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



class PlaceholderCustomEmbeddingLayer(nn.Module):
    """Placeholder custom embedding layer that just uses the original embedding"""
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    
    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class SpanAwareEmbeddingLayer(nn.Module):
    """
    AST Span-Aware embedding layer that incorporates structural code information.
    
    Combines token embeddings with:
    - AST span type embeddings (keyword, identifier, string, etc.)
    - Position within span embeddings  
    - Span boundary embeddings (start/middle/end)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Traditional token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        
        # AST span type embeddings
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
        
        # Position within span (0, 1, 2, ... for tokens in a span)
        self.position_embeddings = nn.Embedding(32, config.hidden_size)  # max 32 tokens per span
        
        # Span boundaries (0=middle, 1=start, 2=end, 3=single)
        self.boundary_embeddings = nn.Embedding(4, config.hidden_size)
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(4))
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, input_ids, span_metadata=None):
        """
        Args:
            input_ids: [batch_size, seq_len] token IDs
            span_metadata: Optional dict with span info (falls back to token-only if None)
        """
        # Get token embeddings
        token_emb = self.token_embeddings(input_ids)
        
        # If no span metadata, return traditional embeddings
        if span_metadata is None:
            return self.layer_norm(token_emb)
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get span embeddings with fallback to zeros
        span_types = span_metadata.get('span_types', torch.zeros(batch_size, seq_len, dtype=torch.long, device=device))
        positions = span_metadata.get('positions', torch.zeros(batch_size, seq_len, dtype=torch.long, device=device))
        boundaries = span_metadata.get('boundaries', torch.zeros(batch_size, seq_len, dtype=torch.long, device=device))
        
        span_type_emb = self.span_type_embeddings(span_types)
        position_emb = self.position_embeddings(positions)
        boundary_emb = self.boundary_embeddings(boundaries)
        
        # Combine with learnable weights
        weights = torch.softmax(self.combination_weights, dim=0)
        combined_emb = (
            weights[0] * token_emb +
            weights[1] * span_type_emb + 
            weights[2] * position_emb +
            weights[3] * boundary_emb
        )
        
        return self.layer_norm(combined_emb)

class OnlyMeanPooledSpanEmbeddingLayer(nn.Module):
    """
    Mean Pooling Only Approach - No weighted combination, just mean pooling within spans.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, input_ids, span_metadata=None):
        """
        Forward with mean pooling over AST spans only.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            span_metadata: Dict containing raw_spans with token_indices
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get original token embeddings
        token_emb = self.token_embeddings(input_ids)  # [B, L, H]
        
        if span_metadata is None or 'raw_spans' not in span_metadata:
            return self.layer_norm(token_emb)
        
        # Initialize output as copy of token embeddings
        output_emb = token_emb.clone()  # [B, L, H]
        
        # Process each batch
        for batch_idx in range(batch_size):
            batch_spans = span_metadata.get('raw_spans', [])
            if not batch_spans or batch_idx >= len(batch_spans):
                continue
                
            item_spans = batch_spans[batch_idx]
            if not item_spans:
                continue
                
            # Process each span in this batch item
            for span in item_spans:
                if not isinstance(span, dict):
                    continue
                    
                token_indices = span.get('token_indices', [])
                
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
                pooled_span_emb = span_token_embs.mean(dim=0)  # [H]
                
                # Assign this span embedding to all tokens in the span
                for idx in valid_indices:
                    output_emb[batch_idx, idx] = pooled_span_emb
        
        return self.layer_norm(output_emb)


class SpanTypeOnlyEmbeddingLayer(nn.Module):
    """
    Span Type Only Approach - Focus specifically on predicting span types.
    
    This approach:
    1. Uses standard token embeddings as base
    2. Adds a span type prediction head
    3. Trains to predict the correct span type for each token
    4. Can be used for span type classification tasks
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Standard token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        
        # Span type vocabulary (same as other approaches for consistency)
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
        
        # Span type prediction head
        self.span_type_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, self.num_span_types)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, input_ids, span_metadata=None):
        """
        Forward pass that returns both embeddings and span type predictions.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs
            span_metadata: Dict containing span_types for training
            
        Returns:
            dict with:
                - embeddings: [batch_size, seq_len, hidden_size] 
                - span_type_logits: [batch_size, seq_len, num_span_types]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get token embeddings
        token_emb = self.token_embeddings(input_ids)  # [B, L, H]
        token_emb = self.layer_norm(token_emb)
        
        # Predict span types for each token
        span_type_logits = self.span_type_head(token_emb)  # [B, L, num_span_types]
        
        return {
            'embeddings': token_emb,
            'span_type_logits': span_type_logits
        }
    
    def compute_span_type_loss(self, span_type_logits, target_span_types, attention_mask=None):
        """
        Compute cross-entropy loss for span type prediction.
        
        Args:
            span_type_logits: [batch_size, seq_len, num_span_types]
            target_span_types: [batch_size, seq_len] ground truth span types
            attention_mask: [batch_size, seq_len] mask for valid tokens
            
        Returns:
            loss: scalar tensor
        """
        batch_size, seq_len = target_span_types.shape
        
        # Reshape for cross-entropy loss
        logits_flat = span_type_logits.view(-1, self.num_span_types)  # [B*L, num_span_types]
        targets_flat = target_span_types.view(-1)  # [B*L]
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(logits_flat, targets_flat)  # [B*L]
        
        if attention_mask is not None:
            # Apply attention mask
            mask_flat = attention_mask.view(-1).float()  # [B*L]
            losses = losses * mask_flat
            loss = losses.sum() / mask_flat.sum().clamp(min=1.0)
        else:
            loss = losses.mean()
        
        return loss


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
    
    def __init__(self, config, freeze_token_span_weights=False, span_dropout_prob=0.0):
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
        
        # ✨ NEW: Auxiliary span type prediction head (per-token classification)
        self.span_type_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, self.num_span_types)
        )
        
        # Span enhancement projection (optional refinement of pooled embeddings)
        self.span_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Token-Span combination weights (legacy; retained for compatibility)
        self.token_span_weights = nn.Parameter(torch.tensor([0.8, 0.2]))  # [token_weight, span_weight]
        
        # Freeze token-span weights if requested
        if freeze_token_span_weights:
            self.token_span_weights.requires_grad = False
            print(f"🔒 Token-span weights frozen: {self.token_span_weights.data.tolist()}")
        else:
            print(f"🔓 Token-span weights trainable: {self.token_span_weights.data.tolist()}")
        
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
            # Still produce logits for potential auxiliary supervision later
            logits_input = self.layer_norm(combined_emb)
            span_type_logits = self.span_type_head(logits_input)
            return {
                'embeddings': combined_emb,
                'span_type_logits': span_type_logits
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
            drop_vec = (torch.rand(batch_size, device=device) < self.span_dropout_prob).float()  # 1 => drop
            # Apply to mask: if dropped, zero out all span tokens for that sample
            span_mask = span_mask * (1.0 - drop_vec.view(-1, 1))

        # Adaptive gating between token_emb and span_emb
        # Compute gate on normalized span branch while keeping token branch raw
        span_emb_norm = self.span_norm(span_emb)
        gate_input = torch.cat([token_emb, span_emb_norm], dim=-1)  # [B, L, 2H]
        gate = self.gate_activation(self.gate_linear(gate_input))  # [B, L, H] in [0,1]
        # Ensure no-span positions (or dropped) fall back to token-only
        gate = gate * span_mask.unsqueeze(-1)
        # Store for logging
        self.last_gate = gate.detach()
        self.last_span_mask = span_mask.detach()
        # Gating-only combination (no global token/span weights)
        combined_emb = (1.0 - gate) * token_emb + gate * span_emb
        
        # Return combined without a post-combine normalization to avoid shifting base distribution
        combined_emb = self.dropout(combined_emb)
        
        # ✨ NEW: Produce auxiliary span type logits from combined embeddings
        logits_input = self.layer_norm(combined_emb)
        span_type_logits = self.span_type_head(logits_input)  # [B, L, num_span_types]
        
        return {
            'embeddings': combined_emb,
            'span_type_logits': span_type_logits,
            'gate': gate,
            'span_mask': span_mask
        }

    def compute_span_type_loss(self, span_type_logits, target_span_types, attention_mask=None):
        """
        Compute cross-entropy loss for span type prediction (auxiliary head).
        """
        batch_size, seq_len, _ = span_type_logits.shape
        logits_flat = span_type_logits.view(-1, self.num_span_types)  # [B*L, C]
        targets_flat = target_span_types.view(-1)  # [B*L]
        
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(logits_flat, targets_flat)  # [B*L]
        
        if attention_mask is not None:
            mask_flat = attention_mask.view(-1).float()  # [B*L]
            losses = losses * mask_flat
            loss = losses.sum() / mask_flat.sum().clamp(min=1.0)
        else:
            loss = losses.mean()
        
        return loss


class CustomQwen2Model(Qwen2ForCausalLM):
    def __init__(self, config, embedding_type="placeholder", freeze_token_span_weights=False):
        super().__init__(config)
        
        self.embedding_type = embedding_type
        
        if embedding_type == "multi_component":
            # Multi-component additive embeddings (original approach)
            self.model.embed_tokens = SpanAwareEmbeddingLayer(config)
            print("Using Multi-Component AST span-aware embedding layer")
        elif embedding_type == "mean_pooling":
            # Mean pooling + weighted combination (new approach)
            self.model.embed_tokens = MeanPooledSpanEmbeddingLayer(config, freeze_token_span_weights=freeze_token_span_weights)
            print("Using Mean Pooling AST span-aware embedding layer")
        elif embedding_type == "mean_pooling_only":
            # Mean pooling only (no weighted combination)
            self.model.embed_tokens = OnlyMeanPooledSpanEmbeddingLayer(config)
            print("Using Mean Pooling Only AST span-aware embedding layer")
        elif embedding_type == "span_type_only":
            # Span type prediction only approach
            self.model.embed_tokens = SpanTypeOnlyEmbeddingLayer(config)
            print("Using Span Type Only AST span-aware embedding layer")
        else:
            # Use placeholder embedding layer
            self.model.embed_tokens = PlaceholderCustomEmbeddingLayer(config)
            print("Using placeholder embedding layer")
    
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

        if self.embedding_type == "span_type_only":
            # Special handling for span type prediction
            embedding_output = self.model.embed_tokens(input_ids, span_metadata)
            
            # Extract embeddings and span type logits
            inputs_embeds = embedding_output['embeddings']
            span_type_logits = embedding_output['span_type_logits']
            
            # Forward pass through transformer with embeddings
            # Remove conflicting keys from kwargs to avoid duplicate arguments
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['input_ids', 'inputs_embeds']}
            outputs = super().forward(
                input_ids=None,
                attention_mask=attention_mask, 
                inputs_embeds=inputs_embeds,
                **filtered_kwargs
            )
            
            # Add span type predictions to outputs
            outputs.span_type_logits = span_type_logits
            
            # Compute span type loss if span_metadata contains target span types
            if span_metadata is not None and 'span_types' in span_metadata:
                span_type_loss = self.model.embed_tokens.compute_span_type_loss(
                    span_type_logits, 
                    span_metadata['span_types'], 
                    attention_mask
                )
                outputs.span_type_loss = span_type_loss
            
            return outputs
            
        elif hasattr(self.model.embed_tokens, 'forward') and 'span_metadata' in self.model.embed_tokens.forward.__code__.co_varnames:
            # Other span-aware embedding layers
            inputs_embeds = self.model.embed_tokens(input_ids, span_metadata)
            # Remove conflicting keys from kwargs to avoid duplicate arguments
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['input_ids', 'inputs_embeds']}
            return super().forward(
                input_ids=None,
                attention_mask=attention_mask, 
                inputs_embeds=inputs_embeds,
                **filtered_kwargs
            )
        else:
            # Regular embedding layer
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )


# Step 1: Load the original model (only when run as main)
def create_span_aware_model():
    """Create and initialize the span-aware model."""
    original_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Step 2: Create your custom model with the same config (with span awareness)
    config = original_model.config
    
    custom_model = CustomQwen2Model(config, embedding_type=embedding_approach, freeze_token_span_weights=freeze_token_span_weights)
    
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
    if embedding_approach in ["multi_component", "mean_pooling", "mean_pooling_only", "span_type_only"]:
        # Copy to token embeddings component of span-aware layer
        with torch.no_grad():
            custom_model.model.embed_tokens.token_embeddings.weight.copy_(original_embedding)
            print(f"Copied token embeddings to {embedding_approach} layer: {original_embedding.shape}")
            if embedding_approach == "span_type_only":
                print(f"{embedding_approach} span type prediction head initialized randomly (will learn during training)")
            else:
                print(f"{embedding_approach} embedding components initialized randomly (will learn during training)")
    else:
        # Copy to placeholder layer
        with torch.no_grad():
            custom_model.model.embed_tokens.embed_tokens.weight.copy_(original_embedding)
            print(f"Copied embedding weights to placeholder layer: {original_embedding.shape}")
    
    return custom_model.cuda()

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
    consistency_kl_weight=0.1,
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
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✅ Multi-language training dataloader created:")
    print(f"   Total dataset size: {len(train_dataloader.dataset):,} samples")
    print(f"   Languages: {list(dataloaders.keys())}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of batches: {len(train_dataloader):,}")
    print(f"   Max sequence length: {max_length}")
    
    # Set up optimizer (only train the new embedding components)
    print(f"\n🔧 Setting up optimizer...")
    
    # Parameters to train (focus on span-aware components)
    trainable_params = []
    
    if embedding_approach == "mean_pooling":
        # Train span-specific components
        trainable_params.extend(list(model.model.embed_tokens.span_type_embeddings.parameters()))
        trainable_params.extend(list(model.model.embed_tokens.span_projection.parameters()))
        
        # ✨ NEW: Add token adapter parameters
        trainable_params.extend(list(model.model.embed_tokens.token_adapter.parameters()))
        
        # Only add token-span weights if they're not frozen
        if model.model.embed_tokens.token_span_weights.requires_grad:
            trainable_params.extend([model.model.embed_tokens.token_span_weights])
            print("   ✅ Token-span combination weights")
        else:
            print("   🔒 Token-span combination weights (frozen)")
            
        trainable_params.extend(list(model.model.embed_tokens.layer_norm.parameters()))
        # Also train the span-only normalization used for gating
        if hasattr(model.model.embed_tokens, 'span_norm'):
            trainable_params.extend(list(model.model.embed_tokens.span_norm.parameters()))
        trainable_params.extend(list(model.model.embed_tokens.gate_linear.parameters()))
        
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
            if hasattr(model.model, 'embed_tokens') and hasattr(model.model.embed_tokens, 'span_dropout_prob'):
                model.model.embed_tokens.span_dropout_prob = float(span_dropout_prob)
                print(f"   ✅ Set span_dropout_prob to {model.model.embed_tokens.span_dropout_prob}")
        except Exception as e:
            print(f"   ⚠️  Could not set span_dropout_prob: {e}")
        
    elif embedding_approach == "span_type_only":
        # Train span type prediction head only
        trainable_params.extend(list(model.model.embed_tokens.span_type_head.parameters()))
        trainable_params.extend(list(model.model.embed_tokens.layer_norm.parameters()))
        
        print("   Training parameters:")
        print("   ✅ Span type prediction head")
        print("   ✅ Layer normalization")
        print("   ❄️  Token embeddings (frozen)")
        print("   ❄️  Transformer layers (frozen)")
        
    else:
        print(f"   ⚠️  Warning: embedding_approach is '{embedding_approach}', not 'mean_pooling' or 'span_type_only'")
        print("   Training all model parameters...")
        trainable_params = list(model.parameters())

    # Additionally unfreeze last K transformer blocks and lm_head to ensure generation is learnable
    try:
        if unfreeze_lm_head and hasattr(model, 'lm_head'):
            trainable_params.extend(list(model.lm_head.parameters()))
            print("   ✅ Unfreezing lm_head")
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
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
        log_dir = f"/data/home/zhangsj/qwen_coder_1.5b/tensorboard_logs/{trail_name}_{embedding_approach}_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"\n📊 TensorBoard logging setup:")
    print(f"   Log directory: {log_dir}")
    print(f"   View logs with: tensorboard --logdir {log_dir}")
    
    # Get model config for logging
    model_config = model.config
    
    # Log initial model configuration
    writer.add_text("config/embedding_approach", embedding_approach)
    writer.add_text("config/learning_rate", str(learning_rate))
    writer.add_text("config/batch_size", str(batch_size))
    writer.add_text("config/max_length", str(max_length))
    writer.add_text("config/num_epochs", str(num_epochs))
    writer.add_text("config/dataset_size", str(len(train_dataloader.dataset)))
    
    # Log important setup parameters from lines 15-21
    writer.add_text("setup/model_path", model_path)
    writer.add_text("setup/trail_name", trail_name)
    writer.add_text("setup/freeze_token_span_weights", str(freeze_token_span_weights))
    writer.add_text("setup/training_languages", str(training_languages))
    writer.add_text("setup/embedding_approach", embedding_approach)
    writer.add_text("setup/unfreeze_last_k_layers", str(unfreeze_last_k_layers))
    writer.add_text("setup/unfreeze_lm_head", str(unfreeze_lm_head))
    writer.add_text("setup/consistency_kl_weight", str(consistency_kl_weight))
    writer.add_text("setup/span_dropout_prob", str(span_dropout_prob))
    writer.add_text("setup/span_type_loss_weight", str(span_type_loss_weight))
    
    # Log model configuration details
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
        'consistency_kl_weight': consistency_kl_weight
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
    
    model.train()
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
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
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # Move span metadata to device
            span_metadata = {}
            for key, value in batch['span_metadata'].items():
                if isinstance(value, torch.Tensor):
                    span_metadata[key] = value.to(model.device)
                else:
                    span_metadata[key] = value
            
            # Forward pass
            if embedding_approach in ["multi_component", "mean_pooling", "mean_pooling_only", "span_type_only"]:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    span_metadata=span_metadata,
                    labels=input_ids  # Causal LM training
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
            
            # Handle different loss types
            if embedding_approach == "span_type_only":
                # For span type only, use combined loss: LM loss + span type loss
                lm_loss = outputs.loss
                
                if hasattr(outputs, 'span_type_loss'):
                    span_type_loss = outputs.span_type_loss
                    # Combined loss with configurable weighting
                    loss = (1-span_type_loss_weight) * lm_loss + span_type_loss_weight * span_type_loss
                    
                    # Log both losses separately for monitoring
                    writer.add_scalar('train/lm_loss', lm_loss.item(), step)
                    writer.add_scalar('train/span_type_loss', span_type_loss.item(), step)
                    writer.add_scalar('train/combined_loss', loss.item(), step)
                else:
                    # Fallback to LM loss only if span type loss not available
                    loss = lm_loss
                    writer.add_scalar('train/lm_loss', lm_loss.item(), step)
            else:
                # Base LM loss
                loss = outputs.loss

                # Optional consistency KL between AST-on and AST-off paths
                # Only apply for embedding approaches that use span metadata at training
                if embedding_approach in ["mean_pooling", "multi_component", "mean_pooling_only"] and training_setup.get('scheduler') is not None:
                    try:
                        consistency_kl_weight = float(training_setup.get('consistency_kl_weight', 0.0))
                    except Exception:
                        consistency_kl_weight = 0.0
                    if consistency_kl_weight > 0.0:
                        with torch.no_grad():
                            # Forward without span metadata (teacher/student choice is arbitrary; use teacher = with spans)
                            outputs_no_ast = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=None,
                                span_metadata=None
                            )
                        # Compute KL on logits (align no-ast to with-ast)
                        if hasattr(outputs, 'logits') and hasattr(outputs_no_ast, 'logits'):
                            logits_ast = outputs.logits.detach()  # [B, L, V]
                            logits_no_ast = outputs_no_ast.logits  # [B, L, V]
                            # Use only positions where attention_mask == 1
                            mask = attention_mask.bool()
                            vocab_size = logits_ast.size(-1)
                            log_p_no_ast = F.log_softmax(logits_no_ast, dim=-1)
                            p_ast = F.softmax(logits_ast, dim=-1)
                            kl = F.kl_div(
                                log_p_no_ast[mask],
                                p_ast[mask],
                                reduction='batchmean'
                            )
                            loss = loss + consistency_kl_weight * kl
                            writer.add_scalar('train/consistency_kl', kl.item(), step)
            
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
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/learning_rate', current_lr, step)
            
            # Log embedding weights periodically
            if step % (log_steps * 2) == 0:
                if embedding_approach == "mean_pooling":
                    # Log token-span combination weights
                    token_span_weights = torch.softmax(model.model.embed_tokens.token_span_weights, dim=0)
                    writer.add_scalar('embeddings/token_weight', token_span_weights[0].item(), step)
                    writer.add_scalar('embeddings/span_weight', token_span_weights[1].item(), step)
                    
                    # Log span type embedding norms
                    span_type_norms = torch.norm(model.model.embed_tokens.span_type_embeddings.weight, dim=1)
                    writer.add_histogram('embeddings/span_type_norms', span_type_norms, step)
                    
                    # Log token-span weights status
                    if model.model.embed_tokens.token_span_weights.requires_grad:
                        writer.add_scalar('embeddings/token_span_weights_trainable', 1.0, step)
                    else:
                        writer.add_scalar('embeddings/token_span_weights_trainable', 0.0, step)
                    
                    # ✨ NEW: Log adapter statistics
                    adapter_weight_norm = torch.norm(model.model.embed_tokens.token_adapter[0].weight)
                    adapter_output_norm = torch.norm(model.model.embed_tokens.token_adapter[2].weight)
                    writer.add_scalar('embeddings/adapter_weight_norm', adapter_weight_norm.item(), step)
                    writer.add_scalar('embeddings/adapter_output_norm', adapter_output_norm.item(), step)
                    
                    # Log how much the adapter changes embeddings
                    with torch.no_grad():
                        sample_ids = input_ids[0:1, :min(100, seq_len)]  # Sample first 100 tokens or seq_len
                        frozen_emb = model.model.embed_tokens.token_embeddings(sample_ids)
                        adapted_emb = frozen_emb + model.model.embed_tokens.token_adapter(frozen_emb)
                        adaptation_magnitude = torch.norm(adapted_emb - frozen_emb) / torch.norm(frozen_emb)
                        writer.add_scalar('embeddings/adapter_change_ratio', adaptation_magnitude.item(), step)
                
                elif embedding_approach == "span_type_only":
                    # Log span type prediction accuracy
                    if hasattr(outputs, 'span_type_logits') and span_metadata and 'span_types' in span_metadata:
                        with torch.no_grad():
                            predictions = torch.argmax(outputs.span_type_logits, dim=-1)
                            targets = span_metadata['span_types']
                            
                            # Calculate accuracy (considering attention mask)
                            if attention_mask is not None:
                                mask = attention_mask.bool()
                                correct = (predictions == targets) & mask
                                accuracy = correct.sum().float() / mask.sum().float()
                            else:
                                correct = (predictions == targets)
                                accuracy = correct.float().mean()
                            
                            writer.add_scalar('train/span_type_accuracy', accuracy.item(), step)
                
                # Log gradients if available (for all embedding approaches)
                for name, param in model.model.embed_tokens.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'gradients/{name}', param.grad, step)
                        writer.add_scalar(f'gradients/{name}_norm', torch.norm(param.grad), step)
            
            # Console logging
            if step % log_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                log_msg = f"   Step {step:,}/{total_steps:,} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                
                # Add detailed loss breakdown for span_type_only approach
                if embedding_approach == "span_type_only":
                    if hasattr(outputs, 'span_type_loss'):
                        log_msg += f" | LM: {lm_loss.item():.4f} | Span: {span_type_loss.item():.4f}"
                    
                    # Add accuracy if available
                    if hasattr(outputs, 'span_type_logits') and span_metadata and 'span_types' in span_metadata:
                        with torch.no_grad():
                            predictions = torch.argmax(outputs.span_type_logits, dim=-1)
                            targets = span_metadata['span_types']
                            
                            if attention_mask is not None:
                                mask = attention_mask.bool()
                                correct = (predictions == targets) & mask
                                accuracy = correct.sum().float() / mask.sum().float()
                            else:
                                correct = (predictions == targets)
                                accuracy = correct.float().mean()
                            
                            log_msg += f" | Span Acc: {accuracy.item():.3f}"
                
                print(log_msg)
                
                # Log span metadata statistics for current batch
                if embedding_approach == "mean_pooling" and span_metadata:
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
            if step % save_steps == 0:
                checkpoint_path = f"/data/home/zhangsj/qwen_coder_1.5b/checkpoints/{trail_name}/checkpoint_step_{step}"
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
        writer.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
        
        print(f"   Epoch {epoch + 1} completed | Average Loss: {avg_epoch_loss:.4f}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = f"/data/home/zhangsj/qwen_coder_1.5b/checkpoints/{trail_name}/best_model"
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"   🏆 New best model saved: {best_model_path} (loss: {best_loss:.4f})")
            writer.add_text('checkpoints/best_model', f"Epoch {epoch+1}: {best_model_path} (loss: {best_loss:.4f})", step)
    
    # Save final model
    final_path = "/data/home/zhangsj/qwen_coder_1.5b/checkpoints/{trail_name}/final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Final TensorBoard logging
    writer.add_text('training/status', 'COMPLETED', step)
    writer.add_scalar('training/final_loss', loss.item(), step)
    writer.add_scalar('training/best_loss', best_loss, step)
    
    # Log final embedding weights
    if embedding_approach == "mean_pooling":
        final_weights = torch.softmax(model.model.embed_tokens.token_span_weights, dim=0)
        writer.add_scalar('embeddings/final_token_weight', final_weights[0].item(), step)
        writer.add_scalar('embeddings/final_span_weight', final_weights[1].item(), step)
        
        weight_status = "frozen" if not model.model.embed_tokens.token_span_weights.requires_grad else "trained"
        print(f"   📊 Final embedding weights ({weight_status}) - Token: {final_weights[0].item():.3f}, Span: {final_weights[1].item():.3f}")
    elif embedding_approach == "span_type_only":
        print(f"   📊 Span type prediction model trained successfully")
    
    writer.close()
    
    print(f"\n✅ FINE-TUNING COMPLETED!")
    print(f"   Final model saved to: {final_path}")
    print(f"   Best model saved to: /data/home/zhangsj/qwen_coder_1.5b/best_span_aware_model")
    print(f"   Total steps: {step:,}")
    print(f"   Final loss: {loss.item():.4f}")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   📊 TensorBoard logs: {training_setup['log_dir']}")
    print(f"   🔍 View with: tensorboard --logdir {training_setup['log_dir']}")
    
    return model


# ============================================================================
# MAIN FINE-TUNING EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🎯 AST SPAN-AWARE MODEL FINE-TUNING")
    print(f"Embedding approach: {embedding_approach}")
    print(f"Target languages: {training_languages}")
    
    # Create the span-aware model
    model = create_span_aware_model()
    
    # Set up fine-tuning
    training_setup = setup_finetuning(
        model,
        learning_rate=5e-5,
        batch_size=2,  # Reduced from 4 to 2
        max_length=512,  # Reduced from 1024 to 512
        num_epochs=5,
        unfreeze_last_k_layers=4,
        unfreeze_lm_head=True,
        consistency_kl_weight=0.1,
        span_dropout_prob=0.3,
    )
    
    if training_setup is not None:
        # Start fine-tuning
        trained_model = train_model(
            training_setup,
            num_epochs=5,
            save_steps=5000,
            log_steps=50
        )
        print("🎉 Training completed successfully!")
    else:
        print("❌ Failed to set up training. Check data availability.")