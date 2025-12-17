# How Boundary Head Confidence is Calculated

## Overview
The `boundary_head` is a binary classifier that predicts whether the current token position is a good place to trigger a patch (boundary) or not.

## Architecture

### Definition (blt_adapter_model.py:429)
```python
self.boundary_head = nn.Linear(self.hidden_size, 2)
```
- Simple linear layer: `hidden_size` → 2 outputs
- Input: hidden state vector (size = `hidden_size`, e.g., 1536)
- Output: 2 logits `[logit_no_boundary, logit_boundary]`

## During Inference (blt_inference.py:514-517)

### Step 1: Get hidden state
```python
# Get the hidden state of the last generated token from global transformer
last_hidden = global_hidden_seq[:, -1, :]  # Shape: [1, hidden_size]
```

### Step 2: Forward pass through boundary_head
```python
boundary_logits = model.boundary_head(last_hidden)  # Shape: [1, 2]
# Output: [logit_no_boundary, logit_boundary]
# Example: [-2.5, -1.2] means:
#   - logit_no_boundary = -2.5 (lower = less likely to be "no boundary")
#   - logit_boundary = -1.2 (higher = more likely to be "boundary")
```

### Step 3: Convert logits to probabilities
```python
probs = torch.softmax(boundary_logits, dim=-1)  # Shape: [1, 2]
# Softmax ensures probs[0] + probs[1] = 1.0
# Example: if logits = [-2.5, -1.2], then:
#   probs[0] = exp(-2.5) / (exp(-2.5) + exp(-1.2)) ≈ 0.22 (22% no boundary)
#   probs[1] = exp(-1.2) / (exp(-2.5) + exp(-1.2)) ≈ 0.78 (78% boundary)
```

### Step 4: Extract confidence
```python
boundary_confidence = float(probs[0, 1].item())
# This is the probability that it's a boundary (probs[1])
# Range: 0.0 to 1.0
```

### Step 5: Compare with threshold
```python
if boundary_confidence >= boundary_threshold:  # default 0.65
    # Trigger a patch - use local decoder to refine the span
    is_new_node = True
else:
    # Continue with global transformer only
    is_new_node = False
```

## During Training (blt_adapter_model.py:558-562)

### Input
- `last_hidden_for_heads`: Hidden states from global transformer for all positions
- Shape: `[batch_size, sequence_length, hidden_size]`

### Forward pass
```python
logits = self.boundary_head(last_hidden_for_heads)  # [B, L, 2]
# For each position, outputs 2 logits
```

### Loss calculation
```python
boundary_targets = pos_mask.long()  # [B, L] - ground truth labels
# 0 = not a boundary, 1 = boundary (start/single token)

ce_loss = F.cross_entropy(
    logits[mask].view(-1, 2),      # Predicted logits
    boundary_targets[mask].view(-1),  # True labels
)
```

## What the Numbers Mean

### Example: boundary_confidence = 0.219727
- This means: **21.97% probability** that the current position is a boundary
- Since 0.219727 < 0.65 (threshold), **no patch is triggered**
- The model continues generating with the global transformer only

### Example: boundary_confidence = 0.004913
- This means: **0.49% probability** that the current position is a boundary
- Even lower confidence, so definitely no patch

### Why both are low?
- The boundary_head hasn't learned to output high confidence at boundaries
- Possible reasons:
  1. Not receiving proper gradients during training (was detached before)
  2. Training data doesn't have enough boundary examples
  3. Loss weight too low
  4. Model needs more training

## The Fix
- Now boundary_loss is included in the main loss (not detached)
- This should allow boundary_head to learn properly
- After retraining, boundary_confidence should be higher (closer to 1.0) at actual boundaries

