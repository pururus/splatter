# Batch Size (num_pairs) Compatibility Analysis

## Summary
**Current Status**: ❌ **NOT COMPATIBLE** with batch_size > 1

The codebase is currently hardcoded for batch_size = 1. Supporting num_pairs > 1 would require significant refactoring.

---

## Issues Found

### 1. **Direct `.item()` calls on batch tensors** (CRITICAL)
**Location**: [trainer_fragGS.py](trainer_fragGS.py#L436-L437)

```python
# Line 436-437
query_tracks_2d = self.video_3d_flow.load_target_tracks(ids1.item(), [ids1.item()])
target_tracks = self.video_3d_flow.load_target_tracks(ids1.item(), [ids2.item()], dim=0)
```

**Issue**: `.item()` extracts a single value. With batch_size > 1, `ids1` and `ids2` would be tensors of shape `[batch_size]`, and `.item()` would fail.

**Impact**: Training would crash immediately.

---

### 2. **Hard-coded batch index [0]** (CRITICAL)
Multiple locations assume batch_size = 1:

- **Line 539**: `predicted_track_gs = render_results['track_gs'].permute(0,2,3,1)  # [1, h, w, 3]`
- **Line 546**: `masks_flatten[0, query_pixels[:, 1], query_pixels[:, 0]] = 1.0`
- **Line 565**: `pred_rgb1 = render_results['rgb'][0].permute(1,2,0).reshape(...)`
- **Line 596**: `depth = render_results['depth'][0].permute(1,2,0)`
- **Line 597**: `gt_depth = self.gt_depths[ids1][0,...,None]`

**Issue**: With batch_size > 1, results would have batch dimension, but code only processes `[0]`.

**Impact**: Loss computation would fail or only use first batch item.

---

### 3. **Loss computation assumes single pair**
**Location**: [trainer_fragGS.py](trainer_fragGS.py#L421-L600)

The entire `compute_all_losses()` function:
- Computes frame intervals as scalars (line 425)
- Loads target tracks for single frame pair (line 437)
- Processes render results assuming batch_size=1

**Example**:
```python
# Line 425 - works only with scalar ids
frame_intervals = torch.abs(ids2 - ids1).float()
w_interval = torch.exp(-2 * frame_intervals / self.num_imgs)

# Should be:
# frame_intervals = torch.abs(ids2 - ids1).float()  # [batch_size]
# w_interval = torch.exp(-2 * frame_intervals / self.num_imgs).unsqueeze(-1)  # [batch_size, 1]
```

---

### 4. **Dataset returns scalar indices**
**Location**: [loaders/gs_data2.py](loaders/gs_data2.py#L75-L85)

```python
def __getitem__(self, idx):
    # ...
    data = {'ids1': id1,      # scalar int
            'ids2': id2,      # scalar int
            'gt_rgb1': gt_rgb1,  # [H*W, 3]
            'weights': weights,  # scalar
            }
    return data
```

**DataLoader Behavior**: When batch_size > 1, DataLoader will stack scalars into tensors:
- `ids1` becomes shape `[batch_size]` ✓ This is correct
- Problem: Code doesn't handle this properly

---

### 5. **Model forward pass design**
**Location**: [trainer_fragGS.py](trainer_fragGS.py#L422-L424)

```python
render_dict = self.gs_atlases_model.forward(ids1)
render_dict2 = self.gs_atlases_model.forward(ids2)
```

**Issue**: `forward()` expects single frame index, not batch of indices.

Would need to handle model forward for multiple frames or loop over batch.

---

### 6. **Video 3D Flow interface**
**Location**: [trainer_fragGS.py](trainer_fragGS.py#L436-L437)

```python
self.video_3d_flow.load_target_tracks(ids1.item(), [ids2.item()], dim=0)
```

**Issue**: `load_target_tracks()` is designed for single frame pairs, not batches.

Would require modifying the video flow interface.

---

## What Would Need to Change

### ✅ **Straightforward fixes** (Low effort)
1. Handle `.item()` calls by looping over batch
2. Update indexing from `[0]` to process all batch items
3. Aggregate losses from multiple batch items

### ⚠️ **Moderate complexity** (Medium effort)
1. Modify `compute_all_losses()` to return per-batch-item losses
2. Update depth loss computation for batch processing
3. Handle frame_intervals and w_interval as vectors

### 🔴 **Major refactoring needed** (High effort)
1. Modify model's `forward()` to handle batch of frame indices
2. Update `video_3d_flow.load_target_tracks()` to batch multiple frame pairs
3. Modify renderer to handle multiple frame pairs efficiently
4. Update optimizer update logic for per-image optimization

---

## Recommendations

### Option 1: Keep batch_size = 1 (Recommended)
- **Effort**: 0%
- **Current config already uses this**: `--num_pairs 1` in config files
- The code is optimized for single pairs

### Option 2: Support batch_size > 1
- **Effort**: 30-40% of total code refactoring
- **Benefit**: Potentially faster data loading, but not clear if algorithmic benefit exists
- **Risk**: High likelihood of bugs with complex loss interactions

### Option 3: Hybrid approach
- Keep single pair rendering per loss computation
- Use DataLoader batching only for data loading efficiency
- Loop internally: `for i in range(batch_size): compute_loss(batch[i])`
- **Effort**: 10-15%
- **Trade-off**: Minimal code changes, but loses some batching benefits

---

## Testing Recommendation

Before making changes, test with batch_size = 2 to identify all breaking points:

```bash
python train.py --num_pairs 2
```

Expected errors will show all `.item()` and indexing issues.

---

## Current Safe Configuration
```yaml
--num_pairs 1          # Keep this at 1
--num_workers 4        # Safe to increase
--num_pts 256          # Each pair samples 256 points
```

The current setup processes one image pair per iteration, which is the intended design.
