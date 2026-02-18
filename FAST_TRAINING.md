# ⚡ Fast Training Configuration Guide

Speed optimizations applied to reduce Kaggle training time from **6 hours → 2-3 hours**

## 🚀 What Changed

### Optimizations Applied

| Setting | Before | After | Impact |
|---------|--------|-------|--------|
| **Epochs** | 3 | 2 | -33% time |
| **Batch Size** | 2 | 4 | +100% throughput |
| **Effective Batch** | 8 | 16 | Faster convergence |
| **Learning Rate** | 2e-4 | 3e-4 | Faster convergence |
| **Validation Split** | 5% | 2% | Less eval overhead |
| **Eval Frequency** | 100 steps | 500 steps | Less interruption |
| **Save Frequency** | 100 steps | 500 steps | Less I/O |
| **Checkpoints Kept** | 2 | 1 | Faster saves |

**Result**: Training time cut by ~50% with minimal quality impact

## 🎯 Current Config Performance

| Platform | GPU | Expected Time |
|----------|-----|---------------|
| Kaggle | T4 × 2 | **2-3 hours** ⚡ |
| Colab Free | T4 | 3-4 hours |
| Colab Pro | V100 | 1-2 hours |
| RunPod | RTX 4090 | 45-90 min |

## ⚙️ Even Faster? Try These

### Option 1: Train for 1 Epoch (Fast Test)

```yaml
training:
  num_train_epochs: 1  # Quick training for testing
```
**Time**: 1-1.5 hours on Kaggle  
**Quality**: Lower but usable for testing

### Option 2: Use Smaller Max Sequence Length

```yaml
model:
  max_seq_length: 1024  # Down from 2048
```
**Time**: 30-40% faster  
**Trade-off**: Can't handle very long conversations

### Option 3: Even Larger Batches (if you have VRAM)

```yaml
training:
  per_device_train_batch_size: 8  # Double it
```
**Time**: 20-30% faster  
**Requires**: 16GB+ VRAM (Colab Pro, RunPod with 3090/4090)

### Option 4: Disable Validation

```yaml
training:
  evaluation_strategy: "no"  # Skip validation entirely
```
**Time**: Save 5-10%  
**Trade-off**: No quality monitoring during training

### Option 5: Use Subset for Quick Test

```yaml
dataset:
  max_samples: 5000  # Train on subset only
```
**Time**: ~30 minutes on Kaggle  
**Use case**: Quick testing before full training

## 🔥 Ultra-Fast Config (30 minutes on Kaggle)

For rapid experimentation:

```yaml
model:
  max_seq_length: 512

dataset:
  max_samples: 5000
  validation_split: 0

training:
  num_train_epochs: 1
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  evaluation_strategy: "no"
  save_steps: 1000
  logging_steps: 50
```

**Warning**: Lower quality, use only for testing/debugging

## 📊 Quality vs Speed Trade-offs

| Config | Time | Quality | When to Use |
|--------|------|---------|-------------|
| **Default (current)** | 2-3h | ⭐⭐⭐⭐ | Recommended |
| 1 epoch | 1-1.5h | ⭐⭐⭐ | Quick iteration |
| Subset (5k) | 30min | ⭐⭐ | Testing only |
| Ultra-fast | 20min | ⭐ | Debugging |
| Original (3 epochs) | 4-6h | ⭐⭐⭐⭐⭐ | Maximum quality |

## 💡 Best Practice Workflow

1. **First run**: Use ultra-fast config to test pipeline (30 min)
2. **Iteration**: Use 1 epoch config to tune (1.5 hours)
3. **Final training**: Use current optimized config (2-3 hours)
4. **Production**: Original 3 epochs if needed (4-6 hours)

## 🎯 Recommended Settings by Use Case

### For Learning/Testing
```yaml
num_train_epochs: 1
max_samples: 5000
```

### For Personal Use (Current - Recommended)
```yaml
num_train_epochs: 2
per_device_train_batch_size: 4
# Already configured!
```

### For Best Quality
```yaml
num_train_epochs: 3
learning_rate: 2.0e-4
validation_split: 0.05
```

### For Deployment/Production
```yaml
num_train_epochs: 5
per_device_train_batch_size: 2
learning_rate: 1.5e-4
# Train longer for stability
```

## 🚨 Important Notes

- **Batch size**: Increase only if you have VRAM (will crash if too large)
- **Learning rate**: Higher = faster but less stable
- **Epochs**: Fewer epochs = faster but may underfit
- **Validation**: Skip only if you don't need quality metrics

## ✅ What's Already Optimized

The current `training_config.yaml` is already optimized for:
- ✅ Speed (2-3 hours on Kaggle)
- ✅ Quality (still good results)
- ✅ Memory efficiency (fits on T4)
- ✅ Stability (tested settings)

**You're good to go!** Just run `python train.py`

---

**Current config delivers the best balance of speed and quality for most users.**
