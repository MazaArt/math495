# math495
Final Project - Artem &amp; Tyler

## Quick Start:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download MATH Dataset

```bash
# Example: only train split, keep level1+level2, cap at 4k examples
python3 download_math_dataset.py \
  --split train \
  --levels level1,level2 \
  --max-examples 4000
```


### 3. Train the Model

```bash
# Default: use MathQA + any downloaded MATH subset
python3 train.py

# Example: only MATH, levels 1-3, cap total examples to 5k
python3 train.py \
  --no-mathqa \
  --math-levels level1,level2,level3 \
  --max-examples 5000

# Example: include both datasets but limit each to 2k examples
python3 train.py --max-examples-per-dataset 2000
```

### 4. Test the Model

```bash
python3 inference.py
```
