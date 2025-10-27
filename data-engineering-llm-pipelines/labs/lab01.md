# Lab 01 — NumPy & Jupyter EDA Warm‑Up

**Focus Areas:** Python + NumPy quick refresh, Jupyter for EDA (Exploratory Data Analysis) & pipelines

---

## Outcomes

By the end of this lab, you will be able to:

1. Create and reshape NumPy arrays; explain when and why to use `reshape` vs. `ravel`/`flatten`.
2. Demonstrate vectorization and broadcasting to replace Python `for` loops.
3. Compare **runtime** and **memory footprint** of lists vs. NumPy arrays.
4. Use key Jupyter magics for EDA & reproducibility: `%matplotlib inline`, `%timeit`, `%%time`, `%env`, `%%capture`, and `autoreload`.
5. Produce a short, reproducible EDA narrative that includes figures, random seeds, and environment/version stamps.

---

## Prerequisites & Setup

- Python 3.13 with `numpy`, `pandas`, and `matplotlib` installed.  
- JupyterLab or VS Code with the Jupyter extension.
- Start a new notebook named: `week02_lab01.ipynb`.

**Notebook prologue cell (run first):**

```python
# Reproducibility & environment snapshot
import os, sys, platform, random
import numpy as np
import pandas as pd
import matplotlib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print({
    'python': sys.version.split()[0],
    'platform': platform.platform(),
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'matplotlib': matplotlib.__version__,
    'pid': os.getpid(),
})
```

Add a markdown cell above the prologue titled **"Week 01, Lab 01 — NumPy & Jupyter EDA Warm‑Up"** with your name/date.

---

## Part A — NumPy Quick Refresh

### A1. Arrays vs. Python lists (time & memory)

1. **Create comparable data:**

   ```python
   import sys, numpy as np
   N = 1_000_000
   py_list = list(range(N))
   np_array = np.arange(N)
   ```

2. **Memory footprint:**

   ```python
   # List memory: container + objects (rough estimate)
   list_container_bytes = sys.getsizeof(py_list)
   int_object_bytes = sys.getsizeof(0)  # per small int (implementation dependent)
   approx_list_bytes = list_container_bytes + N * int_object_bytes

   # NumPy memory: contiguous buffer
   numpy_bytes = np_array.nbytes

   print({'approx_list_bytes': approx_list_bytes, 'numpy_bytes': numpy_bytes})
   ```

3. **Runtime with `%timeit`:**
   - In a new cell run:

     ```python
     %timeit sum(py_list)
     %timeit np_array.sum()
     ```

4. **Checkpoint:** In a new markdown cell, write 1–2 sentences explaining **why** NumPy is faster/more memory‑efficient here (contiguity, vectorized C loops, fewer Python object overheads).

### A2. Shape, reshape, ravel, and flatten

1. Create a 1D vector and reshape:

   ```python
   x = np.arange(12)
   x2 = x.reshape(3, 4)
   x3 = x.reshape(2, 2, 3)
   x_ravel = x2.ravel()      # view if possible
   x_flat = x2.flatten()     # always copy
   print(x2.shape, x3.shape, x_ravel.base is x, x_flat.base is x)
   ```

2. **Task:** Prove to yourself which operations return **views** vs **copies**: mutate an element in `x2` and observe `x`, `x_ravel`, and `x_flat`.

3. **Checkpoint:** Note practical guidance: prefer `reshape`/`ravel` for performance when you don’t need an independent copy.

### A3. Broadcasting & vectorization

1. **Broadcasting scaler + vector:**

   ```python
   v = np.arange(5)
   v_plus = v + 10
   v_scaled = v * 2
   ```

2. **Broadcasting 2D + 1D:**

   ```python
   M = np.arange(12).reshape(3,4)
   col = np.array([1, 2, 3]).reshape(3,1)
   M2 = M + col  # adds [1,2,3] to each row
   ```

3. **Loop vs vectorized timing:**

   ```python
   big = np.random.rand(2_000_000)

   def py_loop_square(arr):
       out = [0.0]*len(arr)
       for i, val in enumerate(arr):
           out[i] = val*val
       return out

   %timeit py_loop_square(big)
   %timeit big*big
   ```

4. **Checkpoint:** In a new markdown cell, record the speedup factor you observe (rough order‑of‑magnitude is fine).

---

## Part B — Jupyter for EDA & Pipelines

### B1. Jupyter magics you’ll actually use

Add a new code cell and experiment with these magics (run each and observe output):

```python
# 1) Inline plotting for reports/notebooks
%matplotlib inline
```

```python
%%time
# 2) Timing
import time
_ = [time.sleep(0.001) for _ in range(200)]
```

```python
# 3) Micro-benchmarks (repeat/average)
import numpy as np
arr = np.random.rand(1_000_00)
%timeit arr.mean()
```

```python
# 4) Environment variables for pipelines
%env DATA_DIR=./data
```

```python
%%capture cap
# 5) Capture noisy cell output (useful when logging)
print('This will be captured, not printed.')

# Verify capture worked
print("If you see this, capture is working (the above print was captured)")
```

```python
# 6) Autoreload during iterative development
%load_ext autoreload
%autoreload 2
```

> **Note:** `%%time` measures a single run (wall & CPU time). `%timeit` runs multiple times and reports a stable average — better for micro‑benchmarks.

### B2. Mini‑EDA narrative with reproducibility

1. **Create a small synthetic dataset:**

   ```python
   import pandas as pd, numpy as np
   rng = np.random.default_rng(SEED)
   n = 500
   df = pd.DataFrame({
       'user_id': np.arange(n),
       'age': rng.integers(18, 70, size=n),
       'country': rng.choice(['US', 'SG', 'DE', 'BR', 'IN'], size=n, p=[0.35,0.15,0.2,0.15,0.15]),
       'sessions': rng.poisson(3, size=n),
       'avg_session_sec': rng.normal(300, 50, size=n).clip(30, 1200)
   })
   df.head()
   ```

2. **Quick profile (no external libs):**

   ```python
   df.info()
   df.describe(numeric_only=True).T
   df['country'].value_counts(normalize=True).round(3)
   ```

3. **Plot distributions:**

   ```python
   import matplotlib.pyplot as plt
   df['age'].hist(bins=20)
   plt.title('Age Distribution')
   plt.show()

   df.plot.scatter(x='sessions', y='avg_session_sec', alpha=0.3)
   plt.title('Sessions vs Avg Session Seconds')
   plt.show()
   ```

4. **Reproducibility stamp:** In a new markdown cell, add a markdown bullet list noting **seed**, **versions**, and **DATA_DIR** env var

### B3. Export artifacts

```python
# Persist CSV and a compact Parquet for downstream steps
import os
os.makedirs(os.getenv('DATA_DIR', './data'), exist_ok=True)

out_csv = os.path.join(os.environ['DATA_DIR'], 'mini_eda_users.csv')
out_parquet = os.path.join(os.environ['DATA_DIR'], 'mini_eda_users.parquet')

%time df.to_csv(out_csv, index=False)
%time df.to_parquet(out_parquet, index=False)

out_csv, out_parquet
```

<!-- Add a markdown cell: **"Artifact paths"** and paste the two output paths. These will be used in later labs. -->

---

## Part C — Wrap‑Up

- In a final markdown cell, answer:
  1. When would you *prefer* a view (`ravel`, `reshape`) over a copy (`flatten`) and why?
  2. What’s an example where pure‑Python loops might still be acceptable?
  3. Which Jupyter magic would you use to: (a) benchmark two approaches, (b) hide verbose output, (c) ensure figures render inline in exported HTML?

---

- **Final thoughts:**
  - Confusion between `ravel` vs `flatten` (view vs copy) — explore by mutating data and observe.
  - `%time` vs `%timeit` — remember that `%timeit` is averaged.
  - Memory estimates for lists are approximate; understand object overhead vs contiguous buffers.

---

## Solution Snippets (reference)

**Why NumPy faster?**

- Contiguous memory + vectorized C/Fortran loops reduce Python interpreter overhead; fewer allocations; better CPU cache locality.

**View vs copy demo:**

```python
x = np.arange(6)
x2 = x.reshape(2,3)
r = x2.ravel()
f = x2.flatten()

x2[0,0] = 999
assert x[0] == 999 and r[0] == 999    # view tracks source
assert f[0] != 999                     # copy is independent
```

**Magics mapping:**

- Benchmark: `%timeit` (and `%%time` for single‑run cells)
- Hide output: `%%capture`
- Inline plots: `%matplotlib inline`
- Re-run code after edits to imported modules: `%load_ext autoreload; %autoreload 2`

**Speedup expectation:** On typical laptops, `np_array.sum()` ≫ `sum(py_list)`; vectorized square vs loop often yields **10–100×** depending on hardware.
