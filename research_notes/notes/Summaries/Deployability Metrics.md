# Deployability Metrics — Current State, Justification & Refinement Path

*Written 2026-04-12. Covers the four deployability metrics collected in `code/run_full_baseline_v3.py`.*

---

## Overview

All four metrics are collected inside `_build_result()` (line 116) and `estimate_energy()` (line 100) in `run_full_baseline_v3.py`. The same structure is reused in `run_full_baseline_lora.py`.

The core limitation: **only latency and memory are directly observed; energy and CO2 are model-based estimates.** All four metrics are applied uniformly across all models, so relative rankings are valid even where absolute values carry uncertainty.

---

## 1. GPU Memory

### What it currently is
```python
gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
# _build_result(), line 119
```
`torch.cuda.max_memory_allocated()` returns the peak number of bytes held by live tensors (PyTorch's internal allocator tracking). It resets per-process on first call and accumulates across the full run (model load + inference).

### Justification
- Captures relative model footprint reliably — a model that allocated 3 GB of tensors is lighter than one that allocated 16 GB.
- Consistent across all 10 models: same call, same conditions, same GPU.
- Sufficient for the thesis's core claim (LG-1B ~3 GB < LLaMA-8B ~16 GB).

### Known limitations
- **Excludes CUDA context overhead** (~200–500 MB allocated by the CUDA driver before any model loads — present for every process but invisible to PyTorch).
- **Excludes memory fragmentation** (PyTorch allocates memory slabs; unused space within a slab is not counted).
- **Underestimates actual VRAM by ~10–30%** compared to what `nvidia-smi` reports for the same process.

### How to refine
Replace the one line with `pynvml`, which reads the same NVML data that `nvidia-smi` uses:

```python
import pynvml

def get_gpu_memory_mb() -> float:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024**2)
    except Exception:
        return torch.cuda.max_memory_allocated() / (1024**2)  # fallback
```

Package: `pip install nvidia-ml-py3` (likely already on the Alan cluster).  
**Effort: ~10 min. Worth doing if rerunning Phase 1 for any reason.**

---

## 2. Inference Latency

### What it currently is
```python
ts = time.time()
with torch.no_grad():
    output = model(**inputs)
times.append((time.time() - ts) * 1000)  # ms
# averaged over all samples in _build_result(), line 129
```
Wall-clock time per sample (batch size = 1), Python `time.time()` only, no CUDA synchronization.

### Justification
- Batch size = 1 reflects real deployment: Shareish would moderate one post at a time on incoming requests.
- Consistent across all models — relative ordering is valid.
- CPU models (Detoxify variants, KoalaAI) are fully accurate since their operations are synchronous.

### Known limitations
- **CUDA operations are asynchronous.** When `model(**inputs)` returns on the CPU, the GPU kernel may not have finished. `time.time()` stops before the GPU is done, so **GPU model latency is systematically underestimated.**
- The true latency can be 1.5–3× higher than reported for large generative models (LLaMA-Guard, Mistral) because they run multi-step generation loops.

### How to refine
Add `torch.cuda.synchronize()` before stopping the timer in every `evaluate_*` function that uses a GPU model:

```python
ts = time.time()
with torch.no_grad():
    output = model(**inputs)
torch.cuda.synchronize()          # wait for GPU to finish
times.append((time.time() - ts) * 1000)
```

Apply to: `evaluate_ethicaleye`, `evaluate_koalaai`, `evaluate_citizenlab`, `evaluate_llama_guard`, `evaluate_shieldgemma`, `evaluate_mistral`. Not needed for `evaluate_detoxify` (CPU).  
**Effort: ~10 min. Highest correctness gain for least effort.**

---

## 3. Energy Consumption

### What it currently is
```python
def estimate_energy(gpu_memory_mb, runtime_s, cpu_pct, gpu_available):
    h = runtime_s / 3600
    if gpu_available and gpu_memory_mb > 0:
        w = 50 if gpu_memory_mb < 4000 else (150 if gpu_memory_mb < 10000 else 250)
        gpu_kwh = (w / 1000) * h
    cpu_kwh = (95 * cpu_pct / 100 / 1000) * h
    return gpu_kwh + cpu_kwh, (gpu_kwh + cpu_kwh) * 0.475
# run_full_baseline_v3.py, line 100
```
A three-tier wattage lookup (50 W / 150 W / 250 W) based on memory footprint, multiplied by wall-clock runtime. CPU assumed 95 W TDP × measured cpu_percent.

### Justification
- Captures the order-of-magnitude difference between a 300 MB CPU model and a 16 GB GPU model.
- Adequate for the comparative framing: "LLaMA-8B consumes ~N× more energy than Detoxify per run."

### Known limitations
- **The A5000 (cluster GPU) has a 230 W TDP.** The lookup assigns 150 W to models using 4–10 GB — plausible for idle-to-moderate load, but not validated.
- **Wattage is not constant during inference.** A model may draw 20 W during tokenization and 200 W during the forward pass. The lookup uses a flat average.
- **Does not read actual power draw.** `nvidia-smi --query-gpu=power.draw` provides instantaneous wattage; polling it during inference would replace the lookup entirely.
- **Absolute uncertainty: ±30–50%.** Relative ordering is meaningful; absolute kWh values are not citable as measured results.

### How to refine
Use a background thread polling `pynvml` for real power readings during inference:

```python
import threading, pynvml

class GPUPowerMonitor:
    def __init__(self, interval_s=0.5):
        self.interval = interval_s
        self._readings = []
        self._stop = threading.Event()

    def start(self):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._readings.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
            self._readings.append(mw / 1000)
            self._stop.wait(self.interval)

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return sum(self._readings) / len(self._readings) if self._readings else 0.0
```

Usage (wraps each inference loop):
```python
monitor = GPUPowerMonitor(interval_s=0.5)
monitor.start()
t0 = time.time()
# ... existing inference loop ...
runtime_s = time.time() - t0
avg_watts = monitor.stop()
energy_kwh = (avg_watts / 1000) * (runtime_s / 3600)
```

**Effort: ~1–2h including refactoring `_build_result`. Only worth doing if rerunning all experiments.**

---

## 4. CO2 Equivalent

### What it currently is
```python
return total, total * 0.475  # kWh, kg CO2
```
A fixed carbon intensity factor of **475 gCO2/kWh** applied to the estimated energy.

### Justification
- Adequate for relative comparisons (same factor applied to all models).
- Order-of-magnitude plausible for a European grid mix.

### Known limitations
- **The factor is wrong for Belgium.** Belgium's actual grid carbon intensity in 2023–2024 is approximately **185–233 gCO2/kWh** (IEA, 2024), owing to its large nuclear fleet. The current factor `0.475` is roughly the European average and **overestimates Belgian CO2 by ~2×**.
- Carbon intensity varies by hour of day and season (wind/solar availability). A run at 3am on a windy night may emit half the CO2 of one at 6pm in winter.

### How to refine
**Minimum fix (thesis-appropriate):** replace `0.475` with `0.233` and cite IEA Belgium 2023.

```python
CO2_FACTOR = 0.233  # kg CO2/kWh — IEA Belgium 2023 grid average
return total, total * CO2_FACTOR
```

**Proper fix:** use [CodeCarbon](https://github.com/mlco2/codecarbon), which queries real-time grid intensity from Electricity Maps API and integrates over the run:

```python
from codecarbon import EmissionsTracker
tracker = EmissionsTracker(country_iso_code="BEL")
tracker.start()
# ... inference loop ...
emissions_kg = tracker.stop()  # CO2 in kg, real-time grid data
```

CodeCarbon is installable on the Alan cluster (`pip install codecarbon`) and requires no changes to the inference logic.  
**Effort: ~30 min for the minimum fix. ~1h for CodeCarbon integration.**

---

## Summary Table

| Metric | Method | Absolute reliability | Relative reliability | Quick fix | Full fix |
|--------|--------|:-------------------:|:--------------------:|-----------|----------|
| GPU memory | `max_memory_allocated()` | Moderate (−10–30%) | High | `pynvml` (10 min) | — |
| Latency | `time.time()`, no sync | Moderate for GPU models | High | `cuda.synchronize()` (10 min) | — |
| Energy | Wattage lookup × runtime | Low (±30–50%) | Moderate | — | Power polling thread (1–2h) |
| CO2 | Energy × 0.475 | Low (2× overestimate) | Moderate | Fix factor to 0.233 (5 min) | CodeCarbon (1h) |

**Thesis defense framing:** *"Deployability metrics serve as comparative indicators of relative resource requirements, not precise hardware measurements. GPU memory and latency are directly observed; energy and CO2 are proxy estimates based on measured runtime and a wattage model. The relative ordering of models is robust to measurement uncertainty."*
