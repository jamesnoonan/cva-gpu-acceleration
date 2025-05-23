# Conditional Valuation Algebras: Directionality and Efficient Inference (GPU Acceleration)
The code corresponding to the GPU acceleration section of the thesis: *Conditional Valuation Algebras: Directionality and Efficient Inference*.

- `valuation.h` includes Valuation data types and shared helper functions
- `cpu` folder includes a program to perform combinations of various sizes directly on the CPU
- `gpu` folder includes a program to perform combinations of various sizes on a GPU using CUDA
- `results/graph.py` is a helper script used to generate the graphs seen in the report from CSV data

### Building and Running
**CPU:**
```
cd cpu
make
../out/cpu_algorithm
```

**GPU:**
```
cd gpu
make
../out/gpu_algorithm
```
