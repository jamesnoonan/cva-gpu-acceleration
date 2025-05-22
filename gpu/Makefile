# Makefile
PROG = gpu_algorithm
OUTDIR = ../out
OUT = $(OUTDIR)/$(PROG)

# Source files
CU_SRCS = combination.cu main.cu
C_SRCS  = ../valuation.c
SRC     = $(CU_SRCS) $(C_SRCS)

# GPU architecture (Tesla V100 / Volta → SM 70)
SM          = 70
NVCCFLAGS   = -O3 -std=c++17                             \
              -gencode arch=compute_$(SM),code=sm_$(SM) \
              -gencode arch=compute_$(SM),code=compute_$(SM)

all: $(OUTDIR) $(OUT)

# Ensure output directory exists
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Build binary
$(OUT): $(SRC)
	nvcc $(NVCCFLAGS) -o $@ $^

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
