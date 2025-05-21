# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c11

# Directories
SRC_DIR = cpu
OUT_DIR = out

# Source files
SRCS = $(wildcard $(SRC_DIR)/*.c)
EXTRA_SRCS = valuation.c

# Object files
CPU_OBJS = $(patsubst $(SRC_DIR)/%.c,$(OUT_DIR)/%.o,$(SRCS))
EXTRA_OBJS = $(patsubst %.c,$(OUT_DIR)/%.o,$(EXTRA_SRCS))
OBJS = $(CPU_OBJS) $(EXTRA_OBJS)

# Output executable
TARGET = $(OUT_DIR)/cpu_algorithm

# Default target
all: $(TARGET)

# Rule to build object files from cpu/
$(OUT_DIR)/%.o: $(SRC_DIR)/%.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to build object files from top-level source (e.g., valuation.c)
$(OUT_DIR)/%.o: %.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Ensure output directory exists
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Clean build artifacts
clean:
	rm -rf $(OUT_DIR)

.PHONY: all clean
