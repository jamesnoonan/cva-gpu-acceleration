#include <stdio.h>
#include <cuda_runtime.h>
#include "combination.cuh"
#include "../valuation.h"

#define HANDLE_ERROR(e)                                                       \
  do {                                                                        \
    if ((e) != cudaSuccess) {                                                 \
      fprintf(stderr, "[CUDA Error] %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(e));                                         \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)


// Kernel to fix the tuple pointers on the device
__global__ void fix_tuple_pointers(ValuationRow *rows, uint8_t *tuple_block, uint32_t size, uint16_t domain_size) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        rows[i].tuple = tuple_block + i * domain_size;
    }
}

__host__ Valuation* transfer_valuation_to_gpu(Valuation *host_val) {
    Valuation *device_val = NULL;
    ValuationRow *device_rows;
    uint8_t *device_tuple_block;
    uint16_t *device_domain, *device_target;

    // Allocate device memory for Valuation struct and its members
    HANDLE_ERROR(cudaMalloc((void**) &device_val, sizeof(Valuation))); // Allocate device memory for Valuation struct
    HANDLE_ERROR(cudaMalloc((void**) &device_rows, sizeof(ValuationRow) * host_val->size)); // Allocate device memory for rows
    HANDLE_ERROR(cudaMalloc((void**) &device_tuple_block, host_val->size * host_val->domain_size)); // Allocate device memory for tuple block
    HANDLE_ERROR(cudaMalloc((void**) &device_domain, sizeof(uint16_t) * host_val->domain_size)); // Copy domain array
    HANDLE_ERROR(cudaMalloc((void**) &device_target, sizeof(uint16_t) * host_val->target_size));

    // Copy the data from host to device
    HANDLE_ERROR(cudaMemcpy(device_rows, host_val->rows, sizeof(ValuationRow) * host_val->size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(device_tuple_block,
               host_val->rows[0].tuple,
               host_val->size * host_val->domain_size,
               cudaMemcpyHostToDevice)); // Copy the entire tuple block from host

    HANDLE_ERROR(cudaMemcpy(device_domain, host_val->domain, sizeof(uint16_t) * host_val->domain_size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(device_target, host_val->target, sizeof(uint16_t) * host_val->target_size, cudaMemcpyHostToDevice)); // Copy target array

    // Launch kernel to fix the tuple pointers on the device
    uint32_t threads_per_block = 256;
    uint32_t blocks = (host_val->size + threads_per_block - 1) / threads_per_block;
    fix_tuple_pointers<<<blocks, threads_per_block>>>(device_rows, device_tuple_block, host_val->size, host_val->domain_size);

    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaGetLastError());

    // Set properties
    Valuation temp_val;
    temp_val.size = host_val->size;
    temp_val.domain_size = host_val->domain_size;
    temp_val.target_size = host_val->target_size;
    temp_val.rows = device_rows;
    temp_val.domain = device_domain;
    temp_val.target = device_target;

    HANDLE_ERROR(cudaMemcpy(device_val, &temp_val, sizeof(Valuation), cudaMemcpyHostToDevice));

    return device_val;
}

__host__ void transfer_valuation_from_gpu(Valuation *host_val, Valuation *device_val) {
    Valuation temp_device_val;

    HANDLE_ERROR(cudaMemcpy(&temp_device_val, device_val, sizeof(Valuation), cudaMemcpyDeviceToHost));

    ValuationRow *temp_device_rows = (ValuationRow*) malloc(sizeof(ValuationRow) * host_val->size);
    HANDLE_ERROR(cudaMemcpy(temp_device_rows,
                            temp_device_val.rows,
                            sizeof(ValuationRow) * host_val->size,
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMemcpy(host_val->rows[0].tuple,
                            temp_device_rows[0].tuple,
                            host_val->size * host_val->domain_size,
                            cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < host_val->size; ++i) {
        host_val->rows[i].value = temp_device_rows[i].value;
        host_val->rows[i].tuple = host_val->rows[0].tuple + i * host_val->domain_size;
        host_val->rows[i].incompatible = temp_device_rows[i].incompatible;
    }

    
    // Free device memory
    HANDLE_ERROR(cudaFree(temp_device_val.rows));
    HANDLE_ERROR(cudaFree(temp_device_val.domain));
    HANDLE_ERROR(cudaFree(temp_device_val.target));

    HANDLE_ERROR(cudaFree(temp_device_rows[0].tuple));
    HANDLE_ERROR(cudaFree(device_val));

    free(temp_device_rows);
}



__host__ int main() {
    Valuation *v1, *v2, *dst, *device_v1, *device_v2, *device_dst;
    uint16_t domain_size, overlap;
    uint8_t states_per_var;

    printf("--- Valuation Combination (GPU) ---\n");
    
    overlap = 0;
    states_per_var = 2;
    printf("States per variable: %u\n\n", states_per_var);
    
    uint16_t domain_sizes[] = {1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    int domain_size_count = sizeof(domain_sizes) / sizeof(domain_sizes[0]);

    printf("arg_domain_size,output_size,time,in_transfer,out_transfer,transfer_time\n");
    for (int dom_index = 0; dom_index < domain_size_count; ++dom_index) {
        domain_size = domain_sizes[dom_index];
        printf("%u,", domain_size);

        if(generate_pair(domain_size, overlap, states_per_var, &v1, &v2)) {
            fprintf(stderr, "Failed to create valuations\n");
            return -1;
        }

        dst = alloc_combined_valuation(v1, v2); // This could eventually be done on the GPU
        if (!dst) {
            fprintf(stderr, "Failed to allocate combined valuation\n");
            return -1;
        }
        
        cudaEvent_t start, stop;
        float to_milliseconds = 0, from_milliseconds = 0;

        // Create events
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        device_v1 = transfer_valuation_to_gpu(v1);
        device_v2 = transfer_valuation_to_gpu(v2);
        device_dst = transfer_valuation_to_gpu(dst);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&to_milliseconds, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        printf("%u,", dst->size);
        
        int combine_status = combine_valuations(device_dst, device_v1, device_v2, dst, v1, v2);
        
        if (combine_status != 0) {
            fprintf(stderr, "Failed to combine valuations\n");
            free_valuation(dst);
            return -1;
        }
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Transfer the combined valuation back to the host
        transfer_valuation_from_gpu(dst, device_dst);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&from_milliseconds, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("%.6f,", to_milliseconds);
        printf("%.6f,", from_milliseconds);
        printf("%.6f\n", to_milliseconds + from_milliseconds);

        // printf("Combined valuation size: %u\n", dst->size);
        // if (domain_size == 4) {
        //     display_valuation(dst);
        // }

        // Free the allocated memory
        free_valuation(dst);
        free_valuation(v1);
        free_valuation(v2);
    }

    return 0;
}