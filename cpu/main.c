#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "../valuation.h"
#include "combination.h"
#include <sys/time.h>

/*
This function creates two valuations from the textbook example.

Input: 
    Valuation **v1: pointer to the first valuation
    Valuation **v2: pointer to the second valuation
    
Output: 
    int: 0 on success, -1 on failure
*/
int textbook_example(Valuation **v1, Valuation **v2) {
    uint16_t domain1[2] = {0, 1};
    uint16_t target1[1] = {0};
    uint8_t tuples1[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double values1[4] = {0.7, 0.3, 0.4, 0.6};

    Valuation *new_v1 = create_valuation(4, 2, domain1, 1, target1, tuples1, values1);
    if (!new_v1) return -1;

    uint16_t domain2[2] = {1, 2};
    uint16_t target2[1] = {2};
    uint8_t tuples2[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double values2[4] = {0.1, 0.9, 0.8, 0.2};

    Valuation *new_v2 = create_valuation(4, 2, domain2, 1, target2, tuples2, values2);
    if (!new_v2) {
        free_valuation(new_v1);
        return -1;
    }

    *v1 = new_v1;
    *v2 = new_v2;

    return 0;
}

/*
The main function is the entry point of the program.

Input: 
    None
    
Output:
    int: 0 on success, -1 on failure
*/
int main() {
    Valuation *v1, *v2;
    uint16_t domain_size, overlap;
    uint8_t states_per_var;

    printf("--- Valuation Combination ---\n");
    
    overlap = 0;
    states_per_var = 2;
    printf("States per variable: %u\n\n", states_per_var);
    for (domain_size = 1; domain_size <= 12; ++domain_size) {
        printf("\n--- Domain size: %u ---\n", domain_size);
        if(generate_pair(domain_size, overlap, states_per_var, &v1, &v2)) {
            fprintf(stderr, "Failed to create valuations\n");
            return -1;
        }

        Valuation* dst = alloc_combined_valuation(v1, v2);
        if (!dst) {
            fprintf(stderr, "Failed to allocate combined valuation\n");
            return -1;
        }

        // Start the timer
        struct timeval start, end;
        gettimeofday(&start, NULL);

        // Combine the valuations
        int combine_status = combine_valuations(dst, v1, v2);

        gettimeofday(&end, NULL);
        long seconds = end.tv_sec - start.tv_sec;
        long microseconds = end.tv_usec - start.tv_usec;
        long total_microseconds = (seconds * 1000000) + microseconds;

        printf("Elapsed time: %ld microseconds\n", total_microseconds);

        if (combine_status != 0) {
            fprintf(stderr, "Failed to combine valuations\n");
            free_valuation(dst);
            return -1;
        }

        printf("Combined valuation size: %u\n", dst->size);
        // display_valuation(dst);

        // Free the allocated memory
        free_valuation(dst);
        free_valuation(v1);
        free_valuation(v2);
    }

    return 0;
}


// uint16_t domain[4] = {0, 1, 2, 3};
// uint16_t target[1] = {3};

// uint8_t states_per_var = 2;
// Valuation* test = auto_generate_valuation(4, domain, 1, target, states_per_var);
// if (!test) {
//     fprintf(stderr, "Failed to create valuation\n");
//     return -1;
// }
// printf("\n\n");
// display_valuation(test);
// printf("\n\n");
