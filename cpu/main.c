#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../valuation.h"
#include "combination.h"

// Display the contents of a valuation
void display_valuation(const Valuation *v) {
    printf("<Valuation>\n         {");
    for (uint16_t j = 0; j < v->domain_size; ++j) {
        if (j < v->domain_size - 1)
            // convert intger to char starting at A
            printf("%c, ", 'A' + v->domain[j]);
            else
            printf("%c", 'A' + v->domain[j]);
    }
    printf("}\n");
    for (uint32_t i = 0; i < v->size; ++i) {
        printf("    [%u]: (", i);
        for (uint16_t j = 0; j < v->domain_size; ++j) {
            if (j < v->domain_size - 1)
                printf("%u, ", v->rows[i].tuple[j]);
            else
                printf("%u", v->rows[i].tuple[j]);
        }
        printf(") -> %f\n", v->rows[i].value);
    }
}

int main() {
    printf("\n\n");

    uint16_t *domain = (uint16_t *)malloc(2 * sizeof(uint16_t));
    if (domain != NULL) {
        domain[0] = 0;
        domain[1] = 1;
    }

    uint16_t *target = (uint16_t *)malloc(1 * sizeof(uint16_t));
    if (target != NULL) {
        target[0] = 0;
    }

    uint16_t domain_size = 2;
    uint16_t target_size = 1;
    uint32_t num_rows = 4;

    // Allocate the valuation
    Valuation *v = allocate_valuation(num_rows, domain, domain_size, target, target_size);
    if (!v) {
        fprintf(stderr, "Failed to allocate valuation.\n");
        return 1;
    }

    uint8_t tuples[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double values[4] = {0.1, 0.2, 0.3, 0.4};

    memcpy(v->rows->tuple, tuples, domain_size * num_rows);
    // Populate the valuation
    for (uint32_t i = 0; i < num_rows; ++i) v->rows[i].value = values[i];

    display_valuation(v);

    Valuation* v1 = v;
    Valuation* v2 = v;

    Valuation* dst = alloc_combined_valuation(v1, v2);
    if (!dst) {
        fprintf(stderr, "Failed to allocate combined valuation\n");
        return -1;
    }

    // Combine the valuations
    if (combine_valuations(dst, v1, v2) != 0) {
        fprintf(stderr, "Failed to combine valuations\n");
        free_valuation(dst);
        return -1;
    }

    display_valuation(dst);

    // Free the allocated memory
    free_valuation(dst);
    free_valuation(v1);
    // free_valuation(v2);

    return 0;
}
