#include <stdlib.h>
#include <stdint.h>
#include "combination.h"
#include "../valuation.h"

/*
This function combines two valuations on the CPU.

Input: 
    Valuation *dst: the destination valuation to store the result, should be pre-allocated
    Valuation *v1: the first valuation
    Valuation *v2: the second valuation
    
Output: 
    int: 0 on success, -1 on failure
*/
int combine_valuations(Valuation *dst, Valuation *v1, Valuation *v2)
{
    if (!dst) return -1;
    // Get two arrays with the indices of the matching variables
    uint32_t *v1_indices, *v2_indices;
    uint32_t common_var_size = get_common_vars(v1->domain, v1->domain_size,
                                            v2->domain, v2->domain_size,
                                            &v1_indices, &v2_indices);

    if (v1_indices == NULL || v2_indices == NULL) {
        free(v1_indices);
        free(v2_indices);
        return -1;
    }

    uint32_t count = 0;
    // Iterate over all tuples in v1 and v2
    for (uint32_t i = 0; i < v1->size; ++i) {
        for (uint32_t j = 0; j < v2->size; ++j) {
            // Check if the tuples are compatible
            if (v1->rows[i].incompatible || v2->rows[j].incompatible) continue;

            // If tuples are incompatible, skip
            int do_tuples_match = tuples_match(v1->rows[i].tuple, v2->rows[j].tuple, v1_indices, v2_indices, common_var_size);
            if (!do_tuples_match) continue;

            // Get the current valuation row
            ValuationRow *row = &dst->rows[count];

            // Merge the tuples and insert into row
            merge_tuples(
                row->tuple,
                v1->rows[i].tuple, v1->domain_size,
                v2->rows[j].tuple, v2->domain_size,
                v2_indices, common_var_size);
            
            // Perform the semiring multiplication on the values (here, it's just multiplication)
            row->value = v1->rows[i].value * v2->rows[j].value;
            row->incompatible = 0;

            // Increment the row count
            count++;
        }
    }

    // Trim the unused rows
    dst->size = count;

    free(v1_indices);
    free(v2_indices);
    return 0;
}
