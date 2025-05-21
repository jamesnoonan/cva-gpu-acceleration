#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "valuation.h"

/*
This function allocates memory for a single Valuation

Input: 
    uint32_t size: the number of rows in the valuation
    uint32_t domain_size: the number of variables in the domain
    uint32_t target_size: the number of variables in the target domain
    
Output: an allocated Valuation struct or NULL if allocation fails
*/
Valuation *allocate_valuation(
    uint32_t size,
    uint16_t *domain,
    uint16_t domain_size,
    uint16_t *target,
    uint16_t target_size)
{
    // Allocate memory for Valuation struct
    Valuation *v = (Valuation *) malloc(sizeof(Valuation));
    if (!v) return NULL;

    v->size = size;

    // Allocate block of memory for all ValuationRow structs
    v->rows = (ValuationRow *) malloc(sizeof(ValuationRow) * size);
    if (!v->rows) {
        free(v);
        return NULL;
    }

    // Allocate block of memory for tuples
    uint8_t *tuple_block = (uint8_t *) malloc(size * domain_size);
    if (!tuple_block) {
        free(v->rows);
        free(v);
        return NULL;
    }

    // Initialize each ValuationRow with default values and assign tuple pointer
    for (uint32_t i = 0; i < size; ++i) {
        v->rows[i].value = 0.0;
        v->rows[i].tuple = tuple_block + i * domain_size;
        v->rows[i].incompatible = 0;
    }

    v->domain = domain;
    v->domain_size = domain_size;

    v->target = target;
    v->target_size = target_size;

    return v;
}

Valuation* alloc_combined_valuation(Valuation *v1, Valuation *v2) {
    uint32_t size = v1->size * v2->size;
    uint16_t domain_size, target_size;
    uint16_t *domain, *target;

    domain = union_domain(
        v1->domain, v1->domain_size,
        v2->domain, v2->domain_size,
        &domain_size);
    target = union_domain(
        v1->target, v1->target_size,
        v2->target, v2->target_size,
        &target_size);

    // Allocate memory for the combined valuation
    Valuation *dst = allocate_valuation(size, domain, domain_size, target, target_size);

    return dst;
}

/*
This function frees the memory allocated for a Valuation struct
Input: 
    Valuation *v: the Valuation struct to free
*/
void free_valuation(Valuation *v) {
    if (!v) return;

    // Free the tuple block only once
    if (v->rows && v->rows[0].tuple) {
        free(v->rows[0].tuple);
    }

    // Free the ValuationRows
    free(v->rows);

    // Free the domain and target domain
    free(v->domain);
    free(v->target);

    // Free the valuation itself
    free(v);
}

uint16_t *union_domain(
    const uint16_t *dom1, uint16_t n1,
    const uint16_t *dom2, uint16_t n2,
    uint16_t *out_size)
{
    uint16_t *temp = (uint16_t *) malloc((n1 + n2) * sizeof(uint16_t));
    if (!temp) return NULL;
    memcpy(temp, dom1, n1);
    uint32_t count = n1;

    for (uint32_t i = 0; i < n2; ++i)
    {
        int exists = 0;
        for (uint32_t j = 0; j < n1; ++j)
        {
            if (dom2[i] == dom1[j])
            {
                exists = 1;
                break;
            }
        }
        if (!exists) temp[count++] = dom2[i];
    }

    uint16_t *result = (uint16_t *) malloc(count * sizeof(uint16_t));
    memcpy(result, temp, count);
    free(temp);
    
    *out_size = count;
    return result;
}

uint16_t *intersection_domain(
    const uint16_t *dom1, uint16_t n1,
    const uint16_t *dom2, uint16_t n2,
    uint16_t *out_size)
{
    uint16_t *temp = (uint16_t *) malloc((n1 < n2 ? n1 : n2) * sizeof(uint16_t));
    uint32_t count = 0;

    for (uint32_t i = 0; i < n1; ++i)
    {
        for (uint32_t j = 0; j < n2; ++j)
        {
            if (dom1[i] == dom2[j])
            {
                int already_added = 0;
                for (uint32_t k = 0; k < count; ++k)
                {
                    if (temp[k] == dom1[i])
                    {
                        already_added = 1;
                        break;
                    }
                }

                if (!already_added)
                {
                    temp[count++] = dom1[i];
                }
                break;
            }
        }
    }

    uint16_t *result = (uint16_t *) malloc(count * sizeof(uint16_t));
    memcpy(result, temp, count);
    free(temp);

    *out_size = count;
    return result;
}

uint32_t get_common_vars(const uint16_t *domain1, uint16_t size1,
                         const uint16_t *domain2, uint16_t size2,
                         uint32_t **v1_indices, uint32_t **v2_indices) {
    uint32_t max_common = size1 < size2 ? size1 : size2;
    *v1_indices = malloc(max_common * sizeof(uint32_t));
    *v2_indices = malloc(max_common * sizeof(uint32_t));

    if (*v1_indices == NULL || *v2_indices == NULL) {
        free(*v1_indices);
        free(*v2_indices);
        *v1_indices = *v2_indices = NULL;
        return 0;
    }

    uint32_t count = 0;
    for (uint32_t i = 0; i < size2; i++) {
        for (uint32_t j = 0; j < size1; j++) {
            if (domain2[i] == domain1[j]) {
                (*v2_indices)[count] = i;
                (*v1_indices)[count] = j;
                count++;
                break;
            }
        }
    }

    return count;
}

int tuples_match(const uint8_t *tuple1, const uint8_t *tuple2,
                  const uint32_t *v1_indices, const uint32_t *v2_indices,
                  uint32_t num_indices)
{
    for (uint32_t i = 0; i < num_indices; i++)
    {
        if (tuple1[v1_indices[i]] != tuple2[v2_indices[i]])
        {
            return 0;
        }
    }
    return 1;
}

void merge_tuples(
    uint8_t *dst,
    const uint8_t *tuple1, uint32_t size1,
    const uint8_t *tuple2, uint32_t size2,
    const uint32_t *v2_indices, uint32_t common_size)
    {
    uint32_t idx = 0;

    // Copy all from tuple1
    for (uint32_t i = 0; i < size1; i++) {
        dst[idx++] = tuple1[i];
    }

    uint32_t pointer = 0; 
    for (uint32_t j = 0; j < size2; j++) {
        if (pointer < common_size && v2_indices[pointer] == j) {
            pointer++;
            continue;
        }
        dst[idx++] = tuple2[j];
    }

}