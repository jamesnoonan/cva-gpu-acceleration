#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "valuation.h"

/*
This function prints a valuation to stdout
Input: 
    Valuation *v: the valuation to display
Output:
    None
*/
void display_valuation(const Valuation *v) {
    printf("<Valuation>\n         {");
    for (uint16_t j = 0; j < v->domain_size; ++j) {
        // Convert the integer to a char starting at A for readability
        if (j < v->domain_size - 1)
            printf("%c, ", 'A' + v->domain[j]);
            else
            printf("%c", 'A' + v->domain[j]);
    }
    printf("}\n");
    // Print the rows
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

/*
This function allocates memory for a combination result. Domain and target will be allocated and set to the correct values.
Input: 
    Valuation *v1: the first valuation
    Valuation *v2: the second valuation
Output:
    Valuation *dst: the combined valuation
*/
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
This function creates a Valuation struct from the given parameters.
Input: 
    uint32_t size: the number of rows in the valuation
    uint16_t domain_size: the number of variables in the domain
    uint16_t domain[domain_size]: the variables in the domain
    uint16_t target_size: the number of variables in the target domain
    uint16_t target[target_size]: the variables in the target domain
    uint8_t tuples[size][domain_size]: the tuples for each row
    double values[size]: the semiring values for each row
Output:
    Valuation *v: the created Valuation struct
*/
Valuation *create_valuation(
    uint32_t size,
    uint16_t domain_size,
    uint16_t *domain,
    uint16_t target_size,
    uint16_t *target,
    uint8_t *tuples,
    double *values)
{
    uint16_t *new_domain = (uint16_t *) malloc(domain_size * sizeof(uint16_t));
    if (!new_domain) return NULL;

    uint16_t *new_target = (uint16_t *) malloc(target_size * sizeof(uint16_t));
    if (!new_target) {
        free(new_domain);
        return NULL;
    }

    memcpy(new_domain, domain, domain_size * sizeof(uint16_t));
    memcpy(new_target, target, target_size * sizeof(uint16_t));
    Valuation *v = allocate_valuation(size, new_domain, domain_size, new_target, target_size);

    if (!v) {
        free(new_domain);
        free(new_target);
        return NULL;
    }

    // Populate the valuation
    memcpy(v->rows->tuple, tuples, domain_size * size);
    for (uint32_t i = 0; i < size; ++i) {
        v->rows[i].value = values[i];
    }

    return v;
}

/*
This function creates a Valuation struct with automatically generated tuples and values.
Input: 
    uint16_t domain_size: the number of variables in the domain
    uint16_t domain[domain_size]: the variables in the domain
    uint16_t target_size: the number of variables in the target domain
    uint16_t target[target_size]: the variables in the target domain
    uint8_t states_per_var: the number of states per variable
Output:
    Valuation *v: the generated Valuation struct
*/
Valuation *auto_generate_valuation(
    uint16_t domain_size,
    uint16_t *domain,
    uint16_t target_size,
    uint16_t *target,
    uint8_t states_per_var)
{
    uint32_t size = 1;
    for (uint16_t i = 0; i < domain_size; ++i) {
        size *= states_per_var;
    }

    uint8_t tuples[size][domain_size];
    double values[size];
    uint32_t divisor;

    for (uint32_t i = 0; i < size; ++i) {
        divisor = 1;
        for (uint32_t j = 0; j < domain_size; ++j) {
            tuples[i][j] = (i / divisor) % states_per_var;
            divisor *= states_per_var;
        }
        values[i] = 1.0 / size;
    }

    Valuation *v = create_valuation(size, domain_size, domain, target_size, target, &tuples[0][0], values);
    if (!v) return NULL;

    return v;
}

int generate_pair(uint16_t domain_size, uint16_t overlap, uint8_t states_per_var, Valuation **v1, Valuation **v2) {
    uint16_t domain1[domain_size];
    uint16_t domain2[domain_size];
    uint16_t target1[1] = {domain_size - 1};
    uint16_t target2[1] = {domain_size - 1};

    for (uint16_t i = 0; i < domain_size; ++i) {
        domain1[i] = i;
    }
    for (uint16_t i = 0; i < domain_size; ++i) {
        domain2[i] = i + domain_size - overlap;
    }

    Valuation *new_v1 = auto_generate_valuation(domain_size, domain1, 1, target1, states_per_var);
    if (!new_v1) return -1;

    Valuation *new_v2 = auto_generate_valuation(domain_size, domain2, 1, target2, states_per_var);
    if (!new_v2) {
        free_valuation(new_v1);
        return -1;
    }

    *v1 = new_v1;
    *v2 = new_v2;

    return 0;
}

/*
This function frees the memory allocated for a Valuation struct
Input: 
    Valuation *v: the Valuation struct to free
Output: None
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

/*
This function computes the union of two domains, and will need to be freed by the caller.
Input: 
    const uint16_t *dom1: the first domain
    uint16_t n1: the size of the first domain
    const uint16_t *dom2: the second domain
    uint16_t n2: the size of the second domain
    uint16_t *out_size: pointer to store the size of the resulting domain
Output:
    uint16_t *domain: the resulting domain containing unique elements from either domain
*/
uint16_t *union_domain(
    const uint16_t *dom1, uint16_t n1,
    const uint16_t *dom2, uint16_t n2,
    uint16_t *out_size)
{
    uint16_t *domain = (uint16_t *) malloc((n1 + n2) * sizeof(uint16_t));
    if (!domain) return NULL;

    memcpy(domain, dom1, n1 * sizeof(uint16_t));
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
        if (!exists) domain[count++] = dom2[i];
    }

    *out_size = count;
    return domain;
}

/*
This function computes the intersection of two domains, and will need to be freed by the caller.
Input: 
    const uint16_t *dom1: the first domain
    uint16_t n1: the size of the first domain
    const uint16_t *dom2: the second domain
    uint16_t n2: the size of the second domain
    uint16_t *out_size: pointer to store the size of the resulting domain
Output:
    uint16_t *domain: the resulting domain containing unique elements from both domains
*/
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

/*
This function computes the common variables between two domains and returns their indices in two arrays.
Input: 
    const uint16_t *domain1: the first domain
    uint16_t size1: the size of the first domain
    const uint16_t *domain2: the second domain
    uint16_t size2: the size of the second domain
    uint32_t **v1_indices: pointer to store the indices of the first domain
    uint32_t **v2_indices: pointer to store the indices of the second domain
Output:
    uint32_t count: the number of common variables
*/
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

/*
This function checks if two tuples match at the provided indices.
Input: 
    const uint8_t *tuple1: the first tuple
    const uint8_t *tuple2: the second tuple
    const uint32_t *v1_indices: indices of the first tuple
    const uint32_t *v2_indices: indices of the second tuple
    uint32_t num_indices: number of indices to check
Output:
    int: 1 if tuples match, 0 otherwise
*/
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

/*
This function merges two tuples into a destination tuple, excluding the common variables from the second tuple.
Input: 
    uint8_t *dst: the destination tuple
    const uint8_t *tuple1: the first tuple
    uint32_t size1: size of the first tuple
    const uint8_t *tuple2: the second tuple
    uint32_t size2: size of the second tuple
    const uint32_t *v2_indices: indices of the second tuple
    uint32_t common_size: number of common variables
Output: None
*/
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