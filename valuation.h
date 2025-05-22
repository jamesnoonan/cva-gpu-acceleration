#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef VALUATION_H
#define VALUATION_H


// A single valuation row
typedef struct {
    // The semiring value of the row. Assuming arithmetic/probability potentials
    double value;

    // tuple[i] stores the index of the state of variable[i] in this row
    uint8_t *tuple;

    // 1 if the row is incompatible, 0 otherwise
    uint8_t incompatible;
} ValuationRow;

// A single valuation in a conditional valuation algebra
typedef struct {
    // A pointer to the rows of the domain
    ValuationRow *rows;
    
    // The number of rows in the domain
    uint32_t size;

    // The variables in the domain, ordered in the same way as the tuples
    uint16_t *domain;

    // The variables in the target domain
    uint16_t *target;

    // The number of variables in each domain
    uint16_t domain_size;
    uint16_t target_size;
} Valuation;

// Print the valuation
void display_valuation(const Valuation *v);

// Allocate the memory for a valuation
Valuation *allocate_valuation(
    uint32_t size,
    uint16_t *domain,
    uint16_t domain_size,
    uint16_t *target,
    uint16_t target_size);

// Allocate the memory for a combined valuation
Valuation* alloc_combined_valuation(Valuation *v1, Valuation *v2);

Valuation *create_valuation(
    uint32_t size,
    uint16_t domain_size,
    uint16_t *domain,
    uint16_t target_size,
    uint16_t *target,
    uint8_t *tuples,
    double *values);

Valuation *auto_generate_valuation(
    uint16_t domain_size,
    uint16_t *domain,
    uint16_t target_size,
    uint16_t *target,
    uint8_t states_per_var);

int generate_pair(uint16_t domain_size, uint16_t overlap, uint8_t states_per_var, Valuation **v1, Valuation **v2);

// Free the memory of a valuation
void free_valuation(Valuation *v);


// Helpers
uint16_t *union_domain(
    const uint16_t *dom1, uint16_t n1,
    const uint16_t *dom2, uint16_t n2,
    uint16_t *out_size);

uint16_t *intersection_domain(
    const uint16_t *dom1, uint16_t n1,
    const uint16_t *dom2, uint16_t n2,
    uint16_t *out_size);

uint32_t get_common_vars(const uint16_t *domain1, uint16_t size1,
                         const uint16_t *domain2, uint16_t size2,
                         uint32_t **v1_indices, uint32_t **v2_indices);

int tuples_match(const uint8_t *tuple1, const uint8_t *tuple2,
                  const uint32_t *v1_indices, const uint32_t *v2_indices,
                  uint32_t num_indices);

void merge_tuples(
    uint8_t *dst,
    const uint8_t *tuple1, uint32_t size1,
    const uint8_t *tuple2, uint32_t size2,
    const uint32_t *v2_indices, uint32_t common_size);


#endif

#ifdef __cplusplus
}
#endif
