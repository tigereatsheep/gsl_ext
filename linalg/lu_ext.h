#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>

#define REAL double
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static int singular (const gsl_matrix * LU);

int gsl_linalg_rectLU_decomp (gsl_matrix * A);

int gsl_linalg_cpLU_decomp (gsl_matrix * A, gsl_permutation * rp,
                            gsl_permutation * cp, int *signum);

int gsl_linalg_cpLU_svx(const gsl_matrix * LU, const gsl_permutation * rp,
                        const gsl_permutation * cp, gsl_vector * x);

int gsl_linalg_cpLU_invert (const gsl_matrix * LU, const gsl_permutation * rp,
                            const gsl_permutation * cp, gsl_matrix * inverse);

int gsl_linalg_gaxpyLU_decomp (gsl_matrix * A, gsl_vector * v);

int gsl_linalg_rblockLU_decomp (gsl_matrix * A, size_t r);