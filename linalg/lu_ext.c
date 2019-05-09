#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_blas.h>
#include "lu_ext.h"


static int
singular (const gsl_matrix * LU)
{
  size_t i, n = LU->size1;

  for (i = 0; i < n; i++)
    {
      double u = gsl_matrix_get (LU, i, i);
      if (u == 0) return 1;
    }
 
 return 0;
}

int
gsl_linalg_rectLU_decomp (gsl_matrix * A)
{
  const size_t M = A->size1;
  const size_t N = A->size2;
  const size_t min = MIN(M, N);

  gsl_matrix_view mm = gsl_matrix_submatrix(A, 0, 0, min, min);

  if (singular(&mm.matrix))
    GSL_ERROR ("the submatrix of A is singular", GSL_EDOM);

  size_t i, j, k;
  REAL akk, aik;

  for (k=0; k<N; k++)
    {
      if (k < M)
          akk = gsl_matrix_get(A, k, k);
      for (i=k+1; i<M; i++)
        {
          if (k < M)
              gsl_matrix_set(A, i, k, gsl_matrix_get(A, i, k) / akk);

          if (k < N)
            {
              aik = gsl_matrix_get(A, i, k);
              for (j=k+1; j<N; j++)
                {
                  gsl_matrix_set(A, i, j,
                    gsl_matrix_get(A, i, j) - aik * gsl_matrix_get(A, k, j));
                }
            }
        }
    }
  return GSL_SUCCESS;
}

int
gsl_linalg_cpLU_decomp (gsl_matrix * A, gsl_permutation * rp, gsl_permutation * cp, int * signum)
{
  if (A->size1 != A->size2)
    GSL_ERROR ("complete pivoting LU decomposition requires square matrix", GSL_ENOTSQR);
  if (rp->size != A->size1)
    GSL_ERROR ("row permutation length must match matrix size", GSL_EBADLEN);
  if (cp->size != A->size2)
    GSL_ERROR ("column permutation length must match matrix size", GSL_EBADLEN);

  const size_t N = A->size1;
  size_t i, j, k, r_pivot, c_pivot;
  REAL aik, ajj, max;

  *signum = 1;
  gsl_permutation_init (rp);
  gsl_permutation_init (cp);

  for (j = 0; j < N - 1; j++)
    {
      /* Find maximum in the sub-matrix A(j:N, j:N) */
      max = fabs (gsl_matrix_get (A, j, j));
      r_pivot = j;
      c_pivot = j;

      for (i = j + 1; i < N; i++)
        {
          for (k = j + 1; k < N; k++)
            {
              aik = fabs (gsl_matrix_get (A, i, k));
              if (aik > max)
                {
                  max = aik;
                  r_pivot = i;
                  c_pivot = k;
                }
            }
        }

      if (r_pivot != j)
        {
          gsl_matrix_swap_rows (A, j, r_pivot);
          gsl_permutation_swap (rp, j, r_pivot);
          *signum = -(*signum);
        }

      if (c_pivot != j)
        {
          gsl_matrix_swap_columns (A, j, c_pivot);
          gsl_permutation_swap (cp, j, c_pivot);
          *signum = -(*signum);
        }

      ajj = gsl_matrix_get (A, j, j);

      if (ajj != 0.0)
        {
          for (i = j + 1; i < N; i++)
            {
              REAL aij = gsl_matrix_get (A, i, j) / ajj;
              gsl_matrix_set (A, i, j, aij);

              for (k = j + 1; k < N; k++)
                {
                  REAL aik = gsl_matrix_get (A, i, k);
                  REAL ajk = gsl_matrix_get (A, j, k);
                  gsl_matrix_set (A, i, k, aik - aij * ajk);
                }
            }
        }
    }
  return GSL_SUCCESS;
}

int
gsl_linalg_cpLU_svx(const gsl_matrix * LU, const gsl_permutation * rp,
                    const gsl_permutation * cp, gsl_vector * x)
{
  gsl_permute_vector (rp, x);

  gsl_blas_dtrsv (CblasLower, CblasNoTrans, CblasUnit, LU, x);

  gsl_blas_dtrsv (CblasUpper, CblasNoTrans, CblasNonUnit, LU, x);

  gsl_permute_vector (cp, x);

  return GSL_SUCCESS;
}

int
gsl_linalg_cpLU_invert (const gsl_matrix * LU, const gsl_permutation * rp,
                        const gsl_permutation * cp, gsl_matrix * inverse)
{
  size_t i, n = LU->size1;

  int status = GSL_SUCCESS;

  if (singular (LU)) 
    {
      GSL_ERROR ("matrix is singular", GSL_EDOM);
    }

  gsl_matrix_set_identity (inverse);

  for (i = 0; i < n; i++)
    {
      gsl_vector_view c = gsl_matrix_column (inverse, i);
      int status_i = gsl_linalg_cpLU_svx (LU, rp, cp, &(c.vector));

      if (status_i)
        status = status_i;
    }

  return status;
}

int
gsl_linalg_gaxpyLU_decomp (gsl_matrix * A, gsl_vector * v)
{
  double r;
  const size_t n = A->size1;
  size_t i, j, k;

  for (i=0; i<n; i++)
    {
      if (i == 0) gsl_matrix_get_col (v, A, 0);

      else {
        for (j=1; j<i; j++)
          {
            r = 0.0;
            for (k=0; k<j; k++)
                r += gsl_matrix_get (A, j, k) * gsl_matrix_get (A, k, i);
            gsl_matrix_set (A, j, i, gsl_matrix_get (A, j, i) - r);
          }

        for (j=i; j<n; j++)
          {
            r = 0.0;
            for (k=0; k<i; k++)
                r += gsl_matrix_get (A, j, k) * gsl_matrix_get (A, k, i);
            gsl_vector_set (v, j, gsl_matrix_get (A, j, i) - r);
          }
      }

      gsl_matrix_set (A, i, i, gsl_vector_get(v, i));

      for (j=i+1; j<n; j++)
          gsl_matrix_set (A, j, i, gsl_vector_get (v, j) / gsl_vector_get (v, i));
    }

    return GSL_SUCCESS;
}

int
gsl_linalg_rblockLU_decomp (gsl_matrix * A, size_t r)
/* recursive block LU decomposition */
{
  const size_t N = A->size1;
  if (N <= r)
    {
      gsl_linalg_rectLU_decomp(A);
      return 0;
    }
  else
    {
      /*  solve A(:, 1:r) = (L11, L12)^T * U11 */
      gsl_matrix_view LU1 = gsl_matrix_submatrix(A, 0, 0, N, r);
      gsl_linalg_rectLU_decomp(&LU1.matrix);

      /* solve L11 U12 = A(1:r, r+1:N) for U12 */
      gsl_matrix_view L11 = gsl_matrix_submatrix(A, 0, 0, r, r);
      gsl_matrix_view U12 = gsl_matrix_submatrix(A, 0, r, r, N - r);
      gsl_blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                     1.0, &L11.matrix, &U12.matrix);
      
      /* prepare for next recursive  A^~ = A(r+1:n, r+1:n) - L21 * U12 */
      gsl_matrix_view A_hat = gsl_matrix_submatrix(A, r, r, N - r, N - r);
      gsl_matrix_view L21 = gsl_matrix_submatrix(A, r, 0, N - r, r);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, &L21.matrix, &U12.matrix, 1.0, &A_hat.matrix);
      return gsl_linalg_rblockLU_decomp(&A_hat.matrix, r);
    }
  return GSL_SUCCESS;
}

