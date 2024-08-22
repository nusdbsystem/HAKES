#ifndef HAKES_SEARCHWORKER_INDEX_BLAS_SGEMM_H
#define HAKES_SEARCHWORKER_INDEX_BLAS_SGEMM_H

#ifndef FINTEGER
#define FINTEGER long
#endif

// extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n,
           FINTEGER* k, const float* alpha, const float* a, FINTEGER* lda,
           const float* b, FINTEGER* ldb, float* beta, float* c, FINTEGER* ldc);
// }

#endif  // HAKES_SEARCHWORKER_INDEX_BLAS_SGEMM_H