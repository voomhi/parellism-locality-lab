#include <stdlib.h>
#include <stdio.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 32 
#define N M 
#define P M 
#define Z M/4 

// calculate C = AxB
void matmul(float **A, float **B, float **C) {
  float sum;
  int   i;
  int   j;
  int   k;
  int   h;

  for (h=0; h < M; h+=Z) {
    // for each block of C
    for (i=0; i<Z; i++) {
    // for each row of C
      for (j=0; j<Z; j++) {
      // for each column of C
      sum = 0.0f; // temporary value
        for (k=0; k<Z; k++) {
          // dot product of row from A and column from B
          sum += A[i][k]*B[k][j];
        }
        C[i][j] = sum;
      }
    }
  }
}

// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float*** A, int m, int n) {
  float **T = 0;
  int i;

  T = (float**)malloc( m*sizeof(float*));
  for ( i=0; i<m; i++ ) {
     T[i] = (float*)malloc(n*sizeof(float));
  }
  *A = T;
}

int main() {
  float** A;
  float** B;
  float** C;

  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  matmul(A, B, C);

  return (0);
}
