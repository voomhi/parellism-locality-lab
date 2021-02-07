#include <stdlib.h>
#include <stdio.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 512 
#define N M 
#define P M 
#define Z 4

void matmul(float **A, float **B, float **C) {
  float sum;
  int   i;
  int   j;
  int   k;

  for (i=0; i<M; i++) {
    // for each row of C
    for (j=0; j<N; j++) {
      // for each column of C
      sum = 0.0f; // temporary value
      for (k=0; k<P; k++) {
        // dot product of row from A and column from B
        sum += A[i][k]*B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

// calculate C = AxB
void matmul_block(float **A, float **B, float **C) {
  float sum;
  int   i;
  int   j;
  int   k;
  int   f,g,h;
              

  for (f=0; f < M; f+=Z) {
    for (g=0; g < M; g+=Z) {
      for (h=0; h < M; h+=Z) {
        for (i=0; i<Z; i++) {
          for (j=0; j<Z; j++) {
            for (k=0; k<Z; k++) {
              C[i+f][j+g] += A[i+f][k+h]*B[k+h][j+g];
            }
          } 
        }
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
  float** C_blk;


  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  /* create_matrix(&C, M, N); */
  create_matrix(&C_blk, M, N);

  // assume some initialization of A and B
  // think of this as a library where A and B are
  // inputs in row-major format, and C is an output
  // in row-major.

  int i,j;
  /* for(i = 0; i < M; i++) { */
  /*   for(j = 0; j < M; j++) { */
  /*     A[i][j] = i+j; */
  /*     B[i][j] = i+j+2; */
  /*   } */
  /* } */

  /* matmul(A, B, C); */
  matmul_block(A, B, C_blk);

  /* for(i = 0; i < M; i++) { */
  /*   for(j = 0; j < M; j++) { */
  /*     if(C[i][j] != C_blk[i][j]) { */
  /*     	printf("%f %f\n", C[i][j], C_blk[i][j]); */
  /*     } */
  /*   } */
  /* } */

  return (0);
}
