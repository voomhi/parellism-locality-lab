#include <stdlib.h>

// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 512
#define N 512
#define P 512
#define B_SIZE_LIMIT 2

// calculate C = AxB
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

void recur_matmul(float **A, float **B, float **C, int b_size, int x, int y){
    if(b_size == B_SIZE_LIMIT){ // Base case
        for(int i = 0; i < b_size; i++){
            for(int j = 0; j < b_size; j++){
                sum = 0.0f;
                for(int k = 0; k < b_size; k++){
                    sum += A[i + x][k + y]*B[k + x][j + y];
                }
                C[i + x][j + y] = sum;
            }
        }
    } else{
        int new_b_size = b_size / 2;
        recur_matmul(A, B, C, new_b_size, x, y); // Top-left block
        recur_matmul(A, B, C, new_b_size, x + new_b_size, y); // Top-right block
        recur_matmul(A, B, C, new_b_size, x, y + new_b_size); // Bottom-left block
        recur_matmul(A, B, C, new_b_size, x + new_b_size, y + new_b_size); // Bottom-right block
    }
}

// function to allocate a matrix on the heap
// creates an mXn matrix and returns the pointer.
//
// the matrices are in row-major order.
void create_matrix(float*** A, int m, int n) {
  float **T = 0;
  int i;

  T = (float**)malloc(m*sizeof(float*));
  for ( i=0; i<m; i++ ) {
     T[i] = (float*)malloc(n*sizeof(float));
  }
  *A = T;
}

int main() {
  float** A;
  float** B;
  float** C;
  float** D;

  create_matrix(&A, M, P);
  create_matrix(&B, P, N);
  create_matrix(&C, M, N);

  for(int i = 0; i < M; i++){
    for(int j = 0; j < M; j++){
      A[i][j] = i + j;
      B[i][j] = i * j;
    }
  }  

  matmul(A, B, C);
  recur_matmul(A, B, C, M, 0, 0);

  for(int i = 0; i < M; i++){
    for(int j = 0; j < M; j++){
      if(C[i][j] != D[i][j]){
        return 1; // Failure
      }
    }
  }

  return (0);
}