#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
// define the matrix dimensions A is MxP, B is PxN, and C is MxN
#define M 4
#define N 4
#define P 4
#define Z 16
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

void matmul_block(float **A, float **B, float **C) {
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


void recur_matmul(float **A, float **B, float **C, int b_size, int a1, int a2, int b1, int b2, int c1, int c2){
    float sum;
    if(b_size == B_SIZE_LIMIT){ // Base case
        for(int i = 0; i < b_size; i++){
            for(int j = 0; j < b_size; j++){
                sum = 0;
                for(int k = 0; k < b_size; k++){
                    sum += A[i + a1][k + a2]*B[k + b1][j + b2];
                }
                C[i + c1][j + c2] += sum;
            }
        }
    } else{
        int new_b_size = b_size >> 1;

	recur_matmul(A, B, C, new_b_size, a1, a2, b1, b2, c1, c2); // C1 = A1*B1
        recur_matmul(A, B, C, new_b_size, a1 + new_b_size, a2, b1, b2 + new_b_size, c1, c2); // + A2*B3
	
        recur_matmul(A, B, C, new_b_size, a1, a2, b1 + new_b_size, b2, c1 , c2+ new_b_size); // C2 = A1*B2 
        recur_matmul(A, B, C, new_b_size, a1 + new_b_size, a2, b1 + new_b_size, b2 + new_b_size, c1, c2 + new_b_size); // + A2*B4
	
        recur_matmul(A, B, C, new_b_size, a1, a2 + new_b_size, b1, b2, c1 + new_b_size, c2); // C3 = A3*B1
        recur_matmul(A, B, C, new_b_size, a1 + new_b_size, a2 + new_b_size, b1, b2 + new_b_size, c1 + new_b_size, c2); // + A4*B3
	
        recur_matmul(A, B, C, new_b_size, a1, a2 + new_b_size, b1 + new_b_size, b2, c1 + new_b_size, c2 + new_b_size); // C4 = A3*B2 
        recur_matmul(A, B, C, new_b_size, a1 + new_b_size, a2 + new_b_size, b1 + new_b_size, b2 + new_b_size, c1 + new_b_size, c2 + new_b_size); // + A4*B4
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
    float** C_block;

    create_matrix(&A, M, P);
    create_matrix(&B, P, N);
    create_matrix(&C, M, N);
    create_matrix(&C_block, M, N);

    // assume some initialization of A and B
    // think of this as a library where A and B are
    // inputs in row-major format, and C is an output
    // in row-major.

    for(int J = 0; J < M ; J++)	{
	for(int K = 0; K < P ; K++){
	    A[J][K] = J+K;
	}
    }
    for(int J = 0; J < P ; J++)	{
	for(int K = 0; K < N ; K++){
	    B[J][K] = J+K+3.1;
	}
    }
    for(int J = 0; J < M ; J++)	{
	for(int K = 0; K < N ; K++){
	    C[J][K] = 0;
	}
    }
    double A_D[2], B_D[2];
    __m512d A_vec = _mm512_load_pd(A_D);
    __m512d B_vec = _mm512_load_pd(B_D);
    A_vec = _mm512_add_pd(A_vec,B_vec);
    _mm512_store_pd(A_D,A_vec);

    matmul(A, B, C);
    /* matmul_block(A, B, C_block); */
    recur_matmul(A, B, C_block, M, 0, 0, 0, 0, 0, 0);
    for(int J = 0; J < M ; J++)	{
    	for(int K = 0; K < N ; K++){
	    printf("\n%f,%f\n",C[K][J],C_block[K][J]);
    	    if(C[J][K] != C_block[J][K]){
		printf("\n%i,%i\n",J,K);
		printf("\n%f,%f\n",C[J][K],C_block[J][K]);

    		return (1);
    	    }
    	}
    }
    return (0);
}
