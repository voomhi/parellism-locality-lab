#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
  // TODO: implement and use this interface if necessary  
}

void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary  
}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary  
}

void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {

    const int threadsPerBlock = 256; // change this if necessary

    int size = total_elems * sizeof(float);
    float *xarray_cuda,*yarray_cuda,*resarray_cuda,*alpha_cuda;
    cudaError_t test;

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //

    test = cudaMalloc(&xarray_cuda, size);
    if(test != cudaSuccess)
	    exit(EXIT_FAILURE);
    // printf("malloc1");
    test = cudaMalloc(&yarray_cuda, size);
    if(test != cudaSuccess)
	    exit(EXIT_FAILURE);
    // printf("malloc2");

    test = cudaMalloc(&resarray_cuda, size);
    if(test != cudaSuccess)
	    exit(EXIT_FAILURE);

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();
    int blocks = (total_elems + threadsPerBlock-1)/threadsPerBlock;
    int partSize = size / partitions;

    for (int i=0; i<partitions; i++) {
  
        //
        // TODO: copy input arrays to the GPU using cudaMemcpy
        //

      test = cudaMemcpy(xarray_cuda, xarray + partSize * i, partSize, cudaMemcpyHostToDevice);
      if(test != cudaSuccess)
        exit(EXIT_FAILURE);

      test = cudaMemcpy(yarray_cuda, yarray + partSize * i , partSize, cudaMemcpyHostToDevice);
      if(test != cudaSuccess)
        exit(EXIT_FAILURE);


      // run saxpy_kernel on the GPU
      saxpy_kernel<<<blocks/partitions,threadsPerBlock>>>(total_elems/partitions,alpha,xarray_cuda + partSize * i,yarray_cuda + partSize * i,resarray_cuda + partSize * i);

      cudaError_t errCode = cudaPeekAtLastError();
      if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
      }

      //
      // TODO: copy result from GPU using cudaMemcpy
      //

      test = cudaMemcpy(resultarray,resarray_cuda,size, cudaMemcpyDeviceToHost);
      if(test != cudaSuccess)
    	  exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    //
    // TODO free memory buffers on the GPU
    //
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
