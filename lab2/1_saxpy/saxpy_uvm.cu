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

// static inline
// int getBlocks(long working_set_size, int threadsPerBlock) {
//   // TODO: implement and use this interface if necessary
//     return 0;
// }

void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary
    cudaMallocManaged(xarray, sizeof(float)*size);
    cudaMallocManaged(yarray, sizeof(float)*size);
    cudaMallocManaged(resultarray, sizeof(float)*size);

}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary  
    cudaFree(xarray);
    cudaFree(yarray);
    cudaFree(resultarray);

}

void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {

    const int threadsPerBlock = 32; // change this if necessary

    // float *device_x;
    // float *device_y;
    // float *device_result;

    //
    // TODO: do we need to allocate device memory buffers on the GPU here?
    // Yes can't do it in main due to not being able to include the files

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

  
    //
    // TODO: do we need copy here?
    // No
     
    //
    // TODO: insert time here to begin timing only the kernel
    //
    
    // compute number of blocks and threads per block
    double startGPUTime = CycleTimer::currentSeconds();

    // run saxpy_kernel on the GPU
    // printf("kernal0");
    int blocks = (total_elems + threadsPerBlock-1)/threadsPerBlock;
    saxpy_kernel<<<blocks,threadsPerBlock>>>(total_elems*sizeof(float),alpha,xarray,yarray,resultarray);


    //
    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaDeviceSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    cudaDeviceSynchronize();
    double endGPUTime = CycleTimer::currentSeconds();
    double timeKernel = endGPUTime - startGPUTime;
    
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    
    //
    // TODO: copy result from GPU using cudaMemcpy
    //

    // What would be copy time when we use UVM?
    //Kernal runTime
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;
    timeKernelAvg += timeKernel;
    //
    // TODO free device memory if you allocate some device memory earlier in this function.
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
