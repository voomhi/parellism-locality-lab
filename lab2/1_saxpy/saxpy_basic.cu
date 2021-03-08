#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from dev_offsetition of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
  // TODO: implement and use this interface if necessary
    return 0;
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

    // float *device_x;
    // float *device_y;
    // float *device_result;

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //
    
    int size = total_elems * sizeof(float);
    float *xarray_cuda,*yarray_cuda,*resarray_cuda,*alpha_cuda;
    cudaError_t test;
    // printf("malloc0");
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

    // test = cudaMalloc(&alpha, sizeof(float));
    // if(test != cudaSuccess)
    // 	exit(EXIT_FAILURE);

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();

    //
    // TODO: Compute number of thread blocks.
    //
    int blocks = (total_elems + threadsPerBlock-1)/threadsPerBlock;

    //
    // TODO: copy input arrays to the GPU using cudaMemcpy
    //

    // printf("copy0");
    double startGPUTime_H2D = CycleTimer::currentSeconds();

    test =cudaMemcpy(xarray_cuda,xarray,size, cudaMemcpyHostToDevice);
    if(test != cudaSuccess)
	exit(EXIT_FAILURE);
    // printf("copy1");

    test =cudaMemcpy(yarray_cuda,yarray,size, cudaMemcpyHostToDevice);
    if(test != cudaSuccess)
	exit(EXIT_FAILURE);
    
    cudaDeviceSynchronize();
    double endGPUTime_H2D = CycleTimer::currentSeconds();
    timeCopyH2DAvg += (endGPUTime_H2D-startGPUTime_H2D);

    // test =cudaMemcpy(alpha_cuda,alpha,sizeof(float), cudaMemcpyHostToDevice);
    // if(test != cudaSuccess)
    // 	exit(EXIT_FAILURE);



    //
    // TODO: insert time here to begin timing only the kernel
    //
    double startGPUTime = CycleTimer::currentSeconds();

    // run saxpy_kernel on the GPU
    // printf("kernal0");

    saxpy_kernel<<<blocks,threadsPerBlock>>>(total_elems,alpha,xarray_cuda,yarray_cuda,resarray_cuda);


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
    // printf("copy2");
    double startGPUTime_D2H = CycleTimer::currentSeconds();

    test =cudaMemcpy(resultarray,resarray_cuda,size, cudaMemcpyDeviceToHost);
        if(test != cudaSuccess)
    	exit(EXIT_FAILURE);
    cudaDeviceSynchronize();

    double endGPUTime_D2H = CycleTimer::currentSeconds();
    timeCopyD2HAvg += (endGPUTime_D2H-startGPUTime_D2H);
    
    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;
    timeKernelAvg += timeKernel;
    //
    // TODO free memory buffers on the GPU
    //
    cudaFree(xarray_cuda);
    cudaFree(yarray_cuda);
    cudaFree(resarray_cuda);

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
