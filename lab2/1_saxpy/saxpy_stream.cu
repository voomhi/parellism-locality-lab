#include <stdio.h>
#include <assert.h>
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
    return (working_set_size+threadsPerBlock-1)/threadsPerBlock;
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

//total_elems must be divisible by partitions
    const int threadsPerBlock = 512; // change this if necessary

    // float *device_x;
    // float *device_y;
    // float *device_result;
    cudaStream_t stream[32];
    
    assert(partitions <= 32);

//
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //
    int size = total_elems * sizeof(float);
    float *xarray_cuda,*yarray_cuda,*resarray_cuda;
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

    
    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();
    int elemInPartition = total_elems/partitions;
    // fprintf(stderr,"ElemInPartition %d\n",elemInPartition);

    for (int i=0; i<partitions; i++)
    {
	
        //
        // TODO: copy input arrays to the GPU using cudaMemcpy
        //
	int offset  = i*elemInPartition;

	// fprintf(stderr,"Partition %d Offset %x \n",i,offset);
	// fprintf(stderr,"ADDR %x NEW %x \n",xarray_cuda,xarray_cuda+offset);

	cudaStreamCreate(&stream[i]);
	cudaError_t errCode = cudaPeekAtLastError();
        if (errCode != cudaSuccess) {
            fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
	    exit(1);
        }
	cudaMemcpyAsync(&(xarray_cuda[offset]),&(xarray[offset]),elemInPartition*sizeof(float)
			, cudaMemcpyHostToDevice,stream[i]);
	errCode = cudaPeekAtLastError();

        if (errCode != cudaSuccess) {
            fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
	    exit(1);
        }
	cudaMemcpyAsync(&(yarray_cuda[offset]),&(yarray[offset]),elemInPartition*sizeof(float)
			, cudaMemcpyHostToDevice,stream[i]);
	errCode = cudaPeekAtLastError();
        if (errCode != cudaSuccess) {
            fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
	    exit(1);		    
        }

        //
        // TODO: insert time here to begin timing only the kernel
        //
    
        // compute number of blocks and threads per block

        // run saxpy_kernel on the GPU
    
        //
        // TODO: insert timer here to time only the kernel.  Since the
        // kernel will run asynchronously with the calling CPU thread, you
        // need to call cudaDeviceSynchronize() before your timer to
        // ensure the kernel running on the GPU has completed.  (Otherwise
        // you will incorrectly observe that almost no time elapses!)
        //
	int numblocks=getBlocks(elemInPartition,threadsPerBlock);
        saxpy_kernel<<<numblocks,threadsPerBlock,0,stream[i]>>>(total_elems,alpha,&(xarray_cuda[offset])
								,&(yarray_cuda[offset]),&(resarray_cuda[offset])
								);

	errCode = cudaPeekAtLastError();
        if (errCode != cudaSuccess) {
            fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        }
    
        //
        // TODO: copy result from GPU using cudaMemcpy
        //
	// cudaDeviceSynchronize();

	errCode = cudaPeekAtLastError();
        if (errCode != cudaSuccess) {
            fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        }


    }

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    cudaMemcpy(resultarray,resarray_cuda,size
		    ,cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;
    for (int i = 0; i < partitions; ++i)
	cudaStreamDestroy(stream[i]);
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
