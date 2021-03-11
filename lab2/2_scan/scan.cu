#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <math.h>

#include "CycleTimer.h"


extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
    return (working_set_size+threadsPerBlock-1)/threadsPerBlock;
}
__global__ void
exclusive_scan_kernel_inital( int elem_count, int * device_array, int * device_result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x ;
    if(index < elem_count)
    {
	device_result[index] = (index != 0) ? device_array[index-1] : 0;
    }
}
__global__ void
exclusive_scan_kernel(int cumulation_idx, int elem_count, int * device_result,int* device_temp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x ;  
    // int index = blockIdx.x * blockDim.x + threadIdx.x+ cumulation_idx;  

    if(index < elem_count)
    {
	cumulation_idx = index - cumulation_idx;
	if(cumulation_idx >= 0)
	    device_temp[index] = device_result[index] + device_result[cumulation_idx];
	else
	{
	    device_temp[index] = device_result[index];
	}
	// __syncthreads();
    }

}
__global__ void
exclusive_scan_kernel_copy(int elem_count, int * device_result,int* device_temp,int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x+offset;  
    if(index < elem_count)
    {
	device_result[index] = device_temp[index];
    }

}

__global__ void
exclusive_scan_kernel_work_eff_upper(int elem_count, int * device_result,int current_power,int offset)
{
    int index = (threadIdx.x<<(current_power+1))+offset + ((blockIdx.x * blockDim.x)<<(current_power+1)) ;  
    if(index < elem_count)
    {
	//I is less than log2(numThreads) if numThreads = 256 then I < 
	for(int I=0 ; I < 8;I++)
	{
	    int tempCheck= (1<<I) -1;
	    // int temp = device_result[index];
	    if(((threadIdx.x & tempCheck) == tempCheck))
		// temp = 	device_result[index] + device_result[index -(1<<(I+current_power))];
		device_result[index] += device_result[index -(1<<(I+current_power))];
	     __syncthreads();
	    // device_result[index] = temp;
	}
    }

}
__global__ void
exclusive_scan_kernel_work_eff_bott(long elem_count, int * device_result,int current_power,int offset)
{
    long index = (threadIdx.x<<(current_power+1))+offset + ((blockIdx.x * blockDim.x)<<(current_power+1)) ;  
    if(index < elem_count)
    {
	//I is less than log2(numThreads) if numThreads = 256 then I < 
	// for(int I=0 ; I < 8;I++)
	// {
	    // int tempCheck= (1<<I) -1;
	    // if(((threadIdx.x & tempCheck) == tempCheck))
	device_result[index] += device_result[index - (1<<(current_power))];
	// }
    }

}

void exclusive_scan(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
    * You are passed the locations of the input and output in device memory,
    * but this is host code -- you will need to declare one or more CUDA 
    * kernels (with the __global__ decorator) in order to actually run code
    * in parallel on the GPU.
    * Note you are given the real length of the array, but may assume that
    * both the input and the output arrays are sized to accommodate the next
    * power of 2 larger than the input.
    */

    //length = nextPow2(length); //is this necessary?

    int threadsPerBlock = 256;
    unsigned int numBlocks = (length+threadsPerBlock-1)/threadsPerBlock;// = getBlocks(length, threadsPerBlock);
    // cudaError_t errCode;
	// errCode = cudaPeekAtLastError();
        // if (errCode != cudaSuccess) {
        //     printf( "WARNING START: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        // }

    if(length >= 0)
    {
	unsigned int MSB =  31 - __builtin_clz((unsigned int)length);
	// printf("0: Length %d MSB %d FFS %d \n",length,MSB,__builtin_ffs((unsigned int)length) );
	MSB = (MSB == (__builtin_ffs((unsigned int)length)-1))? MSB : MSB+1;
	// printf("Length %d MSB %d \n",length,MSB);

	// int cumulation_idx;
	exclusive_scan_kernel_inital<<<numBlocks,threadsPerBlock>>>(length,device_start,device_result);	
	// errCode = cudaPeekAtLastError();
        // if (errCode != cudaSuccess) {
        //     printf( "WARNING PRE: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        // }

	for(int I = 0; I < MSB; I+=8)
	{
	    int threads  = (length) >>(I+1);
	    numBlocks = (threads+threadsPerBlock-1)/threadsPerBlock;
	    if(numBlocks > 0)
		exclusive_scan_kernel_work_eff_upper<<<numBlocks,threadsPerBlock>>>(
		    length, device_result,
		    I,(1<<(I+1))-1);
	//     printf("LoopA %d : %d,%d \n",I,numBlocks,threads);
	//     	    cudaDeviceSynchronize();
        // errCode = cudaPeekAtLastError();
        // if (errCode != cudaSuccess) {
        //     printf( "WARNING A %d: A CUDA error occured: code=%d, %s\n",I, errCode, cudaGetErrorString(errCode));
        // }

	}
        // errCode = cudaPeekAtLastError();
        // if (errCode != cudaSuccess) {
        //     printf( "WARNING A: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        // }
	//     cudaDeviceSynchronize();

	for(int I = MSB-2,J = 1; I >=0; I--,J++)
	{
	    int threads = (1<<J) -1;
	    if(threads > length/2)
		threads = length/2;
	    numBlocks = (threads+threadsPerBlock-1)/threadsPerBlock;
	     exclusive_scan_kernel_work_eff_bott<<<numBlocks,threadsPerBlock>>>(
		length, device_result,
		I,(3<<I)-1);
	    // printf("LoopB %d : %d,%d \n",I,numBlocks,threads);
	    // cudaDeviceSynchronize();

	     // printf("Offset %d \n",(3<<I)-1);
	    // errCode = cudaGetLastError();
	    // if (errCode != cudaSuccess) {
	    // 	printf( "WARNING B %d: A CUDA error occured: code=%d, %s\n",J, errCode, cudaGetErrorString(errCode));
	    // }
	    //     cudaDeviceSynchronize();


	}
	// errCode = cudaPeekAtLastError();
        // if (errCode != cudaSuccess) {
        //     printf( "WARNING B: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        // }

    }

   
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    // cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaError_t errCode;
    errCode =  cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
        if (errCode != cudaSuccess) {
            fprintf(stderr, "MALLOC WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        }

    // cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    errCode =  cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
        if (errCode != cudaSuccess) {
            fprintf(stderr, "MALLOC WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        }

    // cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               // cudaMemcpyHostToDevice);
    errCode = cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);
        if (errCode != cudaSuccess) {
            fprintf(stderr, "MEM_COPY WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
        }

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);

    // for(int I = 0; I < (end-inarray); I++)
    // {
    // 	printf("%d ",resultarray[I]);
    // }
    // printf("\n ");

    cudaFree(device_result);
    cudaFree(device_input);
    
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void
indicate_repeat( int *device_input, int length, int *device_output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length-1)
    {
	device_output[index] = (device_input[index] == device_input[index+1]);
    } else if(index == length-1)
    {
	device_output[index] = 0;
    }

}
__global__ void
store_repeat( int *device_input, int length, int *device_output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < length)
    {
	if(device_input[index] != device_input[index+1])
	    device_output[device_input[index]]=  index;
    }
}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */
    int* device_num_repeats;
    int* device_num_repeats_prefix_sum;
    int rounded_length = nextPow2(length);
    int threadsPerBlock = 64;
    unsigned int numBlocks = getBlocks(length, threadsPerBlock);
    cudaError_t test;
    test = cudaMalloc((void **)&device_num_repeats_prefix_sum, sizeof(int) * rounded_length);
    if(test != cudaSuccess)
        exit(EXIT_FAILURE);
    test = cudaMalloc((void **)&device_num_repeats, sizeof(int) * rounded_length);
    if(test != cudaSuccess)
        exit(EXIT_FAILURE);

    indicate_repeat<<<numBlocks,threadsPerBlock>>>( device_input, length, device_num_repeats);

    exclusive_scan( device_num_repeats, length, device_num_repeats_prefix_sum);
    
    //The last integer would indicate the number of repeats in the pattern
    //Uses PrefixSum

    store_repeat<<<numBlocks,threadsPerBlock>>>(device_num_repeats_prefix_sum,length,device_output);

    int numRepeats;
    cudaMemcpy(&numRepeats, &(device_num_repeats_prefix_sum[length-1]), sizeof(int),
               cudaMemcpyDeviceToHost);

    // printf("Number of Repeats %d \n",numRepeats);

    
    cudaFree(device_num_repeats_prefix_sum);
    cudaFree(device_num_repeats);


    return numRepeats;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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
