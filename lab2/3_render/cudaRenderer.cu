#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
// shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr, float rad, float maxDist) {
shadePixel_alt(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr, float rad,float3 inputrgb) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    // float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // This conditional is in the inner loop, but it evaluates the
    // same direction for all threads so it's cost is not so
    // bad. Attempting to hoist this conditional is not a required
    // student optimization in Assignment 2
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
	// float oneMinusAlpha = 1.f - alpha;

	// float4 existingColor = *imagePtr;
	// float4 newColor;
	// newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
	// newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
	// newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
	// newColor.w = alpha + existingColor.w;
	// *imagePtr=newColor;


    } else {
        // simple: each circle has an assigned color
        // int index3 =  (circleIndex<<1) + (circleIndex);
        // rgb = *(float3*)&(cuConstRendererParams.color[index3]);
	rgb = inputrgb;
        alpha = .5f;


	// float4 existingColor = *imagePtr;
	// float4 newColor;
	// newColor.x = alpha * rgb.x + alpha * existingColor.x;
	// newColor.y = alpha * rgb.y + alpha * existingColor.y;
	// newColor.z = alpha * rgb.z + alpha * existingColor.z;
	// newColor.w = alpha + existingColor.w;
	// *imagePtr=newColor;

    }

    float oneMinusAlpha = 1.f - alpha;

    // // BEGIN SHOULD-BE-ATOMIC REGION
    // // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    *imagePtr=newColor;


    
    // // global memory write
    // *imagePtr=newColor;
    // // *imagePtr = existingColor;

    // END SHOULD-BE-ATOMIC REGION
}
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr, float rad) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    // float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // This conditional is in the inner loop, but it evaluates the
    // same direction for all threads so it's cost is not so
    // bad. Attempting to hoist this conditional is not a required
    // student optimization in Assignment 2
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
	// float oneMinusAlpha = 1.f - alpha;

	// float4 existingColor = *imagePtr;
	// float4 newColor;
	// newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
	// newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
	// newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
	// newColor.w = alpha + existingColor.w;
	// *imagePtr=newColor;


    } else {
        // simple: each circle has an assigned color
        int index3 =  (circleIndex<<1) + (circleIndex);
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
	// rgb = inputrgb;
        alpha = .5f;


	// float4 existingColor = *imagePtr;
	// float4 newColor;
	// newColor.x = alpha * rgb.x + alpha * existingColor.x;
	// newColor.y = alpha * rgb.y + alpha * existingColor.y;
	// newColor.z = alpha * rgb.z + alpha * existingColor.z;
	// newColor.w = alpha + existingColor.w;
	// *imagePtr=newColor;

    }

    float oneMinusAlpha = 1.f - alpha;

    // // BEGIN SHOULD-BE-ATOMIC REGION
    // // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    *imagePtr=newColor;


    
    // // global memory write
    // *imagePtr=newColor;
    // // *imagePtr = existingColor;

    // END SHOULD-BE-ATOMIC REGION
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}
__device__ __forceinline__ int
circleInBoxConservative(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{
    // expand box by circle radius.  Test if circle center is in the
    // expanded box.

    if ( circleX >= (boxL - circleRadius) &&
         circleX <= (boxR + circleRadius) &&
         circleY >= (boxB - circleRadius) &&
         circleY <= (boxT + circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}
__device__ __forceinline__ int
circleInBox(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // clamp circle center to box (finds the closest point on the box)
    // float closestX = (circleX > boxL) ? ((circleX < boxR) ? circleX : boxR) : boxL;
    // float closestY = (circleY > boxB) ? ((circleY < boxT) ? circleY : boxT) : boxB;
    float closestX = fmaxf(boxL,fminf(circleX,boxR));
    float closestY = fmaxf(boxB,fminf(circleY,boxT));

    
    // is circle radius less than the distance to the closest point on
    // the box?
    float distX = closestX - circleX;
    float distY = closestY - circleY;

    if ( ((distX*distX) + (distY*distY)) <= (circleRadius*circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}
#define blocksize 2

__global__ void blockRender()   
{
    int imageHeight = cuConstRendererParams.imageHeight;
    int imageWidth = cuConstRendererParams.imageWidth;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    // int index = blockIdx.x * blockDim.x + threadIdx.x;

    //Calculate the box to shade
    int pixelY=blocksize*blockIdx.x;//8x16
    int pixelX=blocksize*threadIdx.x;
    //calculate the array to look at for texture elemination
    float boxL,boxR,boxT,boxB;
    boxL=invWidth *static_cast<float>(pixelX);
    boxR=invWidth *static_cast<float>(pixelX+blocksize);
    boxB=invHeight *static_cast<float>(pixelY);
    boxT=invHeight *static_cast<float>(pixelY+blocksize);
    const int numCirlesToRender = cuConstRendererParams.numCircles;
    __shared__ float3 sharedp[1024*2];
    __shared__ float sharedrad[1024*2];

    // for(int I = 0; I < 1000 && I < cuConstRendererParams.numCircles; I++)
    // for(int I = 0; I < cuConstRendererParams.numCircles; I++)
    // {
    // 	float3 p = *(float3*)(&cuConstRendererParams.position[3*I]);
    // 	float  rad = cuConstRendererParams.radius[I];

    // 	bool cont = circleInBoxConservative(
    // 	    p.x,p.y,rad,
    // 	    boxL, boxR, boxT, boxB);
    // 	if(cont)
    // 	{
    // 	    float maxDist = rad * rad;
    // 	    for(int K = pixelY; K < pixelY+2;K++)
    // 	    {
    // 	    	float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (K * imageWidth + pixelX)]);
    // 	    	for(int J = pixelX; J < pixelX+2;J++)
    // 	    	{
    // 	    	    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(J) + 0.5f),
    // 	    						 invHeight * (static_cast<float>(K) + 0.5f));
    // 	    	    shadePixel(I, pixelCenterNorm, p, imgPtr,rad,maxDist);
    // 	    	    imgPtr++;
    // 	    	}
    // 	    }      	
    // 	}
    // }
    
    for(int J = 0; J < numCirlesToRender; J += 1024*2)
    {
	// int offset= J;
	if(threadIdx.x+J<numCirlesToRender)
	    for(int I = threadIdx.x; I < 1024*2 && I+J < numCirlesToRender; I+= blockDim.x)
	    {
		int indexofcircle = I;
		sharedp[I] = *(float3*)(&cuConstRendererParams.position[3*indexofcircle]);
		sharedrad[I] =  cuConstRendererParams.radius[indexofcircle];
	    }    
	__syncthreads();
	for(int I = 0; I+J < numCirlesToRender && I < 1024*2; I++)
	{
	    float3 p = sharedp[I];
	    float  rad = sharedrad[I];
	    bool cont = circleInBoxConservative(p.x,p.y,rad,
						boxL, boxR, boxT, boxB);
	    if(cont) 
	    {
	    	// float maxDist = rad * rad;
	    	for(int K = pixelY; K < pixelY+blocksize;K++)
	    	{
	    	    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (K * imageWidth + pixelX)]);
	    	    for(int J = pixelX; J < pixelX+blocksize;J++)
	    	    {
	    		float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(J) + 0.5f),
	    						     invHeight * (static_cast<float>(K) + 0.5f));
	    		// shadePixel(I, pixelCenterNorm, p, imgPtr,rad,maxDist);
	    		shadePixel(I, pixelCenterNorm, p, imgPtr,rad);

	    		imgPtr++;
	    	    }
	    	}      	

	    }
	}
    }    

    
}
#undef blocksize
#define blocksize 2

// __global__ void blockRender_alt(int* checkblock,int* checkblock_size,short numboxes,int boxsize)   
// {
//     const int imageHeight = cuConstRendererParams.imageHeight;
//     const int imageWidth = cuConstRendererParams.imageWidth;
//     const float invWidth = 1.f / imageWidth;
//     const float invHeight = 1.f / imageHeight;
//     // int index = blockIdx.x * blockDim.x + threadIdx.x;
//     //calculate the array to look at for texture elemination
//     // for(int I = 0; I < 1000 && I < cuConstRendererParams.numCircles; I++)
//     const int numBlocksPerMega = boxsize*boxsize/blocksize/blocksize/blockDim.x;
//     const int megablock= blockIdx.x/numBlocksPerMega;
//     const int blockInMega= blockIdx.x%numBlocksPerMega;
//     const int rowInThread= (threadIdx.x+blockInMega* blockDim.x ) %( boxsize / blocksize)  ;
//     const int colInThread= (threadIdx.x+blockInMega* blockDim.x) / (boxsize/blocksize);
//     const int megaRow= megablock%(imageHeight/boxsize);
//     const int megaCol= megablock/(imageHeight/boxsize);
//     const int numCirlesToRender= checkblock_size[megablock];
//     //Calculate the box to shade
//     const int pixelX=megaRow*boxsize+blocksize*rowInThread;
//     const int pixelY=megaCol*boxsize+blocksize*colInThread;//8x16
        
//     __shared__ float3 sharedp[1024*2];
//     __shared__ float sharedrad[1024*2];
//     __shared__ int sharedidx[1024*2];
//     const float boxL=invWidth *static_cast<float>(pixelX);
//     const float boxR=invWidth *static_cast<float>(pixelX+blocksize);
//     const float boxB=invHeight *static_cast<float>(pixelY);
//     const float boxT=invHeight *static_cast<float>(pixelY+blocksize);


// //Can improve with shared memory for p and rad
//     for(int J = 0; J < numCirlesToRender; J += 1024*2)
//     {
// 	// int offset= J;
// 	if(threadIdx.x+J<numCirlesToRender)
// 	    for(int I = threadIdx.x; I < 1024*2 && I+J < numCirlesToRender; I+= blockDim.x)
// 	    {
// 		int indexofcircle = checkblock[I+J+megablock*cuConstRendererParams.numCircles];
// 		sharedp[I] = *(float3*)(&cuConstRendererParams.position[3*indexofcircle]);
// 		sharedrad[I] =  cuConstRendererParams.radius[indexofcircle];
// 		sharedidx[I] = indexofcircle;
// 	    }    
// 	__syncthreads();
// 	for(int I = 0; I+J < numCirlesToRender && I < 1024*2; I++)
// 	{
// 	    int indexofcircle = sharedidx[I];
// 	    float3 p = sharedp[I];
// 	    float  rad = sharedrad[I];
// 	    bool cont = circleInBox(p.x,p.y,rad,
// 						boxL, boxR, boxT, boxB);
// 	    if(cont) 
// 	    {
// 	    	// float maxDist = rad * rad;
// 	    	for(int K = pixelY; K < pixelY+blocksize;K++)
// 	    	{
// 	    	    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (K * imageWidth + pixelX)]);
// 	    	    for(int J = pixelX; J < pixelX+blocksize;J++)
// 	    	    {
// 	    		float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(J) + 0.5f),
// 	    						     invHeight * (static_cast<float>(K) + 0.5f));
// 	    		// shadePixel(indexofcircle, pixelCenterNorm, p, imgPtr,rad,maxDist);
// 	    		shadePixel(indexofcircle, pixelCenterNorm, p, imgPtr,rad);

// 	    		imgPtr++;
// 	    	    }
// 	    	}      	

// 	    }
// 	}
//     }    

// }

#define blog2(a) (31-__clz(a))
#define bmod(a,b) (a&(b-1))
__global__ void blockRender_alt_limit(int* checkblock,int* checkblock_size,short numboxes,int boxsize				      )   
{
    // #define blocksize 2
    const int imageHeight = cuConstRendererParams.imageHeight;
    const int imageWidth = cuConstRendererParams.imageWidth;
    const float invWidth = 1.f / imageWidth;
    const float invHeight = 1.f / imageHeight;
    
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    //calculate the array to look at for texture elemination
    // for(int I = 0; I < 1000 && I < cuConstRendererParams.numCircles; I++)
    const int numBlocksPerMega = boxsize*boxsize/blocksize/blocksize/blockDim.x;
    const int blockInMega= blockIdx.x&(numBlocksPerMega-1);
    const int megablock= blockIdx.x>>(blog2(numBlocksPerMega));

    // const int blockrowsize = boxsize / blocksize;
    const int dimSqrt = 1<<(blog2(blockDim.x)/2);
    const int dimBlockSqrt= 1<<(blog2(numBlocksPerMega)/2);
    const int rowInThread=  bmod(threadIdx.x,dimSqrt) + bmod(blockInMega,dimBlockSqrt)*dimSqrt;
    const int colInThread= (threadIdx.x>>blog2(dimSqrt))+(blockInMega>>blog2(dimBlockSqrt))*dimSqrt;
    
    const int MegaBoxDim=imageHeight/boxsize;
    const int megaRow= megablock&(MegaBoxDim-1);
    const int megaCol= megablock>>(31 - __clz(MegaBoxDim));
    
    const int numCirlesToRender= checkblock_size[megablock];

//Calculate the box to shade
    const int pixelX=megaRow*boxsize+blocksize*rowInThread;
    const int pixelY=megaCol*boxsize+blocksize*colInThread;//8x16
// #define sharedmem (768)        
    // __shared__ float3 sharedp[sharedmem];
    // __shared__ float sharedrad[sharedmem];
    // __shared__ int sharedidx[sharedmem];
    // __shared__ bool sharedBlock[sharedmem];

    // extern __shared__ char sharedmemearr[];
    // int *sharedidx = (int*)sharedmemearr; 
    // float3* sharedp = (float3*) &sharedmemearr[1024];
    // float *sharedrad = (float*)(&sharedp[1024]);
    // bool *sharedBlock = (bool*)(&sharedrad[1024]);
    // // assert((float*)(sharedp + sizeof(float3)*sharedmem) == sharedrad);
    // // assert((bool*)(sharedrad + sizeof(float)*sharedmem) == sharedBlock);
    const int sharedmem=256;
    __shared__ float3 sharedp[sharedmem];
    __shared__ float sharedrad[sharedmem];
    __shared__ int sharedidx[sharedmem];
    __shared__ bool sharedBlock[sharedmem];    
    // __shared__ float3 sharedColor[sharedmem];    

    __shared__ float2 botL;
     __shared__ float2 topR;        
    const float boxL=invWidth *static_cast<float>(pixelX);
    const float boxR=invWidth *(static_cast<float>(pixelX+blocksize)+.5f);
    const float boxB=invHeight *static_cast<float>(pixelY);
    const float boxT=invHeight *(static_cast<float>(pixelY+blocksize)+.5f);
    if(threadIdx.x == 0)
    {
	botL.x = boxL;
	botL.y = boxB;       
    }
    if(threadIdx.x == blockDim.x -1)
    {
	topR.x = boxR;
	topR.y = boxT;
    }    
    const short limit = 64;
    short countIterations = 0;
    int startIdx = -1;		// 
    // if(numCirlesToRender > (limit<<1))
    // {
    	for(int J = 0; J < numCirlesToRender; J += sharedmem)
    	{
	    __syncthreads();
    		for(short I = threadIdx.x; I < sharedmem && I+J < numCirlesToRender; I+= blockDim.x)
    		{
    		    int indexofcircle = checkblock[(numCirlesToRender-1-I-J)+megablock*cuConstRendererParams.numCircles];
		    // assert((numCirlesToRender-1-I-J)>=0);
		    float3 pa= *(float3*)(&cuConstRendererParams.position[3*indexofcircle]);
    		    float rad =  cuConstRendererParams.radius[indexofcircle];
    		    sharedidx[I] = indexofcircle;
    		    bool test = circleInBox(pa.x,
    		    					    pa.y,
    		    					    rad,
    		    					    botL.x, topR.x, topR.y, botL.y);
    		    sharedBlock[I] =  test;
    		    if(test)
    		    {
    			sharedrad[I] = rad;
    			sharedp[I] = pa;
    		    }
    		}    
	    __syncthreads();
    	    for(short I = 0; I+J < numCirlesToRender && I < sharedmem; I++)
    	    {
    		if(sharedBlock[I])
    		{
    		    int indexofcircle = sharedidx[I];
    		    float3 p = sharedp[I];
    		    float  rad = sharedrad[I];
    		    bool cont = circleInBox(p.x,p.y,rad,
    							boxL, boxR, boxT, boxB);
    		    if(cont && (startIdx == -1)) 
    		    {
    			countIterations++;
    			startIdx=(countIterations >= limit)? indexofcircle:startIdx;
    		    }

    		}
    	    }
    	}    
    // }
    if(startIdx == -1)
	startIdx = 0;
    for(int J = 0; J < numCirlesToRender; J += sharedmem)
    {
	__syncthreads();
	    for(short I = threadIdx.x; I < sharedmem && I+J < numCirlesToRender; I+= blockDim.x)
	    {
		int indexofcircle = checkblock[I+J+megablock*cuConstRendererParams.numCircles];
		sharedp[I] = *(float3*)(&cuConstRendererParams.position[3*indexofcircle]);
		sharedrad[I] =  cuConstRendererParams.radius[indexofcircle];
		sharedidx[I] = indexofcircle;
		bool test = circleInBox(sharedp[I].x,
						    sharedp[I].y,
						    sharedrad[I],
						    botL.x, topR.x, topR.y, botL.y);
		sharedBlock[I] =  test ;
		// if(test)
		//     sharedColor[I]= *(float3*)&(cuConstRendererParams.color[3*indexofcircle]);

	    }
	__syncthreads();
	for(short I = 0; I+J < numCirlesToRender && I < sharedmem; I++)
	{
	    int indexofcircle = sharedidx[I];
	    if(sharedBlock[I]&& (indexofcircle >= startIdx))
	    {
		float3 p = sharedp[I];
		float  rad = sharedrad[I];
		bool cont = circleInBox(p.x,p.y,rad,
					boxL, boxR, boxT, boxB);
		if(cont) 
		{
		    // float3 inputrgb=sharedColor[I];
#pragma unroll
		    for(short K = 0; K < blocksize;K++)
		    {
			float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * ((K+pixelY) * imageWidth + pixelX)]);
#pragma unroll
			for(short J = 0; J < blocksize;J++)
			{
			    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(J+pixelX) + 0.5f),
								 invHeight * (static_cast<float>(K+pixelY) + 0.5f));
			    // shadePixel(indexofcircle, pixelCenterNorm, p, imgPtr,rad,maxDist);
			    shadePixel(indexofcircle, pixelCenterNorm, p, imgPtr,rad);
			    imgPtr++;
			}
		    }      	
		}
	    }
	}
	// __syncthreads();
    }//end of main shared parser        
}
__global__ void blockRender_alt_limit_small(int* checkblock,int* checkblock_size,short numboxes,int boxsize				      )   
{
    // #define blocksize 2
    const int imageHeight = cuConstRendererParams.imageHeight;
    const int imageWidth = cuConstRendererParams.imageWidth;
    const float invWidth = 1.f / imageWidth;
    const float invHeight = 1.f / imageHeight;
    
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    //calculate the array to look at for texture elemination
    // for(int I = 0; I < 1000 && I < cuConstRendererParams.numCircles; I++)
    const int numBlocksPerMega = boxsize*boxsize/blocksize/blocksize/blockDim.x;
    const int blockInMega= blockIdx.x&(numBlocksPerMega-1);
    const int megablock= blockIdx.x>>(blog2(numBlocksPerMega));

    // const int blockrowsize = boxsize / blocksize;
    const int dimSqrt = 1<<(blog2(blockDim.x)/2);
    const int dimBlockSqrt= 1<<(blog2(numBlocksPerMega)/2);
    const int rowInThread=  bmod(threadIdx.x,dimSqrt) + bmod(blockInMega,dimBlockSqrt)*dimSqrt;
    const int colInThread= (threadIdx.x>>blog2(dimSqrt))+(blockInMega>>blog2(dimBlockSqrt))*dimSqrt;
    
    const int MegaBoxDim=imageHeight/boxsize;
    const int megaRow= megablock&(MegaBoxDim-1);
    const int megaCol= megablock>>(31 - __clz(MegaBoxDim));
    
    const int numCirlesToRender= checkblock_size[megablock];

//Calculate the box to shade
    const int pixelX=megaRow*boxsize+blocksize*rowInThread;
    const int pixelY=megaCol*boxsize+blocksize*colInThread;//8x16
// #define sharedmem (768)        
    // __shared__ float3 sharedp[sharedmem];
    // __shared__ float sharedrad[sharedmem];
    // __shared__ int sharedidx[sharedmem];
    // __shared__ bool sharedBlock[sharedmem];

    // extern __shared__ char sharedmemearr[];
    // int *sharedidx = (int*)sharedmemearr; 
    // float3* sharedp = (float3*) &sharedmemearr[1024];
    // float *sharedrad = (float*)(&sharedp[1024]);
    // bool *sharedBlock = (bool*)(&sharedrad[1024]);
    // // assert((float*)(sharedp + sizeof(float3)*sharedmem) == sharedrad);
    // // assert((bool*)(sharedrad + sizeof(float)*sharedmem) == sharedBlock);
    const int sharedmem=512;
    __shared__ float3 sharedp[sharedmem];
    __shared__ float sharedrad[sharedmem];
    __shared__ int sharedidx[sharedmem];
    __shared__ bool sharedBlock[sharedmem];    
     __shared__ float2 botL;
      __shared__ float2 topR;        
    const float boxL=invWidth *static_cast<float>(pixelX);
    const float boxR=invWidth *(static_cast<float>(pixelX+blocksize)+.5f);
    const float boxB=invHeight *static_cast<float>(pixelY);
    const float boxT=invHeight *(static_cast<float>(pixelY+blocksize)+.5f);
    if(threadIdx.x == 0)
    {
	botL.x = boxL;
	botL.y = boxB;       
    }
    if(threadIdx.x == blockDim.x -1)
    {
	topR.x = boxR;
	topR.y = boxT;
    }    
    // const short limit = 64;
    // short countIterations = 0;
    int startIdx = 0;
	__syncthreads();

for(int J = 0; J < numCirlesToRender; J += sharedmem)
    {
	    for(short I = threadIdx.x; I < sharedmem && I+J < numCirlesToRender; I+= blockDim.x)
	    {
		int indexofcircle = checkblock[I+J+megablock*cuConstRendererParams.numCircles];			
		sharedp[I] = *(float3*)(&cuConstRendererParams.position[3*indexofcircle]);
		sharedrad[I] =  cuConstRendererParams.radius[indexofcircle];
		sharedidx[I] = indexofcircle;
		bool test = circleInBox(sharedp[I].x,
						    sharedp[I].y,
						    sharedrad[I],
						    botL.x, topR.x, topR.y, botL.y);
		sharedBlock[I] =  test ;
	    }
	__syncthreads();
	for(short I = 0; I+J < numCirlesToRender && I < sharedmem; I++)
	{
	    int indexofcircle = sharedidx[I];
	    if(sharedBlock[I]&& (indexofcircle >= startIdx))
	    {
		float3 p = *(float3*)(&cuConstRendererParams.position[3*indexofcircle]);
		float  rad = cuConstRendererParams.radius[indexofcircle];
		bool cont = circleInBox(p.x,p.y,rad,
					boxL, boxR, boxT, boxB);
		if(cont) 
		{
		    // countIterations--;			
#pragma unroll
		    for(short K = 0; K < blocksize;K++)
		    {
			float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * ((K+pixelY) * imageWidth + pixelX)]);
#pragma unroll			
			for(short J = 0; J < blocksize;J++)
			{
			    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(J+pixelX) + 0.5f),
								 invHeight * (static_cast<float>(K+pixelY) + 0.5f));
			    // shadePixel(indexofcircle, pixelCenterNorm, p, imgPtr,rad,maxDist);
			    shadePixel(indexofcircle, pixelCenterNorm, p, imgPtr,rad);
			    imgPtr++;
			}
		    }      	
		}
	    }
	}
	// __syncthreads();
    }//end of main shared parser        
}





__global__ void circle_filter(int* arrayout,short numboxes,int boxsize)   
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= cuConstRendererParams.numCircles)
	return;
    float3 p = *(float3*)(&cuConstRendererParams.position[3*index]);
    float  rad = cuConstRendererParams.radius[index];
    int imageHeight = cuConstRendererParams.imageHeight;
    int imageWidth = cuConstRendererParams.imageWidth;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // dim3 blockDim(256, 1);
    // dim3 gridDim_sub(((numboxes) + blockDim.x - 1) / blockDim.x);    
    
    // circle_filter_sub<<<gridDim_sub, blockDim>>>(arrayout,index,invHeight,invWidth);
    // cudaDeviceSynchronize();
    int xboxes = imageHeight/boxsize;
    for(int I = 0; I < numboxes;I++)
    {
    	int pixelY=boxsize*(I/xboxes);//8x16
    	int pixelX=boxsize*(I%xboxes);
    	float boxL,boxR,boxT,boxB;
    	boxL=invWidth *static_cast<float>(pixelX);
    	boxR=invWidth *(static_cast<float>(pixelX+boxsize) + .5f);
    	boxB=invHeight *static_cast<float>(pixelY);
    	boxT=invHeight *(static_cast<float>(pixelY+boxsize) + .5f);


    	bool cont = circleInBox(p.x,p.y,rad,
    					    boxL, boxR, boxT, boxB);
    	arrayout[(index)+(I*cuConstRendererParams.numCircles)]=cont;
    }

}
__global__ void circle_filter_find_circles(int* arrayin,int* arrayout,int* arrayout_size,short numboxes,int boxsize)   
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= cuConstRendererParams.numCircles)
	return;
    if(index == 0)
    {
	
	for(int I = 0; I < numboxes;I++)
	{
	    int test=I*cuConstRendererParams.numCircles + index;
	    int outputpart = I*cuConstRendererParams.numCircles+arrayin[test]-1;
	    if(arrayin[test] != 0)
		arrayout[outputpart]=index;      
	}
	return;
    }    
    for(int I = 0; I < numboxes;I++)
    {
	int test=I*cuConstRendererParams.numCircles + index;
	if(arrayin[test] != arrayin[test-1])
	{
	    int outputpart = I*cuConstRendererParams.numCircles+arrayin[test]-1;
	    arrayout[outputpart]=index;
	}
    }
    
    if(index ==( cuConstRendererParams.numCircles-1))
    {
	for(int I = 0; I < numboxes;I++)
	{
	    int test=I*cuConstRendererParams.numCircles + index;
	    arrayout_size[I] = arrayin[test];
	}
    }
    else
	return;
}


void
CudaRenderer::render() {

    // 256 threads per block is a healthy number
    int numBlocks = 8192<<5; //each block is 4x4 or the image is 256x256 boxes    
    int numRoughBlocks = 16;
    dim3 blockDim(512, 1);
    dim3 gridDim_Circles(((numCircles) + blockDim.x - 1) / blockDim.x);    

    // cudaMalloc(&boxoutarray, sizeof(short) *(1<<6)*numCircles);


    if(numCircles > 1000)
    {
	//Rough circle elimination with few boxes break up into 2^4 boxes so each box is 256x256 
	//
	size_t length = (1<<2)*numCircles; // 4 boxes
	int boxsize = 512;
	numRoughBlocks=4;

	if(numCircles > 3000)
	{
	    length = (1<<4)*numCircles; // 16 boxes
	    boxsize = 256;
	    numRoughBlocks=16;

	}       
	if(numCircles  > 20000) // 64 boxes
	{
	    length = (1<<8)*numCircles;
	    boxsize = 64;
	    numRoughBlocks = 256;
	}
	thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
	thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
	thrust::device_ptr<int> d_output_reduction = thrust::device_malloc<int>(numRoughBlocks);
	//     int max_size_array = (6000)*sizeof(int)*128;
	    // // thrust::device_free(d_input);
	    // thrust::device_ptr<int> d_fine_blocks = thrust::device_malloc<int>(max_size_array);

	circle_filter<<<gridDim_Circles, blockDim>>>(thrust::raw_pointer_cast(d_input),numRoughBlocks,boxsize);

	for(int roughbox=0;roughbox < numRoughBlocks;roughbox++)
	{
	    thrust::device_ptr<int> d_input_box=d_input+(roughbox*numCircles);
	    thrust::inclusive_scan(thrust::device,d_input_box, d_input_box + numCircles, d_input_box);
	}	

	circle_filter_find_circles<<<gridDim_Circles, blockDim>>>(thrust::raw_pointer_cast(d_input),
								  thrust::raw_pointer_cast(d_output),
								  thrust::raw_pointer_cast(d_output_reduction),
								  numRoughBlocks,boxsize);
	// thrust::device_free(d_input);

	if(numCircles > 3000)
	{
	    // int max_size_array = (6000)*sizeof(int)*128;
	    // // thrust::device_free(d_input);
	    // thrust::device_ptr<int> d_fine_blocks = thrust::device_malloc<int>(max_size_array);
	    // numBlocks = numBlocks >> 2;
	    blockDim.x=256;//must be powers of 4

	    dim3 gridDim_render(((numBlocks) + blockDim.x - 1) / blockDim.x);
		
	    // blockRender_alt_limit<<<gridDim_render, blockDim,1024*(sizeof(int)+sizeof(float3) + sizeof(float) + sizeof(bool))>>>(thrust::raw_pointer_cast(d_output),
	    // 					  thrust::raw_pointer_cast(d_output_reduction),
	    // 							numRoughBlocks,boxsize,1024);
	    // thrust::device_free(d_fine_blocks);
	    blockRender_alt_limit<<<gridDim_render, blockDim>>>(thrust::raw_pointer_cast(d_output),
						      thrust::raw_pointer_cast(d_output_reduction),
						      numRoughBlocks,boxsize);
	}
	else
	{
	    // numBlocks = numBlocks >> 2;
	    blockDim.x=256;//must be powers of 4
	    dim3 gridDim_render(((numBlocks) + blockDim.x - 1) / blockDim.x);
	    // blockRender_alt_limit<<<gridDim_render, blockDim,	512*sizeof(int)+512*sizeof(float3) + 512*sizeof(float) + 512*sizeof(bool)>>>(thrust::raw_pointer_cast(d_output),
	    // 						  thrust::raw_pointer_cast(d_output_reduction),
	    // 							numRoughBlocks,boxsize,512);
	    blockRender_alt_limit_small<<<gridDim_render, blockDim>>>(thrust::raw_pointer_cast(d_output),
								thrust::raw_pointer_cast(d_output_reduction),
								numRoughBlocks,boxsize);
	}
	cudaDeviceSynchronize();
	// printf("%d \n",thrust::max_element(thrust::device,d_output_reduction,d_output_reduction+numRoughBlocks));

	// for(int K = 0; K < numRoughBlocks;K++)
	// {
	//     printf("Index %d ",K);
	//     thrust::for_each(thrust::device,d_output_reduction+K,d_output_reduction+K+1, printf_functor_1());
	//     cudaDeviceSynchronize();

	// }
	    // thrust::device_free(d_fine_blocks);

	thrust::device_free(d_input);
	thrust::device_free(d_output);
	thrust::device_free(d_output_reduction);

    }
    else
    {
	// printf("Didn't filter Circles \n");
	dim3 gridDim_render(((numBlocks) + blockDim.x - 1) / blockDim.x);    
	blockRender<<<gridDim_render, blockDim>>>();
	


    }
    
//Finer circle elimination with more boxes
    //Finer elimination will spawn the child threads that fill in the image
    //
    cudaDeviceSynchronize();

}
