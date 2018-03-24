#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Header.h"

cudaError_t refreshPointsMemoryWithCuda(Point *points, unsigned int n, double time);

__global__ void refreshPointsKernel(Point *points, int n, double time)
{
    int i = blockIdx.x * NUM_THREADS_IN_BLOCK + threadIdx.x;
	if (i < n)
	{
		points[i].x = points[i].x + time * points[i].vx;
		points[i].y = points[i].y + time * points[i].vy;
	}
}

int cudaRefreshPoints(Point* points, int n, double time)
{
    // Add vectors in parallel.
    cudaError_t cudaStatus = refreshPointsMemoryWithCuda(points, n, time);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "refreshPointsMemoryWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t refreshPointsMemoryWithCuda(Point *points, unsigned int n, double time)
{
    Point *dev_points = 0;
    cudaError_t cudaStatus;

	int numBlocks;
	if (n % NUM_THREADS_IN_BLOCK == 0)
		numBlocks = n / NUM_THREADS_IN_BLOCK;
	else
		numBlocks = (n / NUM_THREADS_IN_BLOCK) + 1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
    }
	
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_points, n * sizeof(Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		freeResources(1, dev_points);
		return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_points, points, n * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		freeResources(1, dev_points);
		return cudaStatus;
    }
	
    // Launch a kernel on the GPU with one thread for each element.
    refreshPointsKernel <<<numBlocks, NUM_THREADS_IN_BLOCK>>>(dev_points, n, time);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		freeResources(1, dev_points);
		return cudaStatus;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		freeResources(1, dev_points);
		return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(points, dev_points, n * sizeof(Point), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		freeResources(1, dev_points);
		return cudaStatus;
    }

	freeResources(1, dev_points);
    return cudaStatus;
}