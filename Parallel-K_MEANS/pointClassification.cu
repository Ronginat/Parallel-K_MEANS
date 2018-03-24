#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Header.h"

cudaError_t classifiedPointsMemoryWithCuda(Cluster* clusters, Point *points, unsigned int n, unsigned int k, bool* isPointChangedCluster);

__device__ double Distance(double x1, double y1, double x2, double y2)
{
	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

__global__ void classifiedPointsKernel(Cluster* clusters, Point *points, bool *flags, unsigned int k, unsigned int n)
{
	int idx = blockIdx.x * NUM_THREADS_IN_BLOCK + threadIdx.x;
	if (idx < n)
	{
		int minIndex = -1;
		double minDistance = DBL_MAX;
		for (int i = 0; i < k; i++)
		{
			double distanceTmp = Distance(points[idx].x, points[idx].y, clusters[i].centerX, clusters[i].centerY);
			if (distanceTmp < minDistance)
			{
				minDistance = distanceTmp;
				minIndex = i;
			}
		}
		if (points[idx].clusterID != minIndex)
			flags[idx] = true;

		points[idx].clusterID = minIndex;
	}
}


int cudaClassifiedPoints(Cluster* clusters, Point* points, int n, int k, bool* isPointChangedCluster)
{
	*isPointChangedCluster = false;
	cudaError_t cudaStatus = classifiedPointsMemoryWithCuda(clusters, points, n, k, isPointChangedCluster);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "groupPointsMemoryWithCuda failed!");
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
cudaError_t classifiedPointsMemoryWithCuda(Cluster* clusters, Point *points, unsigned int n, unsigned int k, bool* isPointChangedCluster)
{
	Cluster *dev_clusters;
	Point *dev_points;
	cudaError_t cudaStatus;
	bool* dev_flags;
	bool* flags = (bool*)malloc(n * sizeof(bool));
	int numBlocks;

	initFlagsArrOmp(&flags, n);
	/*for (int i = 0; i < n; i++)
	flags[i] = false;*/

	//dim3 dimGrid(numBlocks, 4, 4);
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
	cudaStatus = cudaMalloc((void**)&dev_clusters, k * sizeof(Cluster));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc clusters failed!");
		freeResources(1, dev_clusters);
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_points, n * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc points failed!");
		freeResources(2, dev_points, dev_clusters);
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_flags, n * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc flags failed!");
		freeResources(3, dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}
	
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, n * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy points failed!");
		freeResources(3, dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_clusters, clusters, k * sizeof(Cluster), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		freeResources(3, dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_flags, flags, n * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy flags failed!");
		freeResources(3, dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}
	// Launch a kernel on the GPU with one thread for each point.
	classifiedPointsKernel << <numBlocks, NUM_THREADS_IN_BLOCK >> >(dev_clusters, dev_points, dev_flags, k, n);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "classifiedPointsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		freeResources(3, dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching classifiedPointsKernel!\n", cudaStatus);
		freeResources(3, dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, n * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy points failed!");
		freeResources(3, dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(flags, dev_flags, n * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy flags failed!");
		freeResources(3, dev_points, dev_clusters, dev_flags);
		return cudaStatus;
	}

	*isPointChangedCluster = mergeFlagsArrOmp(flags, n);

	free(flags);
	freeResources(3, dev_points, dev_clusters, dev_flags);

	return cudaStatus;
}

int freeResources(int size, ...)
{
	cudaError cudaStatus;
	va_list list;
	va_start(list, size);
	for (int i = 0; i < size; i++)
	{
		cudaStatus = cudaFree(va_arg(list, void*));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaFree failed!");
			return 1;
		}
	}
	va_end(list);
	return 0;
}