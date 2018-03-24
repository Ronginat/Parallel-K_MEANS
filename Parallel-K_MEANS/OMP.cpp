#include "Header.h"
#include <omp.h>


void calculateDiameterOmp(Cluster* clusters, Point* points, int amountOfPoints, int n, int k)
{
	double* largeClusters = (double*)calloc(k * omp_get_max_threads(), sizeof(double));

	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int i = 0; i < amountOfPoints; i++)
	{
		int idx = omp_get_thread_num();
		for (int j = i + 1; j < n; j++)
		{
			if (points[i].clusterID == points[j].clusterID)
			{
				double dist = distance(points[i].x, points[i].y, points[j].x, points[j].y);
				if (largeClusters[idx * k + points[i].clusterID] < dist)
					largeClusters[idx * k + points[i].clusterID] = dist;
			}
		}
	}

	//union of 4 arrays of clusters
	for (int i = 0; i < k; i++)
	{
		//for (int j = 0; j < omp_get_max_threads(); j++)
		for (int j = 0; j < omp_get_max_threads(); j++)
		{
			if (clusters[i].diameter < largeClusters[i + j * k])
				clusters[i].diameter = largeClusters[i + j * k];
		}
	}
	free(largeClusters);
}

void UpdateClustersAfterCudaGroupOmp(Cluster* clusters, Point* points, int n, int k)
{
	Cluster* largeClusters = (Cluster*)malloc(omp_get_max_threads() * k * sizeof(Cluster));
	Cluster* temp = largeClusters;
	for (int i = 0; i < omp_get_max_threads(); i++)
	{
		refreshClusters(&temp, k);
		temp += k;
	}
	int currentCluster = 0;
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for private(currentCluster)
	for (int i = 0; i < n; i++)
	{
		currentCluster = omp_get_thread_num() * k + points[i].clusterID;
		largeClusters[currentCluster].sumX += points[i].x;
		largeClusters[currentCluster].sumY += points[i].y;
		largeClusters[currentCluster].numOfPoints++;
	}

	for (int i = 0; i < k * omp_get_max_threads(); i++)
	{
		clusters[i % k].sumX += largeClusters[i].sumX;
		clusters[i % k].sumY += largeClusters[i].sumY;
		clusters[i % k].numOfPoints += largeClusters[i].numOfPoints;
	}
	free(largeClusters);
}


void initFlagsArrOmp(bool **flags, int n)
{
	omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		(*flags)[i] = false;
	}
}


bool mergeFlagsArrOmp(bool *flags, int n)
{
	omp_set_num_threads(omp_get_max_threads());
	bool finalFlag = false;
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		if (finalFlag)
			break;
		if (flags[i])
			finalFlag = true;
	}
	return finalFlag;
}