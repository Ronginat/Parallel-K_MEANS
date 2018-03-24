#ifndef __HEADER_H
#define __HEADER_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#include <string.h>
#include <stdarg.h>

#define _CRT_SECURE_NO_DEPRECATE
#pragma warning (disable : 4996)

#define NO_SWITCH_POINTS -1
#define SWITCH_POINTS -2

#define NUM_THREADS_IN_BLOCK 1000
struct Cluster
{
	int id;
	int numOfPoints;
	double centerX, centerY;
	double sumX, sumY;
	double diameter;
};

struct Point
{
	double x, y;
	double vx, vy;
	int clusterID;
};

int K_Means_Algorithm(int myid, int numprocs);

//cuda functions
int cudaRefreshPoints(Point* points, int n, double time);
int cudaClassifiedPoints(Cluster* clusters, Point* points, int n, int k, bool* isPointChangedCluster);
int freeResources(int size, ...);

//omp functions
bool mergeFlagsArrOmp(bool *flags, int n);
void initFlagsArrOmp(bool **flags, int n);
void calculateDiameterOmp(Cluster* clusters, Point* points, int amountOfPoints, int n, int k);
void UpdateClustersAfterCudaGroupOmp(Cluster* clusters, Point* points, int n, int k);

//sequencial functions
void readFromFile(Point** points, int* n, int* k, double* t, double* dt, int* limit, double* qm);
void printToFile(Cluster* clusters, int k, double q, double t);
void printPoints(Point* points, int numPoints);
void printClusters(Cluster* clusters, int numClusters);
double distance(double x1, double y1, double x2, double y2);
void initiateClusters(Cluster** clusters, Point* points, int k);
void recalculateCenters(Cluster** clusters, int k);
void refreshClusters(Cluster** clusters, int k);
double calculateQuality(Cluster* clusters, int k);
void calculateWorkForDiameterLoadBalancing(int N, int numprocs, int* amountOfDiameterWork, int* startOfDiameter);

//mpi function
void createMpiStructs(MPI_Datatype* PointMPIType, MPI_Datatype* ClusterMPIType);
void mpiMergeClustersAfterGroup(Cluster** clusters, Cluster* clusterFromAllProcs, int k, int numprocs);
void mpiMergeClustersAfterDiameter(Cluster** clusters, Cluster* clusterFromAllProcs, int k, int numprocs);


#endif // !__HEADER_H

