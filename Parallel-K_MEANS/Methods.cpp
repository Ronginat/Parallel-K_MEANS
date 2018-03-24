#include "Header.h"
//#include "cuda_runtime.h"

void readFromFile(Point** points, int* n, int* k, double* t, double* dt, int* limit, double* qm)
{
	FILE* f = fopen("D:\\Final_K_Means_RonGinat\\Parallel-K_MEANS\\Parallel-K_MEANS\\input.txt", "r");
	int offset;
	offset = fscanf(f, "%d", n); // N - number of points
	offset = fscanf(f, "%d", k);// K - number of clusters to find
	offset = fscanf(f, "%lf", t);// T - defines the end of time interval
	offset = fscanf(f, "%lf", dt);// dT - defines moments t = n*dT, n = { 0, 1, 2 _, T/dT} for which calculate the clusters and the quality
	offset = fscanf(f, "%d", limit);// LIMIT - the maximum number of iterations for K-MEANS algorithm
	offset = fscanf(f, "%lf", qm);// QM - quality measure to stop

	*points = (Point*)malloc(*n * sizeof(Point));

	for (int i = 0; i < *n; i++)
	{
		offset = fscanf(f, "%lf", &((*points)[i].x));
		offset = fscanf(f, "%lf", &((*points)[i].y));
		offset = fscanf(f, "%lf", &((*points)[i].vx));
		offset = fscanf(f, "%lf", &((*points)[i].vy));
		(*points)[i].clusterID = -1;
	}
	fclose(f);
}

void printToFile(Cluster* clusters, int k, double q, double t)
{
	FILE* f = fopen("D:\\Final_K_Means_RonGinat\\Parallel-K_MEANS\\Parallel-K_MEANS\\output.txt", "w");
	fprintf(f, "First occurrence at t = %f with q = %f\n", t, q);
	fprintf(f, "Centers of the clusters :\n");

	for (int i = 0; i < k; i++)
	{
		fprintf(f, "%f\t%f\n", clusters[i].centerX, clusters[i].centerY);
	}
	fclose(f);
}

void printPoints(Point* points, int numPoints)
{
	for (int i = 0; i < numPoints; i++)
	{
		printf("points[%d] = xy(%f,%f)   v(%f,%f)\n", i, points[i].x, points[i].y, points[i].vx, points[i].vy);
		fflush(stdout);
		//printf("(%f,%f) - %d \n", points[i].x, points[i].y, points[i].clusterID);
	}
}

void printClusters(Cluster* clusters, int numClusters)
{
	for (int i = 0; i < numClusters; i++)
	{
		printf("%d:  center(%f,%f) diameter = %f \n", i, clusters[i].centerX, clusters[i].centerY, clusters[i].diameter);
		fflush(stdout);
	}
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

void initiateClusters(Cluster** clusters, Point* points, int k)
{
	for (int i = 0; i < k; i++)
	{
		(*clusters)[i].id = i;
		(*clusters)[i].numOfPoints = 0;
		(*clusters)[i].sumX = 0;
		(*clusters)[i].sumY = 0;
		(*clusters)[i].diameter = 0;
		(*clusters)[i].centerX = points[i].x;
		(*clusters)[i].centerY = points[i].y;
	}
}

void refreshClusters(Cluster** clusters, int k)
{
	for (int i = 0; i < k; i++)
	{
		//(*clusters)[i].id = i;
		(*clusters)[i].numOfPoints = 0;
		(*clusters)[i].sumX = 0;
		(*clusters)[i].sumY = 0;
		(*clusters)[i].diameter = 0;
	}
}

void recalculateCenters(Cluster** clusters, int k)
{
	for (int i = 0; i < k; i++)
	{
		(*clusters)[i].centerX = (*clusters)[i].sumX / (*clusters)[i].numOfPoints;
		(*clusters)[i].centerY = (*clusters)[i].sumY / (*clusters)[i].numOfPoints;
		(*clusters)[i].sumX = 0;
		(*clusters)[i].sumY = 0;
		(*clusters)[i].numOfPoints = 0;
		(*clusters)[i].diameter = 0;
	}
}

double calculateQuality(Cluster* clusters, int k)
{
	double sum = 0;
	for (int i = 0; i < k; i++)
	{
		for (int j = 0; j < k; j++)
		{
			if (i != j)
			{
				double dist = distance(clusters[i].centerX, clusters[i].centerY, clusters[j].centerX, clusters[j].centerY);
				sum += clusters[i].diameter / dist;
			}
		}
	}
	return sum / (k*(k - 1));
}

void createMpiStructs(MPI_Datatype* PointMPIType, MPI_Datatype* ClusterMPIType)
{
	struct Point point;
	struct Cluster cluster;

	// mpi datatype Point
	MPI_Datatype type[5] = { MPI_DOUBLE, MPI_DOUBLE , MPI_DOUBLE , MPI_DOUBLE , MPI_INT };
	int blocklen[5] = { 1,1,1,1,1 };
	MPI_Aint disp[5];

	disp[0] = (char*)&point.x - (char*)&point;
	disp[1] = (char*)&point.y - (char*)&point;
	disp[2] = (char*)&point.vx - (char*)&point;
	disp[3] = (char*)&point.vy - (char*)&point;
	disp[4] = (char*)&point.clusterID - (char*)&point;

	MPI_Type_create_struct(5, blocklen, disp, type, PointMPIType);
	MPI_Type_commit(PointMPIType);

	// mpi datatype Cluster
	MPI_Datatype type1[7] = { MPI_INT, MPI_INT, MPI_DOUBLE , MPI_DOUBLE , MPI_DOUBLE , MPI_DOUBLE, MPI_DOUBLE };
	int blocklen1[7] = { 1,1,1,1,1,1,1 };
	MPI_Aint disp1[7];

	disp1[0] = (char*)&cluster.id - (char*)&cluster;
	disp1[1] = (char*)&cluster.numOfPoints - (char*)&cluster;
	disp1[2] = (char*)&cluster.centerX - (char*)&cluster;
	disp1[3] = (char*)&cluster.centerY - (char*)&cluster;
	disp1[4] = (char*)&cluster.sumX - (char*)&cluster;
	disp1[5] = (char*)&cluster.sumY - (char*)&cluster;
	disp1[6] = (char*)&cluster.diameter - (char*)&cluster;

	MPI_Type_create_struct(7, blocklen1, disp1, type1, ClusterMPIType);
	MPI_Type_commit(ClusterMPIType);
}

void mpiMergeClustersAfterGroup(Cluster** clusters, Cluster* clusterFromAllProcs, int k, int numprocs)
{
	for (int i = k; i < k * numprocs; i++)
	{
		(*clusters)[i % k].numOfPoints += clusterFromAllProcs[i].numOfPoints;
		(*clusters)[i % k].sumX += clusterFromAllProcs[i].sumX;
		(*clusters)[i % k].sumY += clusterFromAllProcs[i].sumY;
	}
}

void mpiMergeClustersAfterDiameter(Cluster** clusters, Cluster* clusterFromAllProcs, int k, int numprocs)
{
	for (int i = 0; i < k * numprocs; i++)
	{
		if ((*clusters)[i%k].diameter < clusterFromAllProcs[i].diameter)
			(*clusters)[i%k].diameter = clusterFromAllProcs[i].diameter;

	}
}


/*calculating the work of diameter calculation for each procces for load balancing
* calcualting the sum of work that is sum of an arithmetic series the is N-2 + N-3 + N-4 + ... + 1
* eventually all proccess will have excatly third of the work. 
* N-1 because we need to summarize from 0 and not from 1. so we need the sum from 0 to n-1.
* extra N-1 (N-2) because the work needed to do for the last point is zero, so no need to calculate all the points, but N-2 points.
*/
void calculateWorkForDiameterLoadBalancing(int N, int numprocs, int* amountOfDiameterWork, int* startOfDiameter)
{
	N -= 2;
	long long sumOfPoints = ((long long)N)*((long long)(N + 1)) / 2;
	N += 2;
	long long optimalAmountForDiameter = sumOfPoints / numprocs;
	long long sumOfWorkCurrentProc = 0;
	long long currProc = 0, sumOfAllWork = 0, currProcNumOfPoints = 0;
	for (int i = N - 1; i > 0 && currProc < numprocs; i--)
	{
		sumOfWorkCurrentProc += i;
		currProcNumOfPoints++;
		if (sumOfWorkCurrentProc + i - 1 > optimalAmountForDiameter || i == 0)
		{
			int myDifference = optimalAmountForDiameter - sumOfWorkCurrentProc;
			int nextDifference = sumOfWorkCurrentProc + i - 1 - optimalAmountForDiameter;
			if (myDifference <= nextDifference)
			{
				amountOfDiameterWork[currProc] = currProcNumOfPoints;
			}
			else
			{
				amountOfDiameterWork[currProc] = currProcNumOfPoints + 1;
				i--;
			}
			currProc == 0 ? startOfDiameter[currProc] = 0 : startOfDiameter[currProc] = startOfDiameter[currProc - 1] + amountOfDiameterWork[currProc - 1]; // optional mistake needed +1
			sumOfAllWork += amountOfDiameterWork[currProc];
			currProc++;
			currProcNumOfPoints = 0;
			sumOfWorkCurrentProc = 0;
		}
	}
	int remainsOfPoints = N - sumOfAllWork;
	amountOfDiameterWork[numprocs - 1] += remainsOfPoints;
	
}
