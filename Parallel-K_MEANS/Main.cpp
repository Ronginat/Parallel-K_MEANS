#include "Header.h"

int main(int argc, char* argv[])
{
	int myid, numprocs;
	MPI_Comm comm;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	if (numprocs < 3)
	{
		printf("Need to run at least 3 proccess\n");
		fflush(stdout);
		MPI_Finalize();
		return 1;
	}

	return K_Means_Algorithm(myid, numprocs);
}

int K_Means_Algorithm(int myid, int numprocs)
{
	//create structs
	MPI_Datatype PointMPIType;
	MPI_Datatype ClusterMPIType;
	createMpiStructs(&PointMPIType, &ClusterMPIType);

	Point *allPoints, *points;
	Cluster *clusters, *clustersFromAllProcs;
	int N, K, LIMIT;
	double dT, T, QM;
	double quality = 0, start, end;
	bool isPointsChangedCluster = false;
	double information[6];
	int *recvcountsPoints = (int*)malloc(numprocs * sizeof(int));
	int *displsPoints = (int*)malloc(numprocs * sizeof(int));
	int *flagsFromGroupPoints = (int*)malloc(numprocs * sizeof(int));
	int *flagsFromGroupPointsAfterGather = (int*)malloc(numprocs * sizeof(int));
	int *amountOfDiameterWork = (int*)malloc(numprocs * sizeof(int));
	int *startOfDiameter = (int*)malloc(numprocs * sizeof(int));

	// start
	start = MPI_Wtime();
	//one procces reads input
	if (myid == 0)
	{
		readFromFile(&allPoints, &N, &K, &T, &dT, &LIMIT, &QM);
		information[0] = N;
		information[1] = K;
		information[2] = LIMIT;
		information[3] = T;
		information[4] = dT;
		information[5] = QM;
	}

	MPI_Bcast(information, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	N = (int)information[0];
	K = (int)information[1];
	LIMIT = (int)information[2];
	T = information[3];
	dT = information[4];
	QM = information[5];

	//init recvcounts and displs for gathering points
	displsPoints[0] = 0;
	recvcountsPoints[0] = N / numprocs + N % numprocs;
	for (int i = 1; i < numprocs; i++)
	{
		recvcountsPoints[i] = N / numprocs;
		displsPoints[i] = displsPoints[i - 1] + recvcountsPoints[i - 1];
	}

	clusters = (Cluster*)malloc(K * sizeof(Cluster));
	clustersFromAllProcs = (Cluster*)malloc(K * numprocs * sizeof(Cluster));

	if (myid > 0)
		allPoints = (Point*)malloc(N * sizeof(Point));
	points = (Point*)malloc(N * sizeof(Point));

	MPI_Bcast(allPoints, N, PointMPIType, 0, MPI_COMM_WORLD);

	calculateWorkForDiameterLoadBalancing(N, numprocs, amountOfDiameterWork, startOfDiameter);
	//start of k-means algorithm
	initiateClusters(&clusters, allPoints, K);
	for (double t = 0; t < T; t += dT)
	{
		if (myid == 0)
		{
			printf("time = %f\n", t);
			fflush(stdout);
		}
		if (t != 0)
		{
			cudaRefreshPoints(allPoints + displsPoints[myid], recvcountsPoints[myid], dT);
			//one procces gathers all modified points
			MPI_Gatherv(allPoints + displsPoints[myid], recvcountsPoints[myid], PointMPIType, points, recvcountsPoints, displsPoints, PointMPIType, 0, MPI_COMM_WORLD);
			if (myid == 0)
				memcpy(allPoints, points, N * sizeof(Point));

			MPI_Bcast(allPoints, N, PointMPIType, 0, MPI_COMM_WORLD);
		}
		refreshClusters(&clusters, K);// O(K) method

		for (int i = 0; i < LIMIT; i++)
		{
			if (myid == 0)
			{
				printf("\ti = %d\n", i);
				fflush(stdout);
			}

			cudaClassifiedPoints(clusters, allPoints + displsPoints[myid], recvcountsPoints[myid], K, &isPointsChangedCluster);
			UpdateClustersAfterCudaGroupOmp(clusters, allPoints + displsPoints[myid], recvcountsPoints[myid], K);
			isPointsChangedCluster ? flagsFromGroupPoints[myid] = SWITCH_POINTS : flagsFromGroupPoints[myid] = NO_SWITCH_POINTS;

			//one procces gathers all flags
			MPI_Gather(flagsFromGroupPoints + myid, 1, MPI_INT, flagsFromGroupPointsAfterGather, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(flagsFromGroupPointsAfterGather, numprocs, MPI_INT, 0, MPI_COMM_WORLD);

			//one procces gathers all modified clusters
			MPI_Gather(clusters, K, ClusterMPIType, clustersFromAllProcs, K, ClusterMPIType, 0, MPI_COMM_WORLD);
			mpiMergeClustersAfterGroup(&clusters, clustersFromAllProcs, K, numprocs); // O(K) method
			MPI_Bcast(clusters, K, ClusterMPIType, 0, MPI_COMM_WORLD);

			recalculateCenters(&clusters, K); // O(K) method

			isPointsChangedCluster = false;
			for (int j = 0; j < numprocs; j++)
			{
				if (flagsFromGroupPointsAfterGather[j] == SWITCH_POINTS)
				{
					isPointsChangedCluster = true;
					break;
				}
			}
			if (!isPointsChangedCluster)
				break;
		}
		//one procces gathers all modified points
		//all proccesses need to have all the points because the diameter calculation is devided differently than the points classification
		MPI_Gatherv(allPoints + displsPoints[myid], recvcountsPoints[myid], PointMPIType, points, recvcountsPoints, displsPoints, PointMPIType, 0, MPI_COMM_WORLD);
		if (myid == 0)
			memcpy(allPoints, points, N * sizeof(Point));
		MPI_Bcast(allPoints, N, PointMPIType, 0, MPI_COMM_WORLD);

		calculateDiameterOmp(clusters, allPoints + startOfDiameter[myid], amountOfDiameterWork[myid], N - startOfDiameter[myid], K); //Load Balancing is critical

		//one procces gathers all modified clusters
		MPI_Gather(clusters, K, ClusterMPIType, clustersFromAllProcs, K, ClusterMPIType, 0, MPI_COMM_WORLD);
		MPI_Bcast(clustersFromAllProcs, K*numprocs, ClusterMPIType, 0, MPI_COMM_WORLD);
		mpiMergeClustersAfterDiameter(&clusters, clustersFromAllProcs, K, numprocs); // O(K) method

		quality = calculateQuality(clusters, K);
		if (myid == 0)
			printf("quality = %f\n", quality);
		if (quality <= QM)
		{
			if (myid == 0)
			{
				printf("first occurence at t = %f with q = %f\n", time, quality);
				printf("centers of the clusters\n");
				for (int i = 0; i < K; i++)
				{
					printf("%f\t%f\n", clusters[i].centerX, clusters[i].centerY);
					fflush(stdout);
				}
				printToFile(clusters, K, quality, t);
			}
			break;
		}
	}
	end = MPI_Wtime();
	if (myid == 0)
	{
		printf("time = %f\n", end - start);
		fflush(stdout);
	}

	free(allPoints);
	free(clusters);
	free(clustersFromAllProcs);
	free(recvcountsPoints); free(displsPoints); free(flagsFromGroupPoints); free(amountOfDiameterWork); free(startOfDiameter);

	MPI_Finalize();
	return 0;
}