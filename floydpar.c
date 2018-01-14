/**
* @fichier floydpar.c
* @titre Parallel Floyd-wharshall
* @description Implémentation de l'algo de Floyd-wharshall en parallèle avec MPI
* @auteur Kevin Estalella
* @date 14 Janvier 2018
* @version 1.0
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <mpi.h>
#include <sys/time.h>
#include <time.h>


// Variables gobales
int p_count, p_rank;
const int INF = -9999999;

/**
 * Imprime matrice d'adjacance
 * @param N       ordre matrice
 * @param result  résultat en 1D
 */
void printMatAdj(int N, int *result) {
    int i, j;
    printf("\n    ");
    for (i = 0; i < N; ++i)
        printf("%4c", 'A' + i);
    printf("\n");
    for (i = 0; i < N; ++i) {
        printf("%4c", 'A' + i);
        for (j = 0; j < N; ++j)
            if (result[(i*N)+j] == INF)
              printf("%4c", 'N');
            else
              printf("%4d", result[(i*N)+j]);
        printf("\n");
    }
    printf("\n");
}

/**
 * Crée la matrice d'adjacance aléatoirement
 * @param matrix_base   matrice
 * @param N     ordre de la matrice
 */
void CreateMatAdj(int *matrix_base, int N)
{
  srand(time(NULL)); // initialisation de rand

  int random_number;

  // Génére la matrice avec des valeurs aléatoires
  for(int i = 0; i < N*N; i ++) {
    random_number = (rand()%16) - 1;
		if (random_number == -1)
			matrix_base[i] = INF;
    else
		  matrix_base[i] = random_number;
	}
  // Met la diagonale à 0
	for(int i = 0; i < N; i++)
		matrix_base[i*N + i] = 0;
}

/**
 * Implementation de l'algo de floyd en parallel avec MPI
 * @param matrix_base    matrice
 * @param N              ordre de la matrice
 * @param matrix_part    partie de la matrice
 * @param result         rien
 */
void FloydPar(int *matrix_base, int N, int *matrix_part, int *result)
{
    int start, end, owner_of_k, offset_to_k, sum;
    int nb_val_per_node = N / p_count;
    int row_size = sizeof(int) * N;
    int *row_k = (int*)malloc(row_size); // stock la ligne k
    int j, i, k;

    // Distribue données à tous les noeuds -> process 0 (root) qui lance
    MPI_Scatter(matrix_base, N * nb_val_per_node, MPI_INT, matrix_part, N * nb_val_per_node, MPI_INT, 0, MPI_COMM_WORLD);

    for (k = 0; k < N; ++k)
    {
        // On récupère le propriétaire du k qui nous intéresse
        owner_of_k = k / nb_val_per_node;
        offset_to_k = (k % nb_val_per_node) * N;

        // Le noeud concerné par le k recherché broadcast sa ligne
        if (p_rank == owner_of_k)
            memcpy(row_k, matrix_part + offset_to_k, row_size);
        MPI_Bcast(row_k, N, MPI_INT, owner_of_k, MPI_COMM_WORLD);

        for (i = 0; i < nb_val_per_node; ++i)
        {
            for (j = 0; j < N; ++j)
            {
                if (row_k[j] != INF && matrix_part[i*N + k] != INF)
                {
                    sum = matrix_part[i*N + k] + row_k[j];
                    if (matrix_part[i*N + j] > sum || matrix_part[i*N + j] == INF) {
                      matrix_part[i*N + j] = sum;
                    }
                }
            }
        }
    }

    // Rassemble les valeurs calculées de tout les processus
    MPI_Gather(matrix_part, N * nb_val_per_node, MPI_INT, result, N * nb_val_per_node, MPI_INT, 0, MPI_COMM_WORLD);
}

/**
 * Permet de tester si un nombre est une puissance de 2
 * @param  x  valeur à tester
 * @return    vrai si le nombre est une puissance de 2
 */
int isPowerOfTwo(unsigned int x)
{
 while (((x % 2) == 0) && x > 1) /* While x is even and > 1 */
   x /= 2;
 return (x == 1);
}

int main(int argc, char **argv)
{
    int N;
    int *matrix_base, *result, *matrix_part;
    struct timeval tv_start, tv_end;

    if(argc != 2)
    {
        printf("\n\nErreur -> exemple d'utilisation: mpirun -np 2 ./test 4\n\n\n\n");
        return 1;
    }

    N = atoi(argv[1]);
    if(isPowerOfTwo(N) == false)
    {
        printf("\n\nErreur -> l'ordre de la matrice doit être le résultat d'un entier élevé à une puissance de 2\n\n\n\n");
        return 1;
    }

    MPI_Init(&argc, &argv); //-np val

    MPI_Comm_size(MPI_COMM_WORLD, &p_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_rank);

    matrix_part = (int*)malloc(sizeof(int) * N * (N / p_count)); // Morceau de chaque process

    // master process will compute the sequential version
    if (p_rank == 0)
    {
        int matrix_size = sizeof(int)*N*N;
        // Génère la matrice de base aléatoirement
        matrix_base = (int*)malloc(matrix_size);
        CreateMatAdj(matrix_base, N);
        printMatAdj(N,matrix_base);

        // Alloue espace pour la matrice de résultat
        result = (int*)malloc(matrix_size);
    }

    // On s'assure que tout les processes sont syncronisés avant de commancer
    MPI_Barrier(MPI_COMM_WORLD);

    if (p_rank == 0)
        gettimeofday(&tv_start, NULL);
    // Lance l'algo en parallèle
    FloydPar(matrix_base, N, matrix_part, result);
    if (p_rank == 0)
    {
        gettimeofday(&tv_end, NULL);
        printf("Temps total   = %ld microsecondes\n", (tv_end.tv_sec - tv_start.tv_sec) * 1000000 + tv_end.tv_usec - tv_start.tv_usec);
        printMatAdj(N,result);
    }

    MPI_Finalize();
}
