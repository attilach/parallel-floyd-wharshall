mpicc -o floydpar floydpar.c -lm

mpirun -np 2 ./floydpar
