## Testat 3 

###### Fadil Smajilbasic

#### Task 1.1

Performance based on tile size:

| Tile size | Time elapsed (Pass 1) | Time elapsed (Pass 2) |
|-----------|-----------------------|-----------------------|
| 5         | 0.029                 | 0.028                 |
| 8         | 0.011                 | 0.011                 |
| 10        | 0.014                 | 0.014                 |
| 16        | 0.008                 | 0.008                 |
| 20        | 0.009                 | 0.009                 |
| 25        | 0.012                 | 0.012                 |
| 30        | 0.011                 | 0.011                 |
| 32        | 0.009                 | 0.009                 |
| 40        | Failed                | Failed                |

Overall, the trend seems to be that the larger the tile size, the faster the execution. With a bigger tile size each thread block receives more elements to calculate, thus reducing the number of memory accesses that each thread block has to do. A bigger tile size means that there is also more shared memory between the blocks allowing for better data reuse and reducing the memory traffic, thus improving the performance.

It seems like we get a better performance when the tile size is a power of 2.
If the tile size is greater than 32, the multiplications are not executed because there is probably a problem with boundaries and I'm receiving an error.

#### Task 1.2

