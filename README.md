# distributed_diskann

## Project Origin

This project is based on [parlayANN](https://github.com/cmuparlay/ParlayANN), licensed under the MIT License

Reference: [ParlayANN: Scalable and Deterministic Parallel Graph-Based Approximate Nearest Neighbor Search Algorithms](https://doi.org/10.1145/3627535.3638475)



## Dataset Preparation

Small dataset

By default, the parameter files are `random-xs.json` and `random-s.json` located in the `data/` directory, with sizes of 10K and 100K respectively

You can also download it again, and the downloaded dataset will be stored in the `data/` directory

```
python3.10 create_dataset.py --dataset random-xs --download
python3.10 create_dataset.py --dataset random-s --download
```

Download of large datasets

```
python3.10 create_dataset.py --dataset msspacev-10M --download
python3.10 create_dataset.py --dataset msspacev-1M --download
```

Download the `Microsoft spacev` dataset, with sizes of 10M and 1M

```
python3.10 create_dataset.py --dataset text2image-10M --download
python3.10 create_dataset.py --dataset bigann-10M --download
```

Download the bigann and text2image datasets

There is a configuration file for the corresponding dataset in the project



## Dependencies

This project relies on the following libraries. Please ensure to install them before building:

- **MPI** (>=4.1.2)  -Parallel computing
- **OpenMPI**   -High-performance MPI implementation
- **tbb**   -Parallel programming
- **fmt**   -Modern C++ formatting library
- **mapreduceMPI**   -MapReduce framework for MPI

```
sudo apt update && sudo apt install -y \
  libopenmpi-dev \    # MPI + OpenMPI
  libtbb-dev \        # TBB
  libfmt-dev          # fmt
```

- **lz4**   -Compression algorithm  [lz4 download](https://github.com/powturbo/TurboPFor-Integer-Compression)

- **TurboPFor**  -Integer Compression  [TurboPFor download](https://github.com/powturbo/TurboPFor-Integer-Compression)

## Compilation

This project uses **Makefile**  for compilation

Compile the original non-distributed version of parlayann

```
make build/parlayann
```

Compile the distributed version

```
make build/distributed_diskann
```

The compiled files will be stored in the `build/` directory

## Operation

Non-distributed version running

```
./build/parlayann
```

It will read the running parameters and the required dataset from `para.json`

-------

Distributed version operation

```
mpirun -np X ./build/distributed_diskann
```

X processes will be initiated to run the program

parameters from `para.json`

## Parameters

In para.json

- **R** : the degree bound. Typically between 32 and 128.

- **L** : the beam width to use when building the graph. Should be set at least 30% higher than $R$, and up to 500.

- **alpha** : the pruning parameter. Should be set at 1.0 for similarity measures that are not metrics (e.g. maximum inner product), and between 1.0 and 1.4 for metric spaces. 

- **max** : Number of insertion points ( max = 0 default all points in dataset )

- **data_para** : the dataset json filename

In the dataset configuration JSON

- **nb** : the number of points

- **nq** ：the number of query points

- **d** : dimension

- **basedir** : dataset path

- **ds_fn** : dataset filename

- **qs_fn** : query filename

- **gt_fn** : ground truth filename



## Acknowledgements

- [ParlayANN](https://github.com/magdalendobson/ParlayANN-ppopp24/tree/ppopp-artifact-final-2?tab=readme-ov-file) - MIT License

- [TurboPFor-Integer-Compression: Fastest Integer Compression](https://github.com/powturbo/TurboPFor-Integer-Compression) - GPL-2.0 license
- [lz4: Extremely Fast Compression algorithm](https://github.com/lz4/lz4) -  BSD 2-Clause license
