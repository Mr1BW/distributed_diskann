CFLAGS = -Iinclude/ -Llib/ -Wall -O2 -g -fno-omit-frame-pointer -std=c++17 -pthread
LINK_FLAGS = -lMapReduceMPI -lfmt -ltbb -lic
cc = mpic++
targets = build/parlayann build/distributed_diskann

build/parlayann:
	mpic++ -o $@ $(CFLAGS) -DORIGINAL_PARLAY src/diskann.cpp include/bsp/*.cpp $(LINK_FLAGS)

build/distributed_diskann:
	mpic++ -o $@ $(CFLAGS) src/diskann.cpp include/bsp/*.cpp $(LINK_FLAGS)

build/demo_broadcast_bsp:
	mpic++ -o $@ $(CFLAGS) src/demo_broadcast_bsp.cpp include/bsp/*.cpp $(LINK_FLAGS)
	
build/test:
	mpic++ -o $@ $(CFLAGS) src/test.cpp include/bsp/*.cpp $(LINK_FLAGS)

run: $(targets)
	mpirun -np 2 --bind-to core --map-by slot:pe=2 build/distributed_diskann

run1: $(targets)
	mpirun -np 1 --bind-to core --map-by slot:pe=2 build/distributed_diskann

run_parlay:
	build/parlayann

clean:
	rm -rf build/*
