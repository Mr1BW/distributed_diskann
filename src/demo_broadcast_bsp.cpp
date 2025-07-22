// 演示利用MPI_Alltoallv实现所有进程间的广播通信的方式
#undef NDEBUG
#include <mpi.h>
#include "bsp/broadcast_bsp_framework.hpp"
#include <cassert>

class MyProcessor : public BSP::BroadcastProcessorInterface
{
public:
    void setup(const int32_t my_rank, const int32_t total_rank) override
    {
        printf("Hello, I am rank %d of %d processes\n", my_rank, total_rank);
    }

    int process(const int32_t superstep, const int32_t my_rank, const int32_t total_rank, const char *inbox_buf, const uint64_t inbox_buf_len, 
            std::vector<char> &outbox_buf, BSP::IterationStatus &status) override
    {
        if (superstep == 0)
        {
            // 模拟向缓冲区内填写数据，模拟不同rank的缓冲区长度不同的情况
            outbox_buf.resize(my_rank * 10);
            for(int i = 0; i < my_rank * 10; i++)
            {
                outbox_buf[i] = my_rank + i;
            }
            status.vote_to_halt = false;
        }
        if (superstep == 1)
        {
            // 从缓冲区读取数据并检查数据正确性
            assert((const int)inbox_buf_len == (0 + total_rank - 1) * total_rank / 2 * 10);
            int offset = 0;
            for(int i = 0; i < total_rank; i++)
            {
                int len = i * 10;
                for(int j = 0; j < len; j++)
                {
                    assert(inbox_buf[offset + j] == i + j);
                }
                offset += len;
            }
            status.vote_to_halt = true;
        }
        return 0;
    }

    void cleanup(const int32_t superstep, const int32_t my_rank, const int32_t total_rank) override
    {
        printf("Goodbye, I am rank %d of %d processes\n", my_rank, total_rank);
    }
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Hello, I am rank %d of %d processes\n", rank, size);
    MyProcessor processor;
    BSP::BroadcastBSPWorker worker(processor);
    worker.run();
    MPI_Finalize();
    return 0;
}