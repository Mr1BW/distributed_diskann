// 演示利用MPI_Alltoallv实现所有进程间的广播通信的方式
#undef NDEBUG
#include <mpi.h>
#include "bsp/broadcast_bsp_framework.hpp"
#include <cassert>
#include <ic.h>
#include "parlay/primitives.h"

class MyProcessor : public BSP::BroadcastProcessorInterface
{
public:
    void setup(const int32_t my_rank, const int32_t total_rank) override
    {
        printf("Hello, I am rank %d of %d processes\n", my_rank, total_rank);
    }

    int process(const int32_t superstep, const int32_t my_rank, const int32_t total_rank, const char *inbox_buf, const uint64_t inbox_buf_len, 
            std::vector<char> &outbox_buf,uint64_t &outbox_len, BSP::IterationStatus &status)
    {
        if (superstep == 0)
        {
            // 模拟向缓冲区内填写数据，模拟不同rank的缓冲区长度不同的情况
            std::vector<std::vector<unsigned int>> new_out_ = {
                {my_rank*10 + 1},
                {my_rank*10 + 2},
            };
            unsigned int all_size = 2; // 假设所有rank的邻居总数为2
            unsigned int node_count = 2;
            std::vector<unsigned int> offset = {0, 1};
            std::vector<unsigned int> new_out_lens = {1, 1};
            std::vector<unsigned int> new_node_id = {my_rank*2+1, my_rank*2+2};
            std::vector<char> compressed_buf(500);
            std::memcpy(compressed_buf.data(), &node_count, sizeof(unsigned int));
            for(int i = 0;i<new_out_.size();i++){
                std::memcpy(compressed_buf.data() + sizeof(unsigned int) + i * sizeof(unsigned int), &new_out_lens[i], sizeof(unsigned int));
                std::memcpy(compressed_buf.data() + sizeof(unsigned int) + new_out_.size() * sizeof(unsigned int) + offset[i] * sizeof(unsigned int), new_out_[i].data(), new_out_lens[i]*sizeof(unsigned int));
                std::memcpy(compressed_buf.data() + sizeof(unsigned int) + new_out_.size() * sizeof(unsigned int) + all_size * sizeof(unsigned int) + i * sizeof(unsigned int), &new_node_id[i], sizeof(unsigned int));
            }
            size_t src_len = sizeof(unsigned int) + new_out_.size() * sizeof(unsigned int) + all_size * sizeof(unsigned int) + new_node_id.size() * sizeof(unsigned int);
            std::cout<<"src_len: "<<src_len<<std::endl;
            for(int i=0;i<src_len;i++){
                std::cout<<static_cast<int>(compressed_buf[i])<<" ";
            }
            outbox_len = p4nenc32(reinterpret_cast<uint32_t*>(compressed_buf.data()), src_len/sizeof(unsigned int), reinterpret_cast<unsigned char*>(outbox_buf.data()+sizeof(size_t)*2));
            std::memcpy(outbox_buf.data(), &outbox_len, sizeof(size_t));//压后大小
            std::memcpy(outbox_buf.data() + sizeof(size_t), &src_len, sizeof(size_t));//原始数据大小
            outbox_len = outbox_len + 2*sizeof(size_t);
            std::cout<<"my_rank: "<<my_rank<<" outbox_len: "<<outbox_len<<std::endl;
            
            // for(int i=0;i<outbox_len;i++){
            //     std::cout<<static_cast<int>(*(outbox_buf.data()+i))<<" ";
            // }
            std::cout<<std::endl;
            std::cout<<"\033[31m[dbg] my_rank: "<<my_rank<<" outbox_len: "<<outbox_len<<"\033[0m\n";
        }
        if (superstep == 1 && my_rank == 0)
        {
            for(int i=0;i<inbox_buf_len;i++){
                std::cout<<static_cast<int>(*(inbox_buf + i))<<" ";
            }
            std::cout<<std::endl;
            std::cout<<std::endl;
            std::vector<std::vector<unsigned int>> recovered(total_rank*2 + 1);
            size_t buf_offset = 0;
            while (buf_offset < inbox_buf_len) {
            //读取节点数量
            size_t compressed_size;
            size_t src_size;
            std::memcpy(&compressed_size, inbox_buf + buf_offset, sizeof(size_t));
            buf_offset += sizeof(size_t);
            std::memcpy(&src_size, inbox_buf + buf_offset, sizeof(size_t));
            buf_offset += sizeof(size_t);
            //读取待解压的数据
            std::vector<char> compressed_data(compressed_size,0);
            std::memcpy(compressed_data.data(), inbox_buf + buf_offset, compressed_size);
            std::vector<unsigned int> src_data(src_size / sizeof(unsigned int));
            p4ndec32(reinterpret_cast<unsigned char*>(compressed_data.data()),src_size / sizeof(unsigned int),src_data.data());
            std::cout<<"src data size: "<<src_data.size()<<" int"<<std::endl;
            for(auto it :src_data){
                std::cout<<it<<" ";
            }
            std::cout<<std::endl;
            buf_offset += compressed_size;
            size_t sub_buf_offset = 0;
            const char* src_data_bytes = reinterpret_cast<const char*>(src_data.data());
            std::cout<<"compressed_size: "<<compressed_size<<" src_size: "<<src_size<<std::endl;
            for(int i=0;i<compressed_size;i++){
                std::cout<<static_cast<int>(compressed_data[i])<<" ";
            }
            std::cout<<std::endl;

            while(sub_buf_offset < src_size){
            //读取节点数量
            unsigned int node_count;
            std::memcpy(&node_count, src_data_bytes + sub_buf_offset, sizeof(unsigned int));
            std::cout<<"my_rank: "<<my_rank<<" node_count: "<<node_count<<" supersteps"<<superstep<<std::endl;
            //std::cout<<"[dbg] node_count: "<<node_count<<" supersteps"<<superstep<<std::endl;
            sub_buf_offset += sizeof(unsigned int);
            //读取所有邻居大小，计算偏移
            std::vector<unsigned int> lens(node_count);
            std::memcpy(lens.data(), src_data_bytes + sub_buf_offset, node_count * sizeof(unsigned int));
            sub_buf_offset += node_count * sizeof(unsigned int);
            auto scan_result = parlay::scan(lens);
            auto total_neighbor_size = scan_result.second;
            auto neighbor_offset =scan_result.first;
        
            //读取所有邻居数据
            const char* neighbor_data_start = src_data_bytes + sub_buf_offset;
            sub_buf_offset += total_neighbor_size * sizeof(unsigned int);
            //读取节点ID数组
            std::vector<unsigned int> node_ids(node_count);
            std::memcpy(node_ids.data(), src_data_bytes + sub_buf_offset, node_count * sizeof(unsigned int));
            sub_buf_offset += node_count * sizeof(unsigned int);
            // 解析每个节点的邻居数据
            for(size_t i = 0; i < node_count; ++i) {
            // 读取第 i 个节点的邻居数据
            //std::cout<<"supersteps"<< superstep<<"node_ids[i]: "<<node_ids[i]<<" lens[i]: "<<lens[i]<<std::endl;
                size_t neighbor_count = lens[i];
                std::vector<unsigned int> neighbors(neighbor_count);
                std::memcpy(neighbors.data(), 
                       neighbor_data_start + neighbor_offset[i]*sizeof(unsigned int), 
                       lens[i]*sizeof(unsigned int));
                recovered[node_ids[i]] = std::move(neighbors);
            };
            }
            }
            if(my_rank==0){
            for(int i=0;i<recovered.size();i++){
                std::cout<<"recovered["<<i<<"] size: "<<recovered[i].size()<<" neighbors: ";
                for(auto &it: recovered[i]){
                    std::cout<<it<<" ";
                }
                std::cout<<std::endl;
            }}
        }
        if(superstep == 2)
            status.vote_to_halt = true;
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