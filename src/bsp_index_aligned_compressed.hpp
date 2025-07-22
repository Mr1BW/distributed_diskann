#include <cstdint>
#include <fmt/core.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <random>
#include <set>
#include <atomic>
#include <stdio.h>
#include <lz4.h>
//#include "bsp/bsp_framework.hpp"
#include "bsp/broadcast_bsp_framework.hpp"
#include "utils/NSGDist.h"
#include "utils/point_range.h"
#include "utils/graph.h"
#include "utils/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "utils/beamSearch.h"

#include "dataset.hpp"

#define COMPRESSION_THRESHOLD 1024

using indexType = unsigned int;
struct GraphBuffer {//图索引用
    indexType* node_ptr; // 邻居指针
    size_t size; // 节点数量
    indexType val;
};

struct GraphBuffer_recv{
    indexType val;
    parlay::sequence<indexType> seq;
};

// 序列化图索引 - 序列化指针指向的内容
void serialize(const GraphBuffer& s, std::string& buf) {
    // 序列化size
    buf.append(reinterpret_cast<const char*>(&s.size), sizeof(size_t));
    
    // 序列化指针指向的数据内容
    if (s.size > 0 && s.node_ptr != nullptr) {
        buf.append(reinterpret_cast<const char*>(s.node_ptr), s.size * sizeof(indexType));
    }
    
    // 序列化val
    buf.append(reinterpret_cast<const char*>(&s.val), sizeof(indexType));
}

// 反序列化图索引 - 返回 GraphBuffer_recv 类型
GraphBuffer_recv deserialize(const std::string& buf) {
    size_t offset = 0;
    GraphBuffer_recv result;
    
    // 反序列化size
    size_t size;
    std::memcpy(&size, buf.data() + offset, sizeof(size_t));
    offset += sizeof(size_t);
    
    // 反序列化指针指向的数据内容到 parlay::sequence
    if (size > 0) {
        result.seq.resize(size);
        std::memcpy(result.seq.data(), buf.data() + offset, size * sizeof(indexType));
        offset += size * sizeof(indexType);
    }
    
    // 反序列化val
    std::memcpy(&result.val, buf.data() + offset, sizeof(indexType));
    
    return result;
}




template<typename Point, typename PointRange, typename indexType>
struct BSP_index {
    using distanceType = typename Point::distanceType;
    using pid = std::pair<indexType, distanceType>;
    using PR = PointRange;
    using GraphI = Graph<indexType>;

class insert: public BSP::BroadcastProcessorInterface
{

public:
    GraphI G;
    PointRange Points;
    BuildParams BP;
    stats<indexType> BuildStats;
    double local_start_time = 0, local_end_time = 0;
    indexType start_point = 0;
    int my_id;
    //每个进程分配的节点索引
    size_t start = 0;
    size_t end = 0;

    size_t n;//图的大小
    size_t m;//插入节点的数量
    size_t inc = 0; // 批次计数器
    size_t count = 0; // 已处理节点数
    float frac = 0.0; // 进度百分比
    size_t max_batch_size;//最大处理批次
    double alpha = 1.2;
    double max_fraction = 0.02;
    double base = 2;

    parlay::sequence<indexType> inserts;//插入点的索引
    parlay::sequence<int> rperm;
    parlay::sequence<indexType> shuffled_inserts;
    parlay::sequence<parlay::sequence<indexType>> new_out_;
    
    parlay::internal::timer all_time;
    parlay::internal::timer comm_time;
    parlay::internal::timer t_prune;
    parlay::internal::timer beam_time;
    parlay::internal::timer supersteps_time;
    parlay::internal::timer flatten_time;
    parlay::internal::timer flatten_time_t1;
    parlay::internal::timer flatten_time_t2;
    parlay::internal::timer flatten_time_t3;
    dataset_para dp;
    std::atomic<int64_t> beam_global = 0;
    std::atomic<int64_t> update_global = 0;
    
    size_t floor;
    size_t ceiling;
    std::ofstream log;

    insert(BuildParams &BP, Graph<unsigned int>& G, PointRange Points,stats<indexType> BuildStats, dataset_para &dp){
        this->BP = BP;
        this->G = G;
        this-> Points = Points;
        this-> BuildStats = BuildStats;
        this-> alpha = dp.alpha;
        this-> max_fraction = dp.xita;
        start_point = dp.start_point_index;
        this->dp = dp;
        update_global.store(0);
    }

    void setup(const int32_t my_rank, const int32_t total_rank){
        all_time.reset();
        all_time.stop();
        comm_time.reset();
        comm_time.stop();
        t_prune.reset();
        t_prune.stop();
        beam_time.reset();
        beam_time.stop();
        supersteps_time.reset();
        supersteps_time.stop();
        flatten_time.reset();
        flatten_time.stop();

        n = G.size(); //图的大小
        inserts = parlay::tabulate(Points.size(), [&] (size_t i){
            return static_cast<indexType>(i);});
        m = inserts.size(); //插入点的数量
        
        rperm = parlay::tabulate(m, [&](int i) { return i; });
        shuffled_inserts = parlay::tabulate(m, [&](size_t i) { return inserts[rperm[i]]; });
        if(dp.max_build!=0)m=std::min(dp.max_build,m);
        //计算最多插入的大小
        max_batch_size = std::min(
        static_cast<size_t>(max_fraction * static_cast<float>(m)), 1000000ul);
        if(max_batch_size == 0) max_batch_size = m;

        std::cout<<"max_batch_size: "<<max_batch_size<<std::endl;
        my_id = my_rank;
        all_time.start();

    }

    int process(const int32_t superstep, const int32_t my_rank, const int32_t total_rank, const char *inbox_buf, const uint64_t inbox_buf_len, 
            std::vector<char> &outbox_buf, BSP::IterationStatus &status)
    {
        supersteps_time.stop();
        double start_time =MPI_Wtime();
        //要return 0否则不会进入下一步
        //std::cout<<count<<"  "<<m<<std::endl;

        if(superstep%2 == 0){//偶数次superstep，第一个parallel for，即 greedySearch那部分
            comm_time.start();
            size_t recv_offset = 0;
            while(recv_offset <inbox_buf_len){
            int compressed_size = *reinterpret_cast<const int*>(inbox_buf + recv_offset);
            recv_offset += sizeof(int);
            int src_size = *reinterpret_cast<const int*>(inbox_buf + recv_offset);
            recv_offset += sizeof(int);
            //解压缩
            std::vector<char> src_data(src_size);
            int decompressed_size = LZ4_decompress_safe(inbox_buf + recv_offset, src_data.data(), compressed_size, src_size);
            recv_offset += compressed_size;
            
            size_t recv_counts = decompressed_size / ((BP.R + 1) * sizeof(indexType));
            parlay::sequence<parlay::sequence<indexType>> recovered(ceiling - floor);
            parlay::parallel_for(0, recv_counts, [&](size_t i) {
                const indexType* block_start = reinterpret_cast<const indexType*>(src_data.data() + i * (BP.R + 1) * sizeof(indexType));
                
                indexType node_index = *(block_start + BP.R);
                const indexType* end_pos = std::find(block_start, block_start + BP.R, -1);
                if (end_pos != block_start) {
                    G[node_index].update_neighbors(parlay::make_slice(block_start, end_pos));
                }
            });
            }
            comm_time.stop();

        if (count >= m)
        {
            status.vote_to_halt = true;
            supersteps_time.stop();
            return 0;
        }
            if (pow(base, inc) <= max_batch_size) {
                floor = static_cast<size_t>(pow(base, inc)) - 1;
                ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)), m) - 1;
                count = std::min(static_cast<size_t>(pow(base, inc + 1)), m) - 1;
              } else {
                floor = count;
                ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
                count += static_cast<size_t>(max_batch_size);
            }
            size_t total = ceiling - floor;
            size_t base_len = total / total_rank;
            int remainder = total % total_rank; 
            
            start = floor + my_id * base_len + std::min(my_id, remainder);
            end = start + base_len + (my_id < remainder ? 1 : 0);
            std::cout<<"\033[31m[dbg] my rank:"<<my_rank<<" floor:"<<floor<<" ceiling: "<<ceiling<<" [start:"<<start<<" end:"<<end<<")\033[0m\n";

            new_out_.resize(end - start);
            std::vector<indexType> new_node_id(new_out_.size());
            beam_time.start();
            parlay::parallel_for(start, end, [&](size_t i) {
                parlay::internal::timer beam_temp_time;
                beam_temp_time.start();
                auto per_point_start_time = std::chrono::system_clock::now();
                size_t index = shuffled_inserts[i];
                new_node_id[i - start] = i - floor;
                //printf("parallel for i = %d, index = %d\n",i, index);
                QueryParams QP((long) 0, BP.L, (double) 0.0, (long) Points.size(), (long) G.max_degree());
                parlay::sequence<pid> visited = beam_search<Point, PointRange, indexType>(Points[index], G, Points, start_point, QP).first.second;
                
                BuildStats.increment_visited(index, visited.size());
                beam_temp_time.stop();

                new_out_[i-start] = robustPrune(index, visited, G, Points, alpha);
                auto per_point_end_time = std::chrono::system_clock::now();
                double per_point_all_time = std::chrono::duration<double>(per_point_end_time - per_point_start_time).count();
                beam_global.fetch_add(static_cast<int64_t>(per_point_all_time*1000));
            });
            beam_time.stop();
            std::cout<<"\033[31m[dbg] new_out size"<<new_out_.size()<<"\033[0m\n";
            
            //广播
            comm_time.start();
            //对齐，R个邻居+自身索引
            std::vector<char>outbox_buf_temp((BP.R + 1) * sizeof(indexType) * new_out_.size());
            parlay::parallel_for(0, new_out_.size(), [&](size_t i) {
                size_t actual_neighbors_count = new_out_[i].size();
                std::memcpy(outbox_buf_temp.data() + i * (BP.R + 1) * sizeof(indexType), new_out_[i].data(),
                    actual_neighbors_count * sizeof(indexType));
                //填充-1
                if(actual_neighbors_count < BP.R) {
                    std::memset(outbox_buf_temp.data() + i * (BP.R + 1) * sizeof(indexType) + actual_neighbors_count * sizeof(indexType), 0xff,
                        (BP.R - actual_neighbors_count) * sizeof(indexType));
                }
                // 将节点索引存储在最后一个位置
                std::memcpy(outbox_buf_temp.data() + i * (BP.R + 1) * sizeof(indexType) + BP.R * sizeof(indexType), &new_node_id[i], sizeof(indexType));
            });

            //压缩
            //压缩后大小+压缩前数据+压缩数据
            int dst_size = LZ4_compressBound(outbox_buf_temp.size());
            outbox_buf.resize(dst_size + 2 * sizeof(int));
            //压缩前数据大小
            int temp_box_size=outbox_buf_temp.size();
            std::memcpy(outbox_buf.data() + sizeof(int), &temp_box_size, sizeof(int));
            int compressed_size = LZ4_compress_default(outbox_buf_temp.data(), outbox_buf.data()+2*sizeof(int), outbox_buf_temp.size(), outbox_buf.size());
            outbox_buf.resize(compressed_size + 2 * sizeof(int));
            //压缩后数据大小
            std::memcpy(outbox_buf.data(), &compressed_size, sizeof(int));

            comm_time.stop();
        }
        else{//奇数次superstep， 执行更新外邻那部分
            comm_time.start();
            size_t recv_offset = 0;
            parlay::sequence<parlay::sequence<indexType>> recovered(ceiling - floor);

            while(recv_offset <inbox_buf_len){
            int compressed_size = *reinterpret_cast<const int*>(inbox_buf + recv_offset);
            recv_offset += sizeof(int);
            int src_size = *reinterpret_cast<const int*>(inbox_buf + recv_offset);
            recv_offset += sizeof(int);
            //解压缩
            std::vector<char> src_data(src_size);
            int decompressed_size = LZ4_decompress_safe(inbox_buf + recv_offset, src_data.data(), compressed_size, src_size);
            recv_offset += compressed_size;
            
            size_t recv_counts = decompressed_size / ((BP.R + 1) * sizeof(indexType));
            parlay::parallel_for(0, recv_counts, [&](size_t i) {
                const indexType* block_start = reinterpret_cast<const indexType*>(src_data.data() + i * (BP.R + 1) * sizeof(indexType));
                
                indexType node_index = *(block_start + BP.R);
                const indexType* end_pos = std::find(block_start, block_start + BP.R, -1);
                if (end_pos != block_start) {
                    recovered[node_index] = parlay::to_sequence(parlay::make_slice(block_start, end_pos));
                }
            });
            }
            comm_time.stop();

            
            flatten_time.start();


            flatten_time_t2.start();
            parlay::parallel_for(floor, ceiling, [&](size_t i) {
                //printf("recovered::%d",recovered[i-start].size());
                G[shuffled_inserts[i]].update_neighbors(recovered[i-floor]);
            });
            
            flatten_time_t2.stop();

            flatten_time_t1.start();
            auto to_flatten = parlay::tabulate(ceiling - floor, [&](size_t i) {
            auto per_point_start_time = std::chrono::system_clock::now();
            indexType index = shuffled_inserts[i + floor];
            //auto evens = parlay::filter(recovered[i], [&](indexType x) { return x % total_rank == my_rank;});
            std::vector<std::pair<indexType,indexType>> edges;
            for(auto& it: recovered[i]){
                if(it % total_rank == my_rank){
                        edges.push_back(std::make_pair(it,index));
                }
            }
                return edges;
           });
            flatten_time_t1.stop();
            flatten_time_t3.start();
            auto grouped_by = parlay::group_by_key(parlay::flatten(to_flatten));
            flatten_time_t3.stop();
            flatten_time.stop();
            std::cout<<"\033[31m[dbg]"<<"my_rank "<<my_id <<" grouped_by size: "<<grouped_by.size()<<"\033[0m\n";
 
            t_prune.start();
            parlay::sequence<unsigned int> new_neighbor_lens(grouped_by.size(), 0);

            parlay::parallel_for(0,grouped_by.size(), [&](size_t j) {              
            auto per_point_start_time = std::chrono::system_clock::now();

            auto &[index, candidates] = grouped_by[j];
            size_t newsize = candidates.size() + G[index].size();
            auto s1 = candidates.size();
            //std::cout<<"index: "<<index<<std::endl;
            if (newsize <= BP.R) {
                G[index].append_neighbors(candidates);
            } else {
                auto new_out_2_ = robustPrune(index, std::move(candidates), G, Points, alpha);  
                G[index].update_neighbors(new_out_2_);    
            }
            // std::cout<<"G["<<index<<"] size: "<<G[index].size();
            // for(auto &it: G[index]){
            //     std::cout<<it<<" ";
            // }
            // std::cout<<std::endl;
            new_neighbor_lens[j] = G[index].size();
            auto per_point_end_time = std::chrono::system_clock::now();
            double per_point_all_time = std::chrono::duration<double>(per_point_end_time - per_point_start_time).count();
            update_global.fetch_add(static_cast<int64_t>(per_point_all_time*1000));
             //广播G[index]
            });
            auto scan_result = parlay::scan(new_neighbor_lens);
            auto all_size = scan_result.second;
            auto offset = scan_result.first;
            t_prune.stop();

            //广播
            comm_time.start();

            std::vector<char>outbox_buf_temp((BP.R + 1) * sizeof(indexType) * grouped_by.size());
            parlay::parallel_for(0, grouped_by.size(), [&](size_t i) {
                size_t actual_neighbors_count = G[grouped_by[i].first].size();
                std::memcpy(outbox_buf_temp.data() + i * (BP.R + 1) * sizeof(indexType), G[grouped_by[i].first].begin(),
                    actual_neighbors_count * sizeof(indexType));
                //填充-1
                if(actual_neighbors_count < BP.R) {
                    std::memset(outbox_buf_temp.data() + i * (BP.R + 1) * sizeof(indexType) + actual_neighbors_count * sizeof(indexType), 0xff,
                        (BP.R - actual_neighbors_count) * sizeof(indexType));
                }
                // 将节点索引存储在最后一个位置
                std::memcpy(outbox_buf_temp.data() + i * (BP.R + 1) * sizeof(indexType) + BP.R * sizeof(indexType), &grouped_by[i].first, sizeof(indexType));
            });

            //原数据大小+压缩前数据+压缩数据
            int dst_size = LZ4_compressBound(outbox_buf_temp.size());
            outbox_buf.resize(dst_size + 2 * sizeof(int));
            //压缩前数据大小
            int temp_box_size=outbox_buf_temp.size();
            std::memcpy(outbox_buf.data() + sizeof(int), &temp_box_size, sizeof(int));
            int compressed_size = LZ4_compress_default(outbox_buf_temp.data(), outbox_buf.data()+2*sizeof(int), outbox_buf_temp.size(), outbox_buf.size());
            outbox_buf.resize(compressed_size + 2 * sizeof(int));
            //压缩后数据大小
            std::memcpy(outbox_buf.data(), &compressed_size, sizeof(int));


            comm_time.stop();
            inc += 1;

        }
        supersteps_time.start();
        return 0;
    }

    void cleanup(const int32_t superstep, const int32_t my_rank, const int32_t total_rank){
        this->local_end_time = MPI_Wtime();
       // out_log.close();
        all_time.stop();
        int64_t local_update_time = update_global.load();
        int64_t global_update_time = 0;
        MPI_Allreduce(&local_update_time, &global_update_time, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        int64_t local_beam_search_time = beam_global.load();
        int64_t global_beam_time = 0;
        MPI_Allreduce(&local_beam_search_time, &global_beam_time, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (my_rank == 0) {
            fmt::print(stdout, "beam search 总串行时间: {} ms, update neighbor 总串行时间: {} ms\n", global_beam_time, global_update_time);
        }
        fmt::print("Rank {}: local beam time: {} ms, local update neighbor time: {} ms\n", my_rank, local_beam_search_time, local_update_time);

        if(my_rank == 0){
            printf("\033[33m 总耗时：%lf, 通信时间: %lf, beam search时间: %lf, 更新邻居时间: %lf, 超级步迭代时间: %lf, 添加双向边时间：%lf\033[0m\n",
                all_time.total_time(),comm_time.total_time(),beam_time.total_time(),t_prune.total_time(),supersteps_time.total_time(), flatten_time.total_time());
            printf("\033[33m %lf, %lf, %lf\n\033[0m",flatten_time_t1.total_time(),flatten_time_t2.total_time(),flatten_time_t3.total_time());
        }
        if (my_rank == 0 && false) {
            fmt::print("正在对输出索引进行排序...");
            parlay::parallel_for (0, G.size(), [&] (long i) {
                auto less = [&] (indexType j, indexType k) {
		        return Points[i].distance(Points[j]) < Points[i].distance(Points[k]);};
                G[i].sort(less);});
            FILE *tmp_out = fopen("build/tmp_out_distributed.txt", "w");
            for(int i=0;i<G.size();i++){
                auto seq=G[i];
                for(int j=0;j<seq.size();j++){
                    fprintf(tmp_out,"%d ",seq[j]);
                }
                fprintf(tmp_out,"\n");
            }
            fclose(tmp_out);
        }
        if(dp.save && my_rank == 0) {
            
            G.save(dp.output_path.data());
        }
       // my_rank,this->local_end_time - this->local_start_time,ct.total_time(), pt.total_time());
        // fmt::print(fg(fmt::color::red),"[dbg] In cleanup, my rank :{} ,my id:{} test vector :{}\n",
        //             my_rank, my_id, global_vector[nb-1][d-1]);
        
    }


    parlay::sequence<indexType> robustPrune(indexType p, parlay::sequence<pid>& cand,
        GraphI &G, PR &Points, double alpha, bool add = true) {
    // add out neighbors of p to the candidate set.
   // printf("PRUNE:cand = %d\n",cand.size());
    size_t out_size = G[p].size();
    std::vector<pid> candidates;
    for (auto x : cand) candidates.push_back(x);
    
    if(add){
    for (size_t i=0; i<out_size; i++) {
    // candidates.push_back(std::make_pair(v[p]->out_nbh[i], Points[v[p]->out_nbh[i]].distance(Points[p])));
    candidates.push_back(std::make_pair(G[p][i], Points[G[p][i]].distance(Points[p])));
    }
    }
    
    // Sort the candidate set in reverse order according to distance from p.
    auto less = [&](pid a, pid b) { return a.second < b.second; };
    std::sort(candidates.begin(), candidates.end(), less);
    
    std::vector<indexType> new_nbhs;
    new_nbhs.reserve(BP.R);
    
    size_t candidate_idx = 0;
    
    while (new_nbhs.size() < BP.R && candidate_idx < candidates.size()) {
    // Don't need to do modifications.
        int p_star = candidates[candidate_idx].first;
        candidate_idx++;
        if (p_star == p || p_star == -1) {
            continue;
        }
    
        new_nbhs.push_back(p_star);
    
        for (size_t i = candidate_idx; i < candidates.size(); i++) {
            int p_prime = candidates[i].first;
            if (p_prime != -1) {
                distanceType dist_starprime = Points[p_star].distance(Points[p_prime]);
                distanceType dist_pprime = candidates[i].second;
                if (alpha * dist_starprime <= dist_pprime) {
                    candidates[i].first = -1;
                }
            }
        }
    }
    
    auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
   // printf("PRUNE:new_neighbors_size = %d\n",new_neighbors_seq.size());
    return new_neighbors_seq;
    }
    
    //wrapper to allow calling robustPrune on a sequence of candidates 
    //that do not come with precomputed distances
    parlay::sequence<indexType> robustPrune(indexType p, parlay::sequence<indexType> candidates,
        GraphI &G, PR &Points, double alpha, bool add = true){
    
    parlay::sequence<pid> cc;
    cc.reserve(candidates.size()); // + size_of(p->out_nbh));
    for (size_t i=0; i<candidates.size(); ++i) {
        cc.push_back(std::make_pair(candidates[i], Points[candidates[i]].distance(Points[p])));
    }
        return robustPrune(p, cc, G, Points, alpha, add);
    }
};




void bsp_build(BuildParams &BP, Graph<unsigned int>& G, PointRange& Points,stats<indexType> BuildStats, dataset_para &dp){

    insert p(BP ,G, Points,BuildStats,dp);
    MPI_Barrier(MPI_COMM_WORLD); 
    BSP::BroadcastBSPWorker worker(p);
    worker.run();
}};
