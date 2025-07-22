#undef NDEBUG
#include "broadcast_bsp_framework.hpp"
#include <algorithm>
#include <climits>
#include <mpi.h>
#include <mrmpi/mapreduce.h>
#include <cassert>
#include <fmt/core.h>
#include <oneapi/tbb/parallel_sort.h>
#include <numeric>

static void compute_stats(const std::vector<double>& data, double& min_val, double& q1, double& median, double& q3, double& max_val, double& mean_val) {
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    min_val = sorted_data.front();
    max_val = sorted_data.back();
    mean_val = std::accumulate(sorted_data.begin(), sorted_data.end(), 0.0) / sorted_data.size();

    auto median_idx = sorted_data.size() / 2;
    median = sorted_data[median_idx];

    auto q1_idx = sorted_data.size() / 4;
    auto q3_idx = (3 * sorted_data.size()) / 4;

    q1 = sorted_data[q1_idx];
    q3 = sorted_data[q3_idx];
}

namespace BSP
{

    /* 构造函数 */
    BroadcastBSPWorker::BroadcastBSPWorker(BroadcastProcessorInterface &processor) : processor(processor)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        std::cout<<"inbox buf max: "<<inbox_buf.max_size()<<std::endl;
        std::cout<<"outbox buf max: "<<outbox_buf.max_size()<<std::endl;
        inbox_buf.reserve((uint64_t)INT_MAX);
        outbox_buf.reserve((uint64_t)INT_MAX);
    }

    void BroadcastBSPWorker::run()
    {
        this->inmem_run();
    }

     void BroadcastBSPWorker::inmem_run()
    {
        if (my_rank == 0)
        {
            fmt::print("[BroadcastBSP] Setup...\n", my_rank);
            fflush(stdout);
        }
        double total_step_time = 0, total_calc_time = 0, total_comm_time = 0;
        auto t0 = MPI_Wtime();
        processor.setup(my_rank, total_rank);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = MPI_Wtime();
        fmt::print("[BroadcastBSP] Processor {} setup done! Elapsed time (s): {:.6f}\n", my_rank, t1 - t0);
        // 第一轮调用前准备
        int32_t superstep = 0;
        while (true)
        {
            auto t0 = MPI_Wtime();
            if (my_rank == 0)
            {
                fmt::print(stdout, "[BroadcastBSP] Superstep {} started.\n", superstep);
                fflush(stdout);
            }
            // 1. 调用当前迭代轮次处理函数
            this->status.reset();
            //this->outbox_buf.clear();
            int32_t ret = processor.process(superstep, my_rank, total_rank, this->inbox_buf.data(), this->inbox_len, this->outbox_buf, this->outbox_len, this->status);
            auto t1 = MPI_Wtime();
            if (ret != 0)
            {
                fmt::print("[BroadcastBSP] Error! Processor {} @ Superstep {} returns non-zero value {}.\n", my_rank, superstep, ret);
                MPI_Abort(MPI_COMM_WORLD, ret);
            }
            //std::cout<<"[BSP] test1"<<std::endl;
            // 3. 广播消息
            int halt_flag = this->exchange_messages_via_mpi();
            auto t2 = MPI_Wtime();
            if (halt_flag != 0 && my_rank == 0)
            {
                fmt::print("[BroadcastBSP] Vote to halt at superstep {}.\n", superstep);
                fflush(stdout);
            }
            if (halt_flag != 0)
            {
                break; // 终止迭代
            }
            auto calc_time = t1 - t0;
            auto comm_time = t2 - t1;
            auto step_time = t2 - t0;
            double times[] = {calc_time, -comm_time, step_time};
            double max_times[3] = {0};
            MPI_Reduce(times, max_times, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (my_rank == 0)
            {
                total_step_time += max_times[2];
                total_calc_time += max_times[0];
                total_comm_time += max_times[2] - max_times[0];
                fmt::print(stdout, "[BroadcastBSP] Superstep {} done! Elapsed step/calc/comm time (s): {:.6f} {:.6f} {:.6f}\n", superstep, max_times[2], max_times[0], max_times[2] - max_times[0]);
                fflush(stdout);
            }
            //std::cout<<"[BSP] test2"<<std::endl;
            if (this->debug)
            {
                // 进一步输出执行时间分位数调试信息
                double local_times[3] = {calc_time, comm_time, step_time};
                std::vector<double> all_calc_times(total_rank);
                std::vector<double> all_comm_times(total_rank);
                std::vector<double> all_step_times(total_rank);
                MPI_Gather(&local_times[0], 1, MPI_DOUBLE, all_calc_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gather(&local_times[1], 1, MPI_DOUBLE, all_comm_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gather(&local_times[2], 1, MPI_DOUBLE, all_step_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if (my_rank == 0) {
                    double calc_min, calc_q1, calc_median, calc_q3, calc_max, calc_mean;
                    compute_stats(all_calc_times, calc_min, calc_q1, calc_median, calc_q3, calc_max, calc_mean);

                    double comm_min, comm_q1, comm_median, comm_q3, comm_max, comm_mean;
                    compute_stats(all_comm_times, comm_min, comm_q1, comm_median, comm_q3, comm_max, comm_mean);

                    double step_min, step_q1, step_median, step_q3, step_max, step_mean;
                    compute_stats(all_step_times, step_min, step_q1, step_median, step_q3, step_max, step_mean);

                    fmt::print("[Superstep {}] Calculation Time: Min={:.6f}, Q1={:.6f}, Median={:.6f}, Q3={:.6f}, Max={:.6f}, Mean={:.6f}\n", superstep,
                            calc_min, calc_q1, calc_median, calc_q3, calc_max, calc_mean);
                    fmt::print("[Suprestep {}] Communication Time: Min={:.6f}, Q1={:.6f}, Median={:.6f}, Q3={:.6f}, Max={:.6f}, Mean={:.6f}\n", superstep,
                            comm_min, comm_q1, comm_median, comm_q3, comm_max, comm_mean);
                    fmt::print("[Suprestep {}] Step Time: Min={:.6f}, Q1={:.6f}, Median={:.6f}, Q3={:.6f}, Max={:.6f}, Mean={:.6f}\n",superstep,
                            step_min, step_q1, step_median, step_q3, step_max, step_mean);
                }           
            }
            // 4. 进入下一轮迭代
            superstep++;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0)
        {
            fmt::print("[BroadcastBSP] Cleanup...\n", my_rank);
            fflush(stdout);
        }
        auto t2 = MPI_Wtime();
        processor.cleanup(superstep, my_rank, total_rank);
        auto t3 = MPI_Wtime();
        fmt::print("[BroadcastBSP] Processor {} cleanup done! Elapsed time (s): {:.6f}\n", my_rank, t3 - t2);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t4 = MPI_Wtime();
        if (my_rank == 0)
        {
            fmt::print("[BroadcastBSP] Total elapsed time (s): {:.6f}\n", t4 - t0);
            fmt::print("[BroadcastBSP] Total step/calc/comm time (s): {:.6f} {:.6f} {:.6f}\n", total_step_time, total_calc_time, total_comm_time);
            fmt::print("[BroadcastBSP] Average step/calc/comm time per superstep (s): {:.6f} {:.6f} {:.6f}\n", total_step_time / (superstep + 1), total_calc_time / (superstep + 1), total_comm_time / (superstep+1));
        }
        fmt::print("[BroadcastBSP] Processor {} finalized.\n", my_rank);
        fflush(stdout);
    }

    /**
     * 通过MPI_alltoallv函数在处理器之间交换消息（全内存通信效率高，但内存开销大）
     */
    int BSP::BroadcastBSPWorker::exchange_messages_via_mpi()
    {
        auto t0 = MPI_Wtime();
        // 准备发送缓冲区参数
        assert(this->outbox_len < (uint64_t)INT_MAX); // 发送消息的总大小不能超过INT_MAX
        int send_count = (int)this->outbox_len;
        std::vector<int> send_counts(total_rank, send_count);
        std::vector<int> send_displs(total_rank, 0);
        // 准备接收缓冲区参数
        std::vector<int> recv_counts(total_rank);
        std::vector<int> recv_displs(total_rank);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        auto t1 = MPI_Wtime();
        // 计算接收缓冲区的总大小和位移
        uint64_t total_recv_size = 0;
        for (int i = 0; i < total_rank; i++)
        {
            recv_displs.at(i) = (int)total_recv_size;
            if (total_recv_size > INT_MAX) 
            {
                fmt::print(stderr, "ERROR! The bytes to recv exceeds the maximal of INT, which is not supported by MPI\n");
                exit(11);
            }
            total_recv_size += recv_counts.at(i);
        }
        // 准备接收缓冲区
        //this->inbox_buf.clear();
        this->inbox_buf.reserve(total_recv_size);
        this->inbox_len = total_recv_size;
        auto t2 = MPI_Wtime();
        // 执行MPI_Alltoallv通信
        MPI_Alltoallv(
            this->outbox_buf.data(), send_counts.data(), send_displs.data(), MPI_BYTE,
            this->inbox_buf.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE,
            MPI_COMM_WORLD);
        auto t3 = MPI_Wtime();
        // 检查迭代终止条件是否达成（达成条件：所有进程的 vote_to_halt 均为true并且消息发送量为0
        uint64_t send_halt_flags[2] = {(uint64_t)this->status.vote_to_halt, total_recv_size};
        uint64_t recv_halt_flags[2] = {0, 0};
        MPI_Allreduce(&send_halt_flags, &recv_halt_flags, 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        auto t4 = MPI_Wtime();
        fmt::print(stderr, "[debug] Rank {} Comm detail: total time {} s1 {} s2 {} s3 {} s4 {}\n", my_rank, t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3);
        if (my_rank == 0)
        {
            fmt::print("[BSP] Halt votes {}/{}, total communication cost (byte): {}\n", recv_halt_flags[0], total_rank, recv_halt_flags[1]);
        }
        std::cout<<"[BSP] test3: "<<recv_halt_flags[0]<<" "<<recv_halt_flags[1]<<std::endl;
        if (recv_halt_flags[0] == (uint64_t)total_rank )//&& recv_halt_flags[1] == 0)
        {
            return 1;
        }
        return 0;
    }
}
