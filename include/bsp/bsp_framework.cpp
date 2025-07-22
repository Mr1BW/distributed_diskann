#include "bsp_framework.hpp"
#include "byte_buffer.hpp"
#include <algorithm>
#include <climits>
#include <mpi.h>
#include <mrmpi/mapreduce.h>
#include <cassert>
#include <fmt/core.h>
#include <oneapi/tbb/parallel_sort.h>
#include <numeric>
#undef NDEBUG

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
    BSPWorker::BSPWorker(ProcessorInterface &processor, WORK_MODE mode) : processor(processor)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &total_rank);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        this->mode = mode;
        if (mode == MRMPI)
            fmt::print("[BSP] Processor {} inited with MRMPI mode.\n", my_rank);
        if (mode == InMem)
            fmt::print("[BSP] Processor {} inited with InMem mode.\n", my_rank);
    }

    /* 将key解释为消息目标的进程号（处理器号） */
    int rank_hash(char *key, int key_len)
    {
        assert(key_len == sizeof(int));
        int rank = *(int *)key;
        assert(key_len >= 0);
        return rank;
    }

    /* 将当前所有的KV对加入到每个处理器的inbox消息队列*/
    void collect_messages_scan(char *key, int keybytes, char *value, int valuebytes, void *ptr)
    {
        MessageInbox &inbox = *(MessageInbox *)ptr;
        bb::ByteBuffer buf((uint8_t *)value, valuebytes);
        inbox.messages.push_back(buf);
    }

    /* 将status中的所有message通过MapReduce框架的add函数发射出去 */
    void emit_messages_map(int itask, MAPREDUCE_NS::KeyValue *kv, void *ptr)
    {
        IterationStatus &status = *(IterationStatus *)ptr;
        for (auto &[target_rank, msg_buf] : status.messages)
        {
            kv->add((char *)&target_rank, sizeof(target_rank), (char *)msg_buf.getRawBuffer(), (int)msg_buf.size());
        }
    }

    void BSPWorker::run()
    {
        if (this->mode == MRMPI)
        {
            this->mrmpi_run();
        }
        if (this->mode == InMem)
        {
            this->inmem_run();
        }
    }

    void BSPWorker::mrmpi_run()
    {
        MAPREDUCE_NS::MapReduce *mr = new MAPREDUCE_NS::MapReduce(MPI_COMM_WORLD);
        if (my_rank == 0)
        {
            fmt::print("[BSP] Setup...\n", my_rank);
            fflush(stdout);
        }
        double total_step_time = 0, total_calc_time = 0, total_comm_time = 0;
        auto t0 = MPI_Wtime();
        processor.setup(my_rank, total_rank);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = MPI_Wtime();
        fmt::print("[BSP] Processor {} setup done! Elapsed time (s): {:.6f}\n", my_rank, t1 - t0);
        // 第一轮调用前准备
        mr->verbosity = 1;
        mr->open();
        int32_t superstep = 0;
        while (true)
        {
            auto t0 = MPI_Wtime();
            if (my_rank == 0)
            {
                fmt::print(stdout, "[BSP] Superstep {} started.\n", superstep);
                fflush(stdout);
            }
            // 1. 调用当前迭代轮次处理函数
            this->status.reset();
            int32_t ret = processor.process(superstep, my_rank, total_rank, this->local_inbox, this->status);
            auto t1 = MPI_Wtime();
            if (ret != 0)
            {
                fmt::print("[BSP] Error! Processor {} @ Superstep {} returns non-zero value {}.\n", my_rank, superstep, ret);
                MPI_Abort(MPI_COMM_WORLD, ret);
            }
            // 2. 检查迭代终止条件是否达成（达成条件：所有进程的 vote_to_halt 均为true并且没有任何消息发送
            int halt_flag = 0;
            int recv_flag = 0;
            if (this->status.vote_to_halt && this->status.messages.empty())
            {
                halt_flag = 1;
            }
            MPI_Allreduce(&halt_flag, &recv_flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (recv_flag == this->total_rank)
            {
                if (my_rank == 0)
                {
                    fmt::print("[BSP] Vote to halt at superstep {}.\n", superstep);
                    fflush(stdout);
                }
                break; // 终止迭代
            }
            // 3. 发送当前轮的点对点消息
            this->local_inbox.messages.clear();
            auto num_p2p_msg = mr->map(this->total_rank, emit_messages_map, &(this->status));
            if (num_p2p_msg > 0)
            {
                mr->aggregate(rank_hash);
            }
            mr->scan(collect_messages_scan, &(this->local_inbox));
            auto t3 = MPI_Wtime();
            auto calc_time = t1 - t0;
            auto comm_time = t3 - t1;
            auto step_time = t3 - t0;
            double times[] = {calc_time, -comm_time, step_time};
            double max_times[3] = {0};
            MPI_Reduce(times, max_times, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (my_rank == 0)
            {
                total_step_time += max_times[2];
                total_calc_time += max_times[0];
                total_comm_time += -max_times[1];
                fmt::print(stdout, "[BSP] Superstep {} done! Elapsed step/calc/comm time (s): {} {} {}\n", superstep, max_times[2], max_times[0], max_times[2] - max_times[0]);
                fflush(stdout);
            }
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
            fmt::print("[BSP] Cleanup...\n", my_rank);
            fflush(stdout);
        }
        auto t2 = MPI_Wtime();
        processor.cleanup(superstep, my_rank, total_rank);
        auto t3 = MPI_Wtime();
        fmt::print("[BSP] Processor {} cleanup done! Elapsed time (s): {:.6f}\n", my_rank, t3 - t2);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t4 = MPI_Wtime();
        if (my_rank == 0)
        {
            fmt::print("[BSP] Total elapsed time (s): {:.6f}\n", t4 - t0);
            fmt::print("[BSP] Total step/calc/comm time (s): {:.6f} {:.6f} {:.6f}\n", total_step_time, total_calc_time, total_comm_time);
        }
        delete mr;
        fmt::print("[BSP] Processor {} finalized.\n", my_rank);
        fflush(stdout);
    }

    void BSPWorker::inmem_run()
    {
        if (my_rank == 0)
        {
            fmt::print("[BSP] Setup...\n", my_rank);
            fflush(stdout);
        }
        double total_step_time = 0, total_calc_time = 0, total_comm_time = 0;
        auto t0 = MPI_Wtime();
        processor.setup(my_rank, total_rank);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = MPI_Wtime();
        fmt::print("[BSP] Processor {} setup done! Elapsed time (s): {:.6f}\n", my_rank, t1 - t0);
        // 第一轮调用前准备
        int32_t superstep = 0;
        while (true)
        {
            auto t0 = MPI_Wtime();
            if (my_rank == 0)
            {
                fmt::print(stdout, "[BSP] Superstep {} started.\n", superstep);
                fflush(stdout);
            }
            // 1. 调用当前迭代轮次处理函数
            this->status.reset();
            //std::cout<<"[BSP] test0"<<std::endl;
            int32_t ret = processor.process(superstep, my_rank, total_rank, this->local_inbox, this->status);
            auto t1 = MPI_Wtime();
            if (ret != 0)
            {
                fmt::print("[BSP] Error! Processor {} @ Superstep {} returns non-zero value {}.\n", my_rank, superstep, ret);
                MPI_Abort(MPI_COMM_WORLD, ret);
            }
            //std::cout<<"[BSP] test1"<<std::endl;
            // 3. 发送当前轮的点对点消息
            this->local_inbox.messages.clear();
            int halt_flag = this->exchange_messages_via_mpi();
            if (halt_flag != 0 && my_rank == 0)
            {
                fmt::print("[BSP] Vote to halt at superstep {}.\n", superstep);
                fflush(stdout);
            }
            if (halt_flag != 0)
            {
                break; // 终止迭代
            }
            auto t2 = MPI_Wtime();
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
                fmt::print(stdout, "[BSP] Superstep {} done! Elapsed step/calc/comm time (s): {:.6f} {:.6f} {:.6f}\n", superstep, max_times[2], max_times[0], max_times[2] - max_times[0]);
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
            fmt::print("[BSP] Cleanup...\n", my_rank);
            fflush(stdout);
        }
        auto t2 = MPI_Wtime();
        processor.cleanup(superstep, my_rank, total_rank);
        auto t3 = MPI_Wtime();
        fmt::print("[BSP] Processor {} cleanup done! Elapsed time (s): {:.6f}\n", my_rank, t3 - t2);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t4 = MPI_Wtime();
        if (my_rank == 0)
        {
            fmt::print("[BSP] Total elapsed time (s): {:.6f}\n", t4 - t0);
            fmt::print("[BSP] Total step/calc/comm time (s): {:.6f} {:.6f} {:.6f}\n", total_step_time, total_calc_time, total_comm_time);
            fmt::print("[BSP] Average step/calc/comm time per superstep (s): {:.6f} {:.6f} {:.6f}\n", total_step_time / (superstep + 1), total_calc_time / (superstep + 1), total_comm_time / (superstep+1));
        }
        fmt::print("[BSP] Processor {} finalized.\n", my_rank);
        fflush(stdout);
    }

    /**
     * 通过MPI_alltoallv函数在处理器之间交换消息（全内存通信效率高，但内存开销大）
     */
    int BSP::BSPWorker::exchange_messages_via_mpi()
    {
        //std::cout<<"[BSP] ready_to_sort"<<std::endl;
        // 将待发送消息按目标处理器编号升序排序
        //std::sort(status.messages.begin(), status.messages.end(), [](const std::tuple<int32_t, bb::ByteBuffer> &a, const std::tuple<int32_t, bb::ByteBuffer> &b)
        //          { return std::get<0>(a) < std::get<0>(b); });
        auto t0 = MPI_Wtime();
        tbb::parallel_sort(status.messages, [](const std::tuple<int32_t, bb::ByteBuffer> &a, const std::tuple<int32_t, bb::ByteBuffer> &b)
                                            { return std::get<0>(a) < std::get<0>(b);});
        auto t1 = MPI_Wtime();
        //std::cout<<"[BSP] sort_done"<<std::endl;
        // 准备MPI Alltoallv通信所需的缓冲区及参数
        bb::ByteBuffer send_buf;
        std::vector<int> send_counts(total_rank, 0);
        std::vector<int> send_displs(total_rank, 0);
        uint64_t msg_offset = 0;
        for (int32_t i = 0; i < total_rank; i++)
        {
            // 定位发送给处理器i的第一条消息下标msg_offset
            while (msg_offset < status.messages.size() && std::get<0>(status.messages.at(msg_offset)) < i)
            {
                msg_offset++;
            }
            // 如果没有发送给处理器i的消息
            if (msg_offset >= status.messages.size() || std::get<0>(status.messages.at(msg_offset)) > i)
            {
                send_counts.at(i) = 0;
                send_displs.at(i) = 0;
                continue;
            }
            // 计算发送给处理器i的消息缓冲区拼接并计算发送字节数量
            int send_count = 0;
            if (send_buf.size() > INT_MAX) {
                fmt::print(stderr, "ERROR! The bytes to send exceeds the maximal of INT, which is not supported by MPI\n");
                exit(11);
            }
            send_displs.at(i) = send_buf.size(); // 存储当前发送位置起始下标
            while (msg_offset < status.messages.size() && std::get<0>(status.messages.at(msg_offset)) == i)
            {
                auto &[target_rank, msg] = status.messages.at(msg_offset);
                send_buf.putLong((int64_t)msg.size());
                send_buf.putBytes(msg.getRawBuffer(), msg.size());
                send_count += sizeof(int64_t) + msg.size();
                msg_offset++;
            }
            send_counts.at(i) = send_count;
        }
        auto t2 = MPI_Wtime();
        // 准备接收缓冲区参数
        std::vector<int> recv_counts(total_rank);
        std::vector<int> recv_displs(total_rank);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
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
        bb::ByteBuffer recv_buf(total_recv_size);
        recv_buf.resize(total_recv_size);
        auto t3 = MPI_Wtime();
        // fmt::print("[DEBUG] total_recv_size size: {}\n", total_recv_size);
        // fmt::print("[DEBUG] recv_buf size: {}\n", recv_buf.size());

        // 执行MPI_Alltoallv通信
        MPI_Alltoallv(
            send_buf.getRawBuffer(), send_counts.data(), send_displs.data(), MPI_BYTE,
            recv_buf.putBufPtr(), recv_counts.data(), recv_displs.data(), MPI_BYTE,
            MPI_COMM_WORLD);

        auto t4 = MPI_Wtime();

        // 解析接收到的消息到local_inbox
        uint64_t pos = 0;

        while (pos < recv_buf.size())
        {
            uint64_t msg_size = recv_buf.getLong(pos);
            pos += sizeof(int64_t);
            bb::ByteBuffer msg(recv_buf.getRawBuffer() + pos, msg_size);
            local_inbox.messages.push_back(msg);
            pos += msg_size;
        }
        auto t5  = MPI_Wtime();
        // 检查迭代终止条件是否达成（达成条件：所有进程的 vote_to_halt 均为true并且消息发送量为0
        uint64_t send_halt_flags[2] = {(uint64_t)this->status.vote_to_halt, total_recv_size};
        uint64_t recv_halt_flags[2] = {0, 0};
        MPI_Allreduce(&send_halt_flags, &recv_halt_flags, 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        auto t6 = MPI_Wtime();
        fmt::print(stderr, "[debug]Rank {} Comm detail: s1 {}, s2 {}, s3 {}, s4 {}, s5 {}, s6 {}\n", my_rank, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t1);
        if (my_rank == 0)
        {
            fmt::print("[BSP] Halt votes {}/{}, total communication cost (byte): {}\n", recv_halt_flags[0], total_rank, recv_halt_flags[1]);
        }
        if (recv_halt_flags[0] == (uint64_t)total_rank && recv_halt_flags[1] == 0)
        {
            return 1;
        }
        return 0;
    }
}