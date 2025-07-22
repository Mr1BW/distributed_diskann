#include <cstdint>
#include <mpi.h>
#include <vector>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <mrmpi/mapreduce.h>
#include <mrmpi/keyvalue.h>
#include "bsp_framework.hpp"
#include <cassert>
#include <oneapi/tbb/concurrent_vector.h>

namespace BSP
{

    /**
     * BSP处理器的计算逻辑API接口（面向用户，广播型通信模式）
     * 
     * 该类提供了一系列基础API函数，需要继承该类并实现里面的所有虚函数。
     * 该类描述了每个处理器的本地内存（作为类的成员变量）以及处理器的计算逻辑。
     *
     * 使用说明： 使用者需要创建一个该类的子类，并实现该类中的所有虚函数，处理器的本地数据作为该类的成员变量进行存储。
     * 
     */
    class BroadcastProcessorInterface
    {
    public:
        virtual ~BroadcastProcessorInterface() = default;

        /* 整个计算开始前的准备函数
         * @param my_rank 当前处理器的编号（从0开始）
         * @param total_rank 总处理器数量
         */
        virtual void setup(const int32_t my_rank, const int32_t total_rank) = 0;

        /* 超级步迭代计算函数
        @param superstep 当前超级步轮数（从0开始）
        @param my_rank 当前处理器的编号（从0开始）
        @param total_rank 总处理器数量
        @param inbox_buf 从上一轮超级步中接收到的消息缓冲区
        @param inbox_buf_len 接收消息缓冲区的长度
        @param outbox_buf 本轮超级步需要发送的消息缓冲区
        @param outbox_len 本轮超级步需要发送的消息缓冲区长度
        @param status 本轮超级步需要发送的消息列表以及迭代状态（本轮超级步需要输出的数据）
        @return 如果计算没有出现异常则返回0，否则返回非0值。如果返回非0值，则整个计算过程会被终止。

        超级步从0开始持续迭代，直到满足以下两个条件则计算结束：
        1. 所有处理器的vote_to_halt均为true；
        2. 所有处理器的messages均为空。

        当迭代终止后，会调用cleanup函数进行清理工作。
        */
        virtual int process(const int32_t superstep, const int32_t my_rank, const int32_t total_rank, const char *inbox_buf, const uint64_t inbox_buf_len, 
            std::vector<char> &outbox_buf, uint64_t &outbox_len, IterationStatus &status) = 0;

        /* 整个计算结束后的清理函数
        @param 迭代结束时的超级步轮数（从0开始）
        @param my_rank 当前进程的编号（从0开始）
        @param total_rank 总进程数
        */
        virtual void cleanup(const int32_t superstep, const int32_t my_rank, const int32_t total_rank) = 0;
    };

    /**
     *
     * BSP工作进程驱动
     *
     */
    class BroadcastBSPWorker
    {
    public:/* 成员变量（不要直接访问） */
        // 当前进程的处理器编号
        int32_t my_rank = -1;
        // 总处理器数量
        int32_t total_rank = -1;
        // 用户计算逻辑类对象
        BroadcastProcessorInterface &processor;
        // 当前超级步的轮数
        int32_t superstep = -1;
        // 接收消息缓冲区
        std::vector<char> inbox_buf;
        uint64_t inbox_len = 0;
        // 发送消息缓冲区
        std::vector<char> outbox_buf;
        uint64_t outbox_len = 0;
        // 本处理器的发送消息状态对象
        IterationStatus status;
        // 是否输出调试信息
        bool debug = false;

    public: /* 面向用户的关键函数 */
        /** 构造函数
         * 注意：调用该函数前务必先调用MPI_Init函数进行MPI环境初始化。
        @param processor BSP处理器的计算逻辑对象（由用户创建并传入一个实现了具体计算逻辑的子类对象）
        */
        BroadcastBSPWorker(BroadcastProcessorInterface &processor);
        /** 启动BSP计算，直至计算结束函数才返回。（由用户手动调用） */
        void run();

    private:
        /* 内部私有工具函数，用户请不要调用*/
        void inmem_run();
        int exchange_messages_via_mpi();
    };
};
