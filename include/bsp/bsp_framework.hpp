#include <cstdint>
#include <mpi.h>
#include <vector>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <mrmpi/mapreduce.h>
#include <mrmpi/keyvalue.h>
#include "byte_buffer.hpp"
#include <cassert>
#include <oneapi/tbb/concurrent_vector.h>

namespace BSP
{

    /** 迭代接收的消息收件箱 */
    struct MessageInbox
    {
        // messages队列存储了从上一轮超级步中接收到的所有消息（包含广播的消息与点对点发送的消息）
        tbb::concurrent_vector<bb::ByteBuffer> messages;
    };

    /**
     * 迭代状态信息
     */
    struct IterationStatus
    {
        // 是否终止迭代（true表示本轮超级步投票终止）
        bool vote_to_halt = false;
        // 本轮超级步需要向特定处理器发送的消息列表：<目的处理器编号, 消息内容>
        tbb::concurrent_vector<std::tuple<int32_t, bb::ByteBuffer>> messages;
        // 重置迭代状态信息（该函数由框架自动调用）
        void reset() {
            vote_to_halt = false;
            messages.clear();
        }
    };

    /**
     * BSP处理器的计算逻辑API接口（面向用户）
     * 
     * 该类提供了一系列基础API函数，需要继承该类并实现里面的所有虚函数。
     * 该类描述了每个处理器的本地内存（作为类的成员变量）以及处理器的计算逻辑。
     *
     * 使用说明： 使用者需要创建一个该类的子类，并实现该类中的所有虚函数，处理器的本地数据作为该类的成员变量进行存储。
     * 
     */
    class ProcessorInterface
    {
    public:
        virtual ~ProcessorInterface() = default;

        /* 整个计算开始前的准备函数
         * @param my_rank 当前处理器的编号（从0开始）
         * @param total_rank 总处理器数量
         */
        virtual void setup(const int32_t my_rank, const int32_t total_rank) = 0;

        /* 超级步迭代计算函数
        @param superstep 当前超级步轮数（从0开始）
        @param my_rank 当前处理器的编号（从0开始）
        @param total_rank 总处理器数量
        @param inbox 从上一轮超级步中接收到的消息列表（本轮超级步的输入数据）
        @param status 本轮超级步需要发送的消息列表以及迭代状态（本轮超级步需要输出的数据）
        @return 如果计算没有出现异常则返回0，否则返回非0值。如果返回非0值，则整个计算过程会被终止。

        超级步从0开始持续迭代，直到满足以下两个条件则计算结束：
        1. 所有处理器的vote_to_halt均为true；
        2. 所有处理器的messages均为空。

        当迭代终止后，会调用cleanup函数进行清理工作。
        */
        virtual int process(const int32_t superstep, const int32_t my_rank, const int32_t total_rank, const MessageInbox &inbox, IterationStatus &status) = 0;

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
    class BSPWorker
    {
    public:/* 成员变量（不要直接访问） */
        // 当前进程的处理器编号
        int32_t my_rank = -1;
        // 总处理器数量
        int32_t total_rank = -1;
        // 用户计算逻辑类对象
        ProcessorInterface &processor;
        // 当前超级步的轮数
        int32_t superstep = -1;
        // 本处理器的接收消息收件箱
        MessageInbox local_inbox;
        // 本处理器的发送消息状态对象
        IterationStatus status;
        // 工作模式
        enum WORK_MODE {MRMPI, InMem};
        WORK_MODE mode = MRMPI;
        // 是否输出调试信息
        bool debug = false;

    public: /* 面向用户的关键函数 */
        /** 构造函数
         * 注意：调用该函数前务必先调用MPI_Init函数进行MPI环境初始化。
        @param processor BSP处理器的计算逻辑对象（由用户创建并传入一个实现了具体计算逻辑的子类对象）
        */
        BSPWorker(ProcessorInterface &processor, WORK_MODE mode=InMem);
        /** 启动BSP计算，直至计算结束函数才返回。（由用户手动调用） */
        void run();

    private:
        /* 内部私有工具函数，用户请不要调用*/
        void mrmpi_run();
        void inmem_run();
        int exchange_messages_via_mpi();
    };
};