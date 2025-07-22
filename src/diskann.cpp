#include "utils/types.h"
#include "utils/point_range.h"
#include "utils/graph.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/stats.h"
#include <chrono>
#undef roundup

#include "index.h"
#include "dataset.hpp"
#include <iostream>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include "bsp_index.hpp"

//#define ORIGINAL_PARLAY

template <typename T, typename Point>
void build_index(dataset_para dp,int my_rank){
    //设置参数

    BuildParams BP(dp.R, dp.L, dp.alpha, dp.two_pass);

    //读取文件
    PointRange<T, Point> Points = PointRange<T, Point>((dp.ds_dir).data());
    //提前开辟空间
    //Graph<unsigned int> G("./results/random-xs");
    double start =MPI_Wtime();
    Graph<unsigned int> G = Graph<unsigned int>(dp.R, Points.size());
    stats<unsigned int> BuildStats(G.size());
    int j = 0;

    //std::cout<<"1.parlay 2. diskann"<<std::endl<<">>";
    //std::cin>>j;
    using index = BSP_index<Point, PointRange<T, Point>, unsigned int>;
    index I;
    I.bsp_build(BP,G,Points,BuildStats,dp);
}

template <typename T, typename Point>
void build_index_original_parylay(dataset_para dp){
    //设置参数

    BuildParams BP(dp.R, dp.L, dp.alpha, dp.two_pass);

    //读取文件
    PointRange<T, Point> Points = PointRange<T, Point>((dp.ds_dir).data());
    //提前开辟空间
    //Graph<unsigned int> G("./results/random-xs");
    auto start = std::chrono::high_resolution_clock::now();
    Graph<unsigned int> G = Graph<unsigned int>(dp.R, Points.size());
    stats<unsigned int> BuildStats(G.size());
    int j = 0;

    //std::cout<<"1.parlay 2. diskann"<<std::endl<<">>";
    //std::cin>>j;
    using index = knn_index<Point, PointRange<T, Point>, unsigned int>;
    index I(BP);
    I.build_index(G, Points, BuildStats);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    fmt::print(stdout, "图索引构建用时: {} ms.\n", duration.count());
    /*
    for(auto i:G[0]){
       std::cout<<i<<" ";
    }
       */
    G.save(dp.output_path.data());
    if (true) {
      FILE *tmp_out = fopen("build/tmp_out_parlay.txt", "w");
      for (int i = 0; i < G.size(); i++) {
        auto seq = G[i];
        for (int j = 0; j < seq.size(); j++) {
          fprintf(tmp_out, "%d ", seq[j]);
        }
        fprintf(tmp_out, "\n");
      }
      fclose(tmp_out);
    }
}


int main(int argc, char *argv[]){

    #ifdef ORIGINAL_PARLAY
    dataset_para data_para;
    std::string filename="para.json";
    // 如果用户通过命令行指定参数配置文件
    if (argc >= 2) {
        filename = argv[1];
    }
    fmt::print("Using parameter file: {}\n", filename);
    data_para.set(filename);
    data_para.open(data_para.dataset_parafile);
    data_para.print();
    if(data_para.dtype == "float32")build_index_original_parylay<float,Euclidian_Point<float>>(data_para);
    if(data_para.dtype == "int8")build_index_original_parylay<int8_t,Euclidian_Point<int8_t>>(data_para);
    if(data_para.dtype == "uint8")build_index_original_parylay<uint8_t,Euclidian_Point<uint8_t>>(data_para);
    #else
    MPI_Init(&argc, &argv);
    int my_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    dataset_para data_para;
    std::string filename="para.json";
    // 如果用户通过命令行指定参数配置文件
    if (argc >= 2) {
        filename = argv[1];
    }
    fmt::print("Using parameter file: {}\n", filename);
    data_para.set(filename);
    data_para.open(data_para.dataset_parafile);

    if(my_rank==0){
        data_para.print();
    }
    if(data_para.dtype == "float32")build_index<float,Euclidian_Point<float>>(data_para,my_rank);
    if(data_para.dtype == "int8")build_index<int8_t,Euclidian_Point<int8_t>>(data_para,my_rank);
    if(data_para.dtype == "uint8")build_index<uint8_t,Euclidian_Point<uint8_t>>(data_para,my_rank);
    
    MPI_Finalize();
    #endif
    return 0;
}
