/*
该文件下存放关于数据集描述的数据结构
dataset_para即是存放json中的内容
dataset_vector 存放的是数据集中的向量，目前写了存放数据集的向量。后续对于查询及与真实数据的存放尝试在此写

数据集文件说明
random系列：float32
data存放数据集
    前4字节存放向量数量，5-8字节存放维度，后续存放数据
queries存放查询集
    前4字节存放查询集数量，5-8字节存放维度，后续存放待查询的向量
    注：经验证，查询集和数据集基本无交集
gt存放结果
    前4字节存放查询集数量，5-8字节存放邻居的数量。
    后续分成两部分：
        第一部分存放邻居的索引 每个索引占4字节
        第二部分存放点到邻居的真实距离 距离float32

*/

#include <iostream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <fmt/core.h>

#pragma once

class dataset_para{//数据集参数集合
    public:
    std::string dataset_parafile;//数据集参数文件
    std::string dataset_name;//数据集名称
    int nb;//数据集中向量数量
    int nq;//查询集中向量数量
    int d;//向量维度
    std::string dtype;//向量元素的数据类型
    std::string ds_fn;//数据集的文件名称
    std::string qs_fn;//查询集的文件名称
    std::string gt_fn;//真实位置集合的文件名称
    std::string basedir;//文件所在路径
    
    //文件所在具体位置（相对）
    std::string ds_dir;
    std::string qs_dir;
    std::string gt_dir;
    std::string output_path;

    //构建过程中的参数
    bool build=true;
    bool query=true;
    bool save=true;
    bool two_pass = false;
    uint64_t start_point_index = 0;//起始点索引
    double xita = 0.02;
    double alpha = 1.2;
    uint32_t L = 128;
    uint32_t R = L / 2;//prune中的R
    //查询的参数
    uint32_t Ls = 128;
    uint32_t num_neighbors_query = 100;
    uint64_t max_build = 0;

    void set(std::string filename)
    {

        std::ifstream inputfile(filename);
        if(inputfile.fail()){
            std::cerr<<filename<<"cannot open"<<std::endl;
            return;
          //  exit(0);
        }
        nlohmann::json para_json=nlohmann::json::parse(inputfile);
        build=para_json["build"];
        query=para_json["query"];
        save=para_json["save"];
        two_pass=para_json["two_pass"];

        start_point_index=para_json["start_point_index"];
        xita=para_json["xita"];
        alpha=para_json["alpha"];
        L=para_json["L"];
        R=para_json["R"];
        Ls=para_json["Ls"];
        max_build=para_json["max"];
        num_neighbors_query=para_json["num_neighbors_query"];
        dataset_parafile=para_json["data_para"];
        inputfile.close();
    }


    void open(std::string filename)
    {

        std::ifstream inputfile(filename);
        if(inputfile.fail()){
            std::cerr<<filename<<"cannot open"<<std::endl;
            exit(0);
        }
        

        nlohmann::json para_json=nlohmann::json::parse(inputfile);
        dataset_name=para_json["dataset"];
        nb=para_json["nb"];
        nq=para_json["nq"];
        d=para_json["d"];
        dtype=para_json["dtype"];
        ds_fn=para_json["ds_fn"];
        qs_fn=para_json["qs_fn"];
        gt_fn=para_json["gt_fn"];
        basedir=para_json["basedir"];
    //std::string parameter_content((std::istreambuf_iterator<char>(inputfile)), 
    //                 std::istreambuf_iterator<char>());
    // std::cout<<parameter_content<<std::endl;
    // const char* buf=parameter_content.c_str();
    //    std::cout<<dataset_name<<"   "<<nb
    //            <<"  "<<nq<<"  "<<d<<"  "
    //            <<dtype<<" "<<ds_fn<<" "<<qs_fn<<" "<<gt_fn<<" "<<basedir;
        ds_dir=basedir + "/" + ds_fn;
        qs_dir=basedir + "/" + qs_fn;
        gt_dir=basedir + "/" + gt_fn;
        output_path="results/" + dataset_name;
    //    std::cout<<ds_dir<<" "<<qs_dir<<" "<<gt_dir;
        inputfile.close();
    }

    void print(){
        fmt::print("Dataset name: {}\n",dataset_name);
        fmt::print("The number of vector in dataset: {}\n",nb);
        fmt::print("The number of vector in query set {}\n",nq);
        fmt::print("Dimension: {}\n",d);
        fmt::print("The elements type: {}\n",dtype);
        fmt::print("The dir:\ndataset: {}\nquery set: {}\nground truth: {}\n",ds_dir,qs_dir,gt_dir);
    }
};