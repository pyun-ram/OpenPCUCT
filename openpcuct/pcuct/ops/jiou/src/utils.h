#ifndef JIOU_UTILS_H
#define JIOU_UTILS_H

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>
#include <Eigen/Dense>

#define PI 3.1415926
#define MAX_NUM_SP 30*30*30
#define MAX_NUM_SP_BEV 30*30
#define MAX_NUM_ZNORMED_BEV 10*10
#define BEV_DIM 2
#define MAX_NUM_SP_3D 30*30*30
#define MAX_NUM_ZNORMED_3D 10*10*10
#define D3_DIM 3

namespace py = pybind11;
using namespace std;

template <typename scalar_t>
 void argsort(scalar_t* arr, int* ids, const int N);

template <typename scalar_t>
 scalar_t matget(
    scalar_t* m,
    const int r,
    const int c,
    const int numr,
    const int numc,
    bool rtn_idx=false);

template <typename scalar_t>
 void gen_meshgrid_2d(
    scalar_t* sp,
    const int num_sp,
    const float grid_size,
    const float range);

template <typename scalar_t>
 void gen_meshgrid_3d(
    scalar_t* sp,
    const int num_sp,
    const float grid_size,
    const float range);

template <typename scalar_t>
 void add_mat_vec(
    scalar_t* m1,
    const int numr1,
    const int numc1,
    scalar_t* v,
    const int numrv,
    const int numcv,
    scalar_t* m,
    const int numr,
    const int numc);

template <typename scalar_t>
 void transpose(
    scalar_t* m,
    const int numr,
    const int numc,
    scalar_t* m_T);

template <typename scalar_t>
 void matmul(
    scalar_t* m1,
    const int numr1,
    const int numc1,
    scalar_t* m2,
    const int numr2,
    const int numc2,
    scalar_t* m);

template <typename scalar_t>
 scalar_t jaccard_discrete(
    scalar_t* px1,
    scalar_t* px2,
    const int num_sp);

template <typename scalar_t>
 scalar_t calc_multivariate_normal_pdf_bev(
    const int D,
    scalar_t* point,
    scalar_t* mean,
    scalar_t* cov);

float calc_multivariate_normal_pdf_3d(
    const int D,
    Eigen::Vector3f& point,
    Eigen::Vector3f& mean,
    float cov_det,
    Eigen::Matrix3f& icov);

#endif