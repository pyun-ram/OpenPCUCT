/*
 * File Created: Thu Mar 25 2022

*/
# include "utils.h"

template <typename scalar_t>
 void argsort(scalar_t* arr, int* ids, const int N)
{
    vector<pair<scalar_t, int>> vp;
    for (int i=0; i<N; i++)
        vp.push_back(make_pair(arr[i], i));
    sort(vp.begin(), vp.end());
    for (int i=0; i<N; i++)
    {
        arr[i] = vp[i].first;
        ids[i] = vp[i].second;
    }
}

template <typename scalar_t>
 scalar_t matget(
    scalar_t* m,
    const int r,
    const int c,
    const int numr,
    const int numc,
    bool rtn_idx)
{
    if (rtn_idx)
        return (scalar_t)(r*numc + c);
    return m[r*numc + c];
}

template <typename scalar_t>
 void gen_meshgrid_2d(
    scalar_t* sp,
    const int num_sp,
    const float grid_size,
    const float range)
{
    const int dim_len = range/grid_size;
    scalar_t znormy = -range/2.0 + grid_size/2.0;
    for (int i=0; i<dim_len; i++)
    {
        scalar_t znormx = -range/2.0 + grid_size/2.0;
        for (int j=0; j<dim_len; j++)
        {
            sp[i*dim_len*2+j*2+0] = znormx;
            sp[i*dim_len*2+j*2+1] = znormy;
            znormx += grid_size;
        }
        znormy += grid_size;
    }
}

template <typename scalar_t>
 void gen_meshgrid_3d(
    scalar_t* sp,
    const int num_sp,
    const float grid_size,
    const float range)
{
    const int dim_len = range/grid_size;
    scalar_t znormy = -range/2.0 + grid_size/2.0;
    for (int i=0; i<dim_len; i++)
    {
        scalar_t znormx = -range/2.0 + grid_size/2.0;
        for (int j=0; j<dim_len; j++)
        {
            scalar_t znormz =  -range/2.0 + grid_size/2.0;
            for (int k=0; k<dim_len; k++)
            {
                sp[i*dim_len*dim_len*3+j*dim_len*3+k*3+0] = znormx;
                sp[i*dim_len*dim_len*3+j*dim_len*3+k*3+1] = znormy;
                sp[i*dim_len*dim_len*3+j*dim_len*3+k*3+2] = znormz;
                znormz += grid_size;
            }
            znormx += grid_size;
        }
        znormy += grid_size;
    }
}

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
    const int numc)
{
    for(int r_i=0; r_i<numr1; r_i++)
        for(int c_i=0; c_i<numc1; c_i++)
        {
            const int idx = matget(m, r_i, c_i, numr, numc, true);
            m[idx] = matget(m1, r_i, c_i, numr1, numc1) + matget(v, 0, c_i, numrv, numcv);
        }
}

template <typename scalar_t>
 void transpose(
    scalar_t* m,
    const int numr,
    const int numc,
    scalar_t* m_T)
{
    Eigen::MatrixXf m_m(numr, numc);
    for (int r_i=0; r_i<numr; r_i++)
        for (int c_i=0; c_i<numc; c_i++)
            m_m(r_i, c_i) = matget(m, r_i, c_i, numr, numc);

    for(int r_i=0; r_i<numr; r_i++)
        for(int c_i=0; c_i<numc; c_i++)
        {
            const int idx = matget(m_T, c_i, r_i, numc, numr, true);
            m_T[idx] = m_m(r_i, c_i);
        }
}

template <typename scalar_t>
 void matmul(
    scalar_t* m1,
    const int numr1,
    const int numc1,
    scalar_t* m2,
    const int numr2,
    const int numc2,
    scalar_t* m)
{
    Eigen::MatrixXf m_m1(numr1, numc1);
    for (int r_i=0; r_i<numr1; r_i++)
        for (int c_i=0; c_i<numc1; c_i++)
            m_m1(r_i, c_i) = matget(m1, r_i, c_i, numr1, numc1);

    Eigen::MatrixXf m_m2(numr2, numc2);
    for (int r_i=0; r_i<numr2; r_i++)
        for (int c_i=0; c_i<numc2; c_i++)
            m_m2(r_i, c_i) = matget(m2, r_i, c_i, numr2, numc2);

    Eigen::MatrixXf m_m = m_m1 * m_m2;
    for(int r_i=0; r_i < numr1; r_i++)
        for(int c_i=0; c_i < numc2; c_i++)
        {
            const int idx = matget(m, r_i, c_i, numr1, numc2, true);
            m[idx] = m_m(r_i, c_i);
        }
}

template <typename scalar_t>
 scalar_t jaccard_discrete(
    scalar_t* px1,
    scalar_t* px2,
    const int num_sp)
{
    scalar_t similarity = 0.0;
    int sort_index[MAX_NUM_SP] = {0};
    scalar_t sort_arr[MAX_NUM_SP] = {0.0};
    for (int i=0; i<num_sp; i++)
        sort_arr[i] = px1[i] / (px2[i] + std::numeric_limits<scalar_t>::epsilon());
    argsort(sort_arr, sort_index, num_sp);

    scalar_t px[MAX_NUM_SP] = {0.0};
    scalar_t py[MAX_NUM_SP] = {0.0};
    for (int i=0; i<num_sp; i++)
    {
        px[i] = px1[sort_index[i]];
        py[i] = px2[sort_index[i]];
    }

    scalar_t px_sorted_sum[MAX_NUM_SP] = {0.0};
    scalar_t py_sorted_sum[MAX_NUM_SP] = {0.0};
    py_sorted_sum[0] = py[0];
    for (int i=1; i<num_sp; i++)
    {
        px_sorted_sum[num_sp - i - 1] = px_sorted_sum[num_sp - i] + px[num_sp - i];
        py_sorted_sum[i] = py_sorted_sum[i-1] + py[i];
    }

    long idx[MAX_NUM_SP] = {-1}; int p_idx = 0;
    for (int i=0; i<num_sp; i++)
    {
        if ((px[i] > 0) && (py[i] > 0))
        {
            idx[p_idx] = i;
            p_idx ++;
        }
    }
    for (int i=0; i<p_idx; i++)
    {
        long idx_ = idx[i];
        scalar_t x_y_i = px[idx_] / py[idx_];
        scalar_t tmp = px[idx_] / (px_sorted_sum[idx_] + x_y_i * py_sorted_sum[idx_]);
        similarity += tmp;
    }

    return similarity;
}

template <typename scalar_t>
 scalar_t calc_multivariate_normal_pdf_bev(
    const int D,
    scalar_t* point,
    scalar_t* mean,
    scalar_t* cov)
{
    Eigen::Matrix2f m_cov;
    Eigen::Vector2f v_mean, v_point;
    v_point(0) = matget(point, 0, 0, 1, 2);
    v_point(1) = matget(point, 0, 1, 1, 2);
    v_mean(0) = matget(mean, 0, 0, 1, 2);
    v_mean(1) = matget(mean, 0, 1, 1, 2);
    m_cov(0,0) = matget(cov, 0, 0, 2, 2);
    m_cov(0,1) = matget(cov, 0, 1, 2, 2);
    m_cov(1,0) = matget(cov, 1, 0, 2, 2);
    m_cov(1,1) = matget(cov, 1, 1, 2, 2);

    // - D / 2 * ln (2 * pi)
    scalar_t term1 = - (scalar_t)(D) / 2.0 * logf(2 * PI);
    // - 1 / 2 * ln (m_cov.determinant())
    scalar_t term2 = - 1 / 2.0 * logf(m_cov.determinant());
    // - 1 / 2 * (point - mean)^T m_cov.inverse() (point-mean)
    scalar_t term3 = - 1 / 2.0 * (v_point - v_mean).transpose() * m_cov.inverse() * (v_point - v_mean);

    return exp(term1 + term2 + term3);
}

// Calculate pdf of a multivariate normal distribution
// Args:
// - D: dimension
// - point (D,): x
// - mean (D, ): mean
// - cov (D, D)
float calc_multivariate_normal_pdf_3d(
    const int D,
    Eigen::Vector3f& point,
    Eigen::Vector3f& mean,
    float cov_det,
    Eigen::Matrix3f& icov)
{
    // - D / 2 * ln (2 * pi)
    float term1 = - (float)(D) / 2.0 * logf(2 * PI);
    // - 1 / 2 * ln (m_cov.determinant())
    float term2 = - 1 / 2.0 * logf(cov_det);
    // - 1 / 2 * (point - mean)^T m_cov.inverse() (point-mean)
    float term3 = - 1 / 2.0 * (point - mean).transpose() * icov * (point - mean);

    return exp(term1 + term2 + term3);
}
