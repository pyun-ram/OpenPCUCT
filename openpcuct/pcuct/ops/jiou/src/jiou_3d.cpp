/*
 * File Created: Thu Mar 25 2022

*/
# include "utils.cpp"
# include <omp.h>

// Sample point in 3D space
// Args:
// - mean (7,): [xc, yc, zc, l, w, h, ry]
// - cov (8*8)/(7*7): 
// - mask: if 1, cov shape is (7*7)
// - sp (num_sp,3): sample points (output)
// - num_sp: number of sample points
// - range: sample range
// - grid_size: sample grid_size
template <typename scalar_t>
 void sample_point_3d(
    const scalar_t* mean, // 7, xc, yc, zc, l, w, h, ry
    const scalar_t* cov, // 8*8
    const scalar_t mask,
    scalar_t* sp,
    const int num_sp,
    const float range,
    const float grid_size)
{
    scalar_t xc = mean[0]; scalar_t yc = mean[1]; scalar_t zc = mean[2];
    scalar_t l = mean[3];  scalar_t w = mean[4];  scalar_t h = mean[5];
    scalar_t ry = mean[6];
    scalar_t tran_vec[] = {xc, yc, zc};
    scalar_t scale_m[] = {l, 0, 0,
                          0, w, 0,
                          0, 0, h};
    scalar_t rot_m[] = {cos(ry), -sin(ry), 0,
                        sin(ry),  cos(ry), 0,
                              0,        0, 1};
    scalar_t rot_m_T[3*3];

    // sp=z_normed
    gen_meshgrid_3d(sp, num_sp, grid_size, range);
    // sp=z_normed * x0[3:6]
    matmul(sp, num_sp, 3,
           scale_m, 3, 3, sp);
    transpose(rot_m, 3, 3,
            rot_m_T);
    // sp=(z_normed * x0[3:6]) @ rot_m.T
    matmul(sp, num_sp,3,
           rot_m_T, 3, 3, sp);
    // sp=(z_normed * x0[3:6]) @ rot_m.T + tran_vec
    add_mat_vec(sp, num_sp, 3,
            tran_vec, 1, 3,
            sp, num_sp, 3);
}

// Validate <cov> (3D)
// Args:
// - cov (8*8)/(7*7): 
// - mask: if 1, cov shape is (7*7)
// - grid_size: sample grid_size
// Returns:
// - valid: True, if the max eigen value of <cov>
// is greater than 5 * grid_size.
template <typename scalar_t>
 bool valid_cov_3d(
    const scalar_t* cov,
    const scalar_t mask,
    const float grid_size)
{
    int num_cov_rc = 0;
    if (mask == 1)
        num_cov_rc = 7;
    else
        num_cov_rc = 8;
    Eigen::MatrixXf m_cov(num_cov_rc, num_cov_rc);
    for (int r_i=0; r_i < num_cov_rc; r_i++)
        for (int c_i=0; c_i < num_cov_rc; c_i++)
            m_cov(r_i, c_i) = matget(cov, r_i, c_i, num_cov_rc, num_cov_rc);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(m_cov);
    float max_eigenval = -1;
    for (int i=0; i < num_cov_rc; i++)
        if (eigensolver.eigenvalues()[i] > max_eigenval)
            max_eigenval = eigensolver.eigenvalues()[i] ;
    return std::sqrt(max_eigenval) > grid_size * 2;
}

// Calculate pdf of uniform distribution
// Args:
// - sp (num_sp, 3): sample points
// - px (num_sp, ): probabilities of sample points (output)
// - mean (7,): [xc, yc, zc, l, w, h, ry]
// - num_sp: number of sample points
template <typename scalar_t>
 void calc_uniform_pdf_3d(
     scalar_t* sp,
     scalar_t* px,
     const scalar_t* mean,
     const int num_sp)
{
        scalar_t xc = mean[0]; scalar_t yc = mean[1]; scalar_t zc = mean[2];
        scalar_t l = mean[3];  scalar_t w = mean[4];  scalar_t h = mean[5];
        scalar_t l2 = l/2.0;   scalar_t w2 = w / 2.0; scalar_t h2 = h / 2.0;
        scalar_t ry = mean[6];
        scalar_t tran_vec[3] = {-xc, -yc, -zc};
        scalar_t rot_m[] = {cos(ry), -sin(ry), 0,
                            sin(ry),  cos(ry), 0,
                                  0,        0, 1};
        scalar_t points_aligned_to_pred[MAX_NUM_SP_3D * D3_DIM] = {0.0};
        add_mat_vec(sp, num_sp, D3_DIM,
                    tran_vec, 1, D3_DIM,
                    points_aligned_to_pred, num_sp, D3_DIM);
        matmul(points_aligned_to_pred, num_sp, D3_DIM,
               rot_m, D3_DIM, D3_DIM,
               points_aligned_to_pred);
        int clip_idx[MAX_NUM_SP_3D] = {0}; long pidx = 0;
        for (int i=0; i<num_sp; i++)
        {
            scalar_t x = matget(points_aligned_to_pred, i, 0, num_sp, D3_DIM);
            scalar_t y = matget(points_aligned_to_pred, i, 1, num_sp, D3_DIM);
            scalar_t z = matget(points_aligned_to_pred, i, 2, num_sp, D3_DIM);
            if ((x >= -l2) && (x < l2) &&
                (y >= -w2) && (y < w2) &&
                (z >= -h2) && (z < h2))
            {
                clip_idx[pidx] = i;
                pidx ++;
            }
        }
        if (pidx > 0)
            for(int i=0; i<pidx; i++)
                px[clip_idx[i]] = 1.0 / pidx;
}

// Calculate Jacobian matrix
// Args:
// jacobian_point (3,8): Jacobian matrix (output)
// z_normed (3): znormed
template <typename scalar_t>
 void calc_jacobian_znormed_feat0_3d(
    scalar_t* jacobian_point, // 3*8
    scalar_t* z_normed) // 3
{
    for (int i = 0;i < 24; i++)
        jacobian_point[i] = 0;
    jacobian_point[0*8+0] = 1.0;
    jacobian_point[1*8+1] = 1.0;
    jacobian_point[2*8+2] = 1.0;
    jacobian_point[0*8+3] = z_normed[0];
    jacobian_point[0*8+6] =-z_normed[1];
    jacobian_point[1*8+4] = z_normed[0];
    jacobian_point[1*8+5] = z_normed[1];
    jacobian_point[2*8+7] = z_normed[2];
}

template <typename scalar_t>
 void calc_jacobian_znormed_x0_3d(
    scalar_t* jacobian_point, // 3*7
    const scalar_t* mean, // xc, yc, zc, l, w, h, ry
    scalar_t* z_normed) // 3
{
    for (int i = 0;i < 21; i++)
        jacobian_point[i] = 0;
    scalar_t l = mean[3];
    scalar_t w = mean[4];
    scalar_t ry = mean[6];
    jacobian_point[0*7+0] = 1.0;
    jacobian_point[1*7+1] = 1.0;
    jacobian_point[2*7+2] = 1.0;
    jacobian_point[0*7+3] = cos(ry) * z_normed[0];
    jacobian_point[0*7+4] =-sin(ry) * z_normed[1];
    jacobian_point[0*7+6] =-sin(ry) * z_normed[0] * l
                           -cos(ry) * z_normed[1] * w;
    jacobian_point[1*7+3] = sin(ry) * z_normed[0];
    jacobian_point[1*7+4] = cos(ry) * z_normed[1];
    jacobian_point[1*7+6] = cos(ry) * z_normed[0] * l
                           -sin(ry) * z_normed[1] * w;
    jacobian_point[2*7+5] = z_normed[2];
}

// Calculate uncertainty points
// Args:
// - mean (7,): [xc, yc, zc, l, w, h, ry]
// - cov (8*8)/(7*7): 
// - mask: if 1, cov shape is (7*7)
// - znormed (3,): [znormed_x, znormed_y, znormed_z]
// - center_out (3, ): mean of output
// - cov_out (3, 3): cov of output
template <typename scalar_t>
 void calc_uncertainty_points_3d(
    const scalar_t* mean,
    const scalar_t* cov,
    const scalar_t mask,
    scalar_t* z_normed,
    scalar_t* center_out,
    scalar_t* cov_out)
{
    int numr_cov = 8; int numc_cov = 8;
    scalar_t jacobian_point[3*8] = {0};
    int numr_j = 3; int numc_j = 8;
    if (mask==1.0)
    {
        numr_cov = 7; numc_cov = 7;
        numr_j = 3; numc_j = 7;
        calc_jacobian_znormed_x0_3d(jacobian_point, mean, z_normed);
    }
    else
        calc_jacobian_znormed_feat0_3d(jacobian_point, z_normed);
 
    scalar_t xc = mean[0]; scalar_t yc = mean[1]; scalar_t zc = mean[2];
    scalar_t l = mean[3];  scalar_t w = mean[4];  scalar_t h = mean[5];
    scalar_t ry = mean[6];
    scalar_t tran_vec[] = {xc, yc, zc};
    scalar_t scale_m[] = {l, 0, 0,
                          0, w, 0,
                          0, 0, h};
    scalar_t rot_m[] = {cos(ry), -sin(ry), 0,
                        sin(ry),  cos(ry), 0,
                              0,        0, 1};
    scalar_t rot_m_T[3*3];
    transpose(rot_m, 3, 3,
              rot_m_T);

    matmul(z_normed, 1, 3,
           scale_m, 3, 3,
           center_out);
    matmul(center_out, 1, 3,
           rot_m_T, 3, 3,
           center_out);
    add_mat_vec(center_out, 1, 3,
        tran_vec, 1, 3,
        center_out, 1, 3);

    scalar_t tmp[3 * 8] = {0};
    scalar_t jacobian_point_T[8 * 3] = {0};
    transpose(jacobian_point, numr_j, numc_j,
        jacobian_point_T);

    scalar_t cov_[8*8] = {0};
    for (int i = 0; i < 8*8; i++)
        cov_[i] = cov[i];
    matmul(jacobian_point, numr_j, numc_j,
           cov_, numr_cov, numc_cov,
           tmp); // numr_j X numc_cov
    matmul(tmp, numr_j, numc_cov,
        jacobian_point_T, numc_j, numr_j,
        cov_out);
}

template <typename scalar_t>
 void calculate_probs_3d(
    const scalar_t* mean,
    const scalar_t* cov,
    const scalar_t mask,
    scalar_t* sp,
    scalar_t* px,
    const int num_sp,
    const float grid_size)
{
    if (not valid_cov_3d(cov, mask, grid_size))
    {
        calc_uniform_pdf_3d(sp, px, mean, num_sp);
    }
    else
    {
        const int num_z_normed = 1.0 / grid_size * 1.0 / grid_size * 1.0 / grid_size;
        scalar_t z_normed[MAX_NUM_ZNORMED_3D * D3_DIM] = {0};
        scalar_t centers_out[MAX_NUM_ZNORMED_3D * D3_DIM] = {0};
        scalar_t covs_out[MAX_NUM_ZNORMED_3D * D3_DIM * D3_DIM] = {0};

        gen_meshgrid_3d(z_normed, num_sp, grid_size, (scalar_t)1.0);
        for (int i=0; i<num_z_normed; i++)
            calc_uncertainty_points_3d(mean, cov, mask,
                &z_normed[i*D3_DIM],
                &centers_out[i*D3_DIM], &covs_out[i*D3_DIM*D3_DIM]);
        for (int i=0; i<num_z_normed; i++)
        {
            Eigen::Matrix3f m_cov, m_icov;
            Eigen::Vector3f v_mean;
            for (int k=0; k<D3_DIM; k++)
            {
                v_mean(k) = matget(centers_out, i, k, num_z_normed, D3_DIM);
                for (int l=0; l<D3_DIM; l++)
                    m_cov(k, l) = matget(covs_out, i, k*D3_DIM+l,
                        num_z_normed, D3_DIM*D3_DIM);
            }
            m_icov = m_cov.inverse();
            float cov_det = m_cov.determinant();
            #pragma omp parallel for
            for (int j=0; j<num_sp; j++)
            {
                Eigen::Vector3f v_point = {
                    sp[j*D3_DIM+0], sp[j*D3_DIM+1], sp[j*D3_DIM+2]};
                px[j] += calc_multivariate_normal_pdf_3d(D3_DIM,
                    v_point, v_mean, cov_det, m_icov);
            }
        }

        scalar_t sum_px = 0.0;
        for (int i=0; i<num_sp; i++)
            sum_px += px[i];
        for (int i=0; i<num_sp; i++)
            px[i] /= (sum_px+std::numeric_limits<scalar_t>::epsilon());
    }
}

template <typename scalar_t>
 scalar_t inter_3d(
    const scalar_t* mean1, // 7, xc, yc, zc, l, w, h, ry
    const scalar_t* cov1, // 8*8
    const scalar_t mask1, 
    const scalar_t* mean2, // 7, xc, yc, zc, l, w, h, ry
    const scalar_t* cov2, // 8*8
    const scalar_t mask2,
    const float range,
    const float grid_size)
{
    int num_sp = range / grid_size * range / grid_size * range / grid_size;
    scalar_t sp1[MAX_NUM_SP_3D * D3_DIM] = {0};
    scalar_t px1[MAX_NUM_SP_3D] = {0};
    scalar_t px2[MAX_NUM_SP_3D] = {0};
    
    sample_point_3d(mean1, cov1, mask1, sp1, num_sp, range, grid_size);
    calculate_probs_3d(mean1, cov1, mask1, sp1, px1, num_sp, grid_size);

    calculate_probs_3d(mean2, cov2, mask2, sp1, px2, num_sp, grid_size);
    return jaccard_discrete(px1, px2, num_sp);
}

torch::Tensor jiou_eval_3d_cpp(
    torch::Tensor means, //N, 7
    torch::Tensor covs,  //N, 8*8
    torch::Tensor _masks, //N,
    torch::Tensor query_means, //M, 7
    torch::Tensor query_covs, //M, 8*8
    torch::Tensor _query_masks, //M
    const float range,
    const float grid_size)
{
    auto means_dtype = means.dtype();
    // convert dtypes
    auto means_dev = means.to(torch::kFloat32);
    auto covs_dev = covs.to(torch::kFloat32);
    auto _masks_dev = _masks.to(torch::kFloat32);
    auto query_means_dev = query_means.to(torch::kFloat32);
    auto query_covs_dev = query_covs.to(torch::kFloat32);
    auto _query_masks_dev = _query_masks.to(torch::kFloat32);

    const int N = means.size(0);
    const int K = query_means.size(0);
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided);
    auto iou = torch::zeros({N, K}, options);
    if (N == 0 or K == 0)
        return iou.to(means_dtype);
    for (int n=0; n<N; n++)
        for(int k=0; k<K; k++)
        {
            float* mean1 = query_means_dev[k].data<float>();
            float* cov1 = query_covs_dev[k].data<float>();
            float mask1 = _query_masks_dev[k].data<float>()[0];
            float* mean2 = means_dev[n].data<float>();
            float* cov2 = covs_dev[n].data<float>();
            float mask2 = _masks_dev[n].data<float>()[0];
            iou[n][k] = inter_3d(mean2, cov2, mask2,
                mean1, cov1, mask1, range, grid_size);
        }
    return iou.to(means_dtype);
}

py::array_t<double> jacobian_z_normed_x0_cpp(
    py::array_t<double> mean,     // 7,
    py::array_t<double> z_normed) // N, 3
{
    py::buffer_info buf_mean = mean.request();
    py::buffer_info buf_z_normed = z_normed.request();
    int N = buf_z_normed.shape[0];

    py::array_t<double> result = py::array_t<double>({N, 3, 7});
    py::buffer_info buf_result = result.request();
    double *ptr_mean = (double *) buf_mean.ptr,
           *ptr_z_normed = (double *) buf_z_normed.ptr,
           *ptr_result = (double *) buf_result.ptr;

    for (int i=0; i<N; i++)
        calc_jacobian_znormed_x0_3d(
            &ptr_result[i * 3 * 7],
            ptr_mean,
            &ptr_z_normed[i * 3]);
    result.resize({N, 3, 7});

    return result;
}

py::array_t<double> jacobian_z_normed_feat0_cpp(
    py::array_t<double> z_normed) // N, 3
{
    py::buffer_info buf_z_normed = z_normed.request();
    int N = buf_z_normed.shape[0];

    py::array_t<double> jacobian_point = py::array_t<double>({N, 3, 8});
    py::buffer_info buf_jacobian_point = jacobian_point.request();
    double *ptr_z_normed = (double *) buf_z_normed.ptr,
           *ptr_jacobian_point = (double *) buf_jacobian_point.ptr;

    for (int i=0; i<N; i++)
        calc_jacobian_znormed_feat0_3d(
            &ptr_jacobian_point[i * 3 * 8],
            &ptr_z_normed[i * 3]);
    jacobian_point.resize({N, 3, 8});

    return jacobian_point;
}

py::array_t<double> sample_prob_cpp(
    py::array_t<double> points, // N, 3
    py::array_t<double> mean, // 7
    py::array_t<double> cov, // 7,7 / 8,8
    float sample_grid)
{
    py::buffer_info buf_points = points.request();
    py::buffer_info buf_mean = mean.request();
    py::buffer_info buf_cov = cov.request();
    double mask = 0;
    int N = buf_points.shape[0];
    if (buf_points.shape[1] != 3)
        throw std::runtime_error("Wrong shape of points.");
    if (buf_mean.shape[0] != 7)
        throw std::runtime_error("Wrong shape of mean.");
    if (buf_cov.shape[0] == 7 && buf_cov.shape[1] == 7)
        mask = 1;
    else if (buf_cov.shape[0] == 8 && buf_cov.shape[1] == 8)
        mask = 0;
    else
        throw std::runtime_error("Wrong shape of cov.");

    py::array_t<double> probs = py::array_t<double>({N});
    py::buffer_info buf_probs = probs.request();
    double *ptr_points = (double *) buf_points.ptr,
           *ptr_mean = (double *) buf_mean.ptr,
           *ptr_cov = (double *) buf_cov.ptr,
           *ptr_probs = (double *) buf_probs.ptr;
    for (int i=0; i<N; i++)
        ptr_probs[i] = 0;
    double ptr_cov_[8*8] = {0};
    for (int i=0; i<buf_cov.size; i++)
        ptr_cov_[i] = ptr_cov[i];

    calculate_probs_3d(ptr_mean, ptr_cov_, mask,
        ptr_points, ptr_probs, N, sample_grid);

    probs.resize({N});
    return probs;
}

void calc_uncertainty_points_cpp(
    py::array_t<double> z_normed, // N, 3
    py::array_t<double> mean, // 7
    py::array_t<double> cov, // 7,7 / 8,8
    py::array_t<double> means_out, // N, 3
    py::array_t<double> covs_out)  // N,3,3
{
    py::buffer_info buf_znormed = z_normed.request();
    py::buffer_info buf_mean = mean.request();
    py::buffer_info buf_cov = cov.request();
    py::buffer_info buf_means_out = means_out.request();
    py::buffer_info buf_covs_out = covs_out.request();
    double mask = 0;
    int N = buf_znormed.shape[0];
    if (buf_znormed.shape[1] != 3)
        throw std::runtime_error("Wrong shape of z_normed.");
    if (buf_mean.shape[0] != 7)
        throw std::runtime_error("Wrong shape of mean.");
    if (buf_cov.shape[0] == 7 && buf_cov.shape[1] == 7)
        mask = 1;
    else if (buf_cov.shape[0] == 8 && buf_cov.shape[1] == 8)
        mask = 0;
    else
        throw std::runtime_error("Wrong shape of cov.");
    if (buf_means_out.shape[0] != N or buf_means_out.shape[1] != 3)
        throw std::runtime_error("Wrong shape of means_out.");
    if (buf_covs_out.shape[0] != N \
        or buf_covs_out.shape[1] != 3 \
        or buf_covs_out.shape[2] != 3)
        throw std::runtime_error("Wrong shape of covs_out.");

    double *ptr_znormed = (double *) buf_znormed.ptr,
           *ptr_mean = (double *) buf_mean.ptr,
           *ptr_cov = (double *) buf_cov.ptr,
           *ptr_means_out = (double *) buf_means_out.ptr,
           *ptr_covs_out = (double *) buf_covs_out.ptr;
    double ptr_cov_[8*8] = {0};
    for (int i=0; i<buf_cov.size; i++)
        ptr_cov_[i] = ptr_cov[i];

    for (int i=0; i<N; i++)
        calc_uncertainty_points_3d(
            ptr_mean, ptr_cov_, mask,
            &ptr_znormed[i*D3_DIM],
            &ptr_means_out[i*D3_DIM], &ptr_covs_out[i*D3_DIM*D3_DIM]);

    means_out.resize({N, 3});
    covs_out.resize({N ,3, 3});
}

double jaccard_discrete_cpp(
    py::array_t<double> px, // N, 3
    py::array_t<double> py)
{
    py::buffer_info buf_px = px.request();
    py::buffer_info buf_py = py.request();
    int N = buf_px.shape[0];
    if (buf_px.size != N || buf_py.size != N)
        throw std::runtime_error("Wrong shape.");

    double *ptr_px = (double *) buf_px.ptr,
           *ptr_py = (double *) buf_py.ptr;

    return jaccard_discrete(ptr_px, ptr_py, N);
}

// TODO: Expose more functions
// TODO: Use the same base utils with jiou_3d.cpp
// TODO: Change file name as jiou_bev.cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("jiou_eval_3d_cpp", &jiou_eval_3d_cpp, "jiou_eval_3d_cpp (CPP)");
  m.def("get_max_threads", &omp_get_max_threads, "return max number of threads");
  m.def("set_num_threads", &omp_set_num_threads, "set number of threads");
  m.def("jacobian_z_normed_x0_cpp", &jacobian_z_normed_x0_cpp, "jacobian_z_normed_x0_cpp(CPP)");
  m.def("jacobian_z_normed_feat0_cpp", &jacobian_z_normed_feat0_cpp, "jacobian_z_normed_feat0_cpp(CPP)");
  m.def("sample_prob_cpp", &sample_prob_cpp, "sample_prob_cpp(CPP)");
  m.def("calc_uncertainty_points_cpp", &calc_uncertainty_points_cpp, "calc_uncertainty_points(CPP)");
  m.def("jaccard_discrete_cpp", &jaccard_discrete_cpp, "jaccard_discrete(CPP)");
}