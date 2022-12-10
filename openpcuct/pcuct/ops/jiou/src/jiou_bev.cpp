/*
 * File Created: Thu Mar 25 2022

*/
# include "utils.cpp"

inline int divup(int a, int b) 
{
    return (a % b != 0) ? (a / b + 1) : (a / b); 
}

template <typename scalar_t>
 void sample_point_2d(
    const scalar_t* mean,
    const scalar_t* cov,
    const scalar_t mask,
    scalar_t* sp,
    const int num_sp,
    const float range,
    const float grid_size)
{
    scalar_t xc = mean[0]; scalar_t yc = mean[1];
    scalar_t l = mean[2];  scalar_t w = mean[3];
    scalar_t ry = mean[4];
    scalar_t tran_vec[2] = {xc, yc};
    scalar_t scale_m[] = {l, 0, 
                          0, w};
    scalar_t rot_m[] = {cos(ry), -sin(ry),
                        sin(ry),  cos(ry)};
    scalar_t rot_m_T[2*2];
    // sp=z_normed
    gen_meshgrid_2d(sp, num_sp, grid_size, range);
    // sp=z_normed * x0[2:4]
    matmul(sp, num_sp, 2,
           scale_m, 2, 2, sp);
    transpose(rot_m, 2, 2,
            rot_m_T);
    // sp=(z_normed * x0[2:4]) @ rot_m.T
    matmul(sp, num_sp, 2,
           rot_m_T, 2, 2, sp);
    // sp=(z_normed * x0[2:4]) @ rot_m.T + tran_vec
    add_mat_vec(sp, num_sp, 2,
            tran_vec, 1, 2,
            sp, num_sp, 2);
}

template <typename scalar_t>
 void calc_jacobian_znormed_feat0_bev(
    scalar_t* jacobian_point, // 2*6
    scalar_t* z_normed) // 2
{
    for (int i = 0;i < 12; i++)
        jacobian_point[i] = 0;
    jacobian_point[0*6+0] = 1.0;
    jacobian_point[1*6+1] = 1.0;
    jacobian_point[0*6+2] = z_normed[0];
    jacobian_point[1*6+3] = z_normed[0];
    jacobian_point[1*6+4] = z_normed[1];
    jacobian_point[0*6+5] = -z_normed[1];
}

template <typename scalar_t>
 void calc_jacobian_znormed_x0_bev(
    scalar_t* jacobian_point, // 2*5
    const scalar_t* mean, // xc, yc, l, w, ry
    scalar_t* z_normed) // 2
{
    for (int i = 0;i < 10; i++)
        jacobian_point[i] = 0;
    scalar_t l = mean[2];
    scalar_t w = mean[3];
    scalar_t ry = mean[4];
    jacobian_point[0*5+0] = 1.0;
    jacobian_point[1*5+1] = 1.0;
    jacobian_point[0*5+2] = cos(ry) * z_normed[0];
    jacobian_point[0*5+3] =-sin(ry) * z_normed[1];
    jacobian_point[0*5+4] =-sin(ry) * l * z_normed[0] -cos(ry) * w * z_normed[1];
    jacobian_point[1*5+2] = sin(ry) * z_normed[0];
    jacobian_point[1*5+3] = cos(ry) * z_normed[1];
    jacobian_point[1*5+4] = cos(ry) * l * z_normed[0] -sin(ry) * w * z_normed[1];
}

template <typename scalar_t>
 void calc_uncertainty_points_bev(
    const scalar_t* mean, // (5)
    const scalar_t* cov, // (6,6)/(5,5)
    const scalar_t mask, // if 1, cov is (5,5)
    scalar_t* z_normed, // (2)
    scalar_t* center_out, // out (2)
    scalar_t* cov_out) // out (2,2)
{
    int numr_cov = 6; int numc_cov = 6;
    scalar_t jacobian_point[2*6] = {0};
    int numr_j = 2; int numc_j = 6;
    if (mask==1.0)
    {
        numr_cov = 5; numc_cov = 5;
        numr_j = 2; numc_j = 5;
        calc_jacobian_znormed_x0_bev(jacobian_point, mean, z_normed);
    }
    else
        calc_jacobian_znormed_feat0_bev(jacobian_point, z_normed);
 
    scalar_t xc = mean[0]; scalar_t yc = mean[1];
    scalar_t l = mean[2];  scalar_t w = mean[3];
    scalar_t ry = mean[4];
    scalar_t tran_vec[2] = {xc, yc};
    scalar_t scale_m[] = {l, 0, 
                          0, w};
    scalar_t rot_m[] = {cos(ry), -sin(ry),
                     sin(ry),  cos(ry)};
    scalar_t rot_m_T[2*2];
    transpose(rot_m, 2, 2,
            rot_m_T);

    matmul(z_normed, 1, 2,
           scale_m, 2, 2,
           center_out);
    matmul(center_out, 1, 2,
           rot_m_T, 2, 2,
           center_out);
    add_mat_vec(center_out, 1, 2,
        tran_vec, 1, 2,
        center_out, 1, 2);

    scalar_t tmp[2 * 6] = {0};
    scalar_t jacobian_point_T[2*6] = {0};
    transpose(jacobian_point, numr_j, numc_j,
        jacobian_point_T);

    scalar_t cov_[36] = {0};
    for (int i = 0; i < 36; i++)
        cov_[i] = cov[i];
    matmul(jacobian_point, numr_j, numc_j,
           cov_, numr_cov, numc_cov,
           tmp); // numr_j X numc_cov
    matmul(tmp, numr_j, numc_cov,
        jacobian_point_T, numc_j, numr_j,
        cov_out);
}

template <typename scalar_t>
 bool is_cov_valid(
    const scalar_t* cov,
    const scalar_t mask,
    const float grid_size)
{
    bool valid = false;
    int num_cov_rc = 0;
    if (mask == 1)
        num_cov_rc = 6;
    else
        num_cov_rc = 5;
    Eigen::MatrixXf m_cov(num_cov_rc, num_cov_rc);
    for (int r_i=0; r_i < num_cov_rc; r_i++)
        for (int c_i=0; c_i < num_cov_rc; c_i++)
            m_cov(r_i, c_i) = matget(cov, r_i, c_i, num_cov_rc, num_cov_rc);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(m_cov);
    eigensolver.eigenvalues();
    float max_eigenval = -1;
    for (int i=0; i < num_cov_rc; i++)
        if (eigensolver.eigenvalues()[i] > max_eigenval)
            max_eigenval = eigensolver.eigenvalues()[i] ;
    return std::sqrt(max_eigenval) > grid_size * 2;
}

template <typename scalar_t>
 void calculate_probs(
    const scalar_t* mean,
    const scalar_t* cov,
    const scalar_t mask,
    scalar_t* sp,
    scalar_t* px,
    const int num_sp,
    const float grid_size)
{
    bool valid = is_cov_valid(cov, mask, grid_size);
    if (not valid)
    {
        scalar_t xc = mean[0]; scalar_t yc = mean[1];
        scalar_t l = mean[2];  scalar_t w = mean[3];
        scalar_t l2 = l/2.0;   scalar_t w2 = w / 2.0;
        scalar_t ry = mean[4];
        scalar_t tran_vec[2] = {-xc, -yc};
        scalar_t rot_m[] = {cos(ry), -sin(ry),
                            sin(ry),  cos(ry)};
        scalar_t points_aligned_to_pred[MAX_NUM_SP_BEV * BEV_DIM] = {0.0};
        add_mat_vec(sp, num_sp, BEV_DIM,
                    tran_vec, 1, BEV_DIM,
                    points_aligned_to_pred, num_sp, BEV_DIM);
        matmul(points_aligned_to_pred, num_sp, BEV_DIM,
               rot_m, BEV_DIM, BEV_DIM,
               points_aligned_to_pred);
        int clip_idx[MAX_NUM_SP_BEV] = {0}; long pidx = 0;
        for (int i=0; i<num_sp; i++)
        {
            scalar_t x = matget(points_aligned_to_pred, i, 0, num_sp, BEV_DIM);
            scalar_t y = matget(points_aligned_to_pred, i, 1, num_sp, BEV_DIM);
            if ((x >= -l2) && (x < l2) && (y >= -w2) && (y < w2))
            {
                clip_idx[pidx] = i;
                pidx ++;
            }
        }
        if (pidx > 0)
        {
            for(int i=0; i<pidx; i++)
                px[clip_idx[i]] = 1.0 / pidx;
        }
    }
    else
    {
        const int num_z_normed = 1.0 / grid_size * 1.0 / grid_size;
        scalar_t z_normed[MAX_NUM_ZNORMED_BEV * BEV_DIM] = {0};
        gen_meshgrid_2d(z_normed, num_sp, grid_size, (scalar_t)1.0);
        scalar_t centers_out[MAX_NUM_ZNORMED_BEV * BEV_DIM] = {0};
        scalar_t covs_out[MAX_NUM_ZNORMED_BEV * BEV_DIM * BEV_DIM] = {0};

        for (int i=0; i<num_z_normed; i++)
            calc_uncertainty_points_bev(mean, cov, mask,
                &z_normed[i*BEV_DIM], &centers_out[i*BEV_DIM], &covs_out[i*BEV_DIM*BEV_DIM]);
        for (int i=0; i<num_z_normed; i++)
            for (int j=0; j<num_sp; j++)
                px[j] += calc_multivariate_normal_pdf_bev(BEV_DIM,
                    &sp[j*BEV_DIM], &centers_out[i*BEV_DIM], &covs_out[i*BEV_DIM*BEV_DIM]);
        scalar_t sum_px = 0.0;
        for (int i=0; i<num_sp; i++)
            sum_px += px[i];
        for (int i=0; i<num_sp; i++)
            px[i] /= (sum_px+std::numeric_limits<scalar_t>::epsilon());
    }
}

template <typename scalar_t>
 scalar_t inter(
    const scalar_t* mean1,
    const scalar_t* cov1,
    const scalar_t mask1,
    const scalar_t* mean2,
    const scalar_t* cov2,
    const scalar_t mask2,
    const float range,
    const float grid_size)
{
    //local memory
    // sample points according to rbbox1, mean1, cov1, mask1
    int num_sp = range / grid_size * range / grid_size;
    scalar_t sp1[MAX_NUM_SP_BEV * BEV_DIM] = {0};
    scalar_t px1[MAX_NUM_SP_BEV] = {0};
    
    sample_point_2d(mean1, cov1, mask1, sp1, num_sp, range, grid_size);
    calculate_probs(mean1, cov1, mask1, sp1, px1, num_sp, grid_size);
    scalar_t sp2[MAX_NUM_SP_BEV * BEV_DIM] = {0};
    scalar_t px2[MAX_NUM_SP_BEV] = {0};
    calculate_probs(mean2, cov2, mask2, sp1, px2, num_sp, grid_size);
    // compute jious
    return jaccard_discrete(px1, px2, num_sp);
}

torch::Tensor jiou_eval_bev_cpp(
    torch::Tensor means,
    torch::Tensor covs,
    torch::Tensor _masks,
    torch::Tensor query_means,
    torch::Tensor query_covs,
    torch::Tensor _query_masks,
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
            iou[n][k] = inter(mean2, cov2, mask2,
                mean1, cov1, mask1, range, grid_size);
        }
    return iou.to(means_dtype);
}

py::array_t<double> jacobian_z_normed_x0_cpp(
    py::array_t<double> mean,     // 5,
    py::array_t<double> z_normed) // N, 2
{
    py::buffer_info buf_mean = mean.request();
    py::buffer_info buf_z_normed = z_normed.request();
    int N = buf_z_normed.shape[0];

    py::array_t<double> result = py::array_t<double>({N, 2, 5});
    py::buffer_info buf_result = result.request();
    double *ptr_mean = (double *) buf_mean.ptr,
           *ptr_z_normed = (double *) buf_z_normed.ptr,
           *ptr_result = (double *) buf_result.ptr;

    for (int i=0; i<N; i++)
        calc_jacobian_znormed_x0_bev(
            &ptr_result[i * 2 * 5],
            ptr_mean,
            &ptr_z_normed[i * 2]);
    result.resize({N, 2, 5});

    return result;
}

py::array_t<double> jacobian_z_normed_feat0_cpp(
    py::array_t<double> z_normed) // N, 2
{
    py::buffer_info buf_z_normed = z_normed.request();
    int N = buf_z_normed.shape[0];

    py::array_t<double> jacobian_point = py::array_t<double>({N, 2, 6});
    py::buffer_info buf_jacobian_point = jacobian_point.request();
    double *ptr_z_normed = (double *) buf_z_normed.ptr,
           *ptr_jacobian_point = (double *) buf_jacobian_point.ptr;

    for (int i=0; i<N; i++)
        calc_jacobian_znormed_feat0_bev(
            &ptr_jacobian_point[i * 2 * 6],
            &ptr_z_normed[i * 2]);
    jacobian_point.resize({N, 2, 6});

    return jacobian_point;
}

py::array_t<double> sample_prob_cpp(
    py::array_t<double> points, // N, 2
    py::array_t<double> mean, // 5
    py::array_t<double> cov, // 5,5 / 6,6
    float sample_grid)
{
    py::buffer_info buf_points = points.request();
    py::buffer_info buf_mean = mean.request();
    py::buffer_info buf_cov = cov.request();
    double mask = 0;
    int N = buf_points.shape[0];
    if (buf_points.shape[1] != 2)
        throw std::runtime_error("Wrong shape of points.");
    if (buf_mean.shape[0] != 5)
        throw std::runtime_error("Wrong shape of mean.");
    if (buf_cov.shape[0] == 5 && buf_cov.shape[1] == 5)
        mask = 1;
    else if (buf_cov.shape[0] == 6 && buf_cov.shape[1] == 6)
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
    double ptr_cov_[6*6] = {0};
    for (int i=0; i<buf_cov.size; i++)
        ptr_cov_[i] = ptr_cov[i];

    calculate_probs(ptr_mean, ptr_cov_, mask,
        ptr_points, ptr_probs, N, sample_grid);

    probs.resize({N});
    return probs;
}

void calc_uncertainty_points_cpp(
    py::array_t<double> z_normed, // N, 2
    py::array_t<double> mean, // 5
    py::array_t<double> cov, // 5,5 / 6,6
    py::array_t<double> means_out, // N, 2
    py::array_t<double> covs_out)  // N,2,2
{
    py::buffer_info buf_znormed = z_normed.request();
    py::buffer_info buf_mean = mean.request();
    py::buffer_info buf_cov = cov.request();
    py::buffer_info buf_means_out = means_out.request();
    py::buffer_info buf_covs_out = covs_out.request();
    double mask = 0;
    int N = buf_znormed.shape[0];
    if (buf_znormed.shape[1] != 2)
        throw std::runtime_error("Wrong shape of z_normed.");
    if (buf_mean.shape[0] != 5)
        throw std::runtime_error("Wrong shape of mean.");
    if (buf_cov.shape[0] == 5 && buf_cov.shape[1] == 5)
        mask = 1;
    else if (buf_cov.shape[0] == 6 && buf_cov.shape[1] == 6)
        mask = 0;
    else
        throw std::runtime_error("Wrong shape of cov.");
    if (buf_means_out.shape[0] != N or buf_means_out.shape[1] != 2)
        throw std::runtime_error("Wrong shape of means_out.");
    if (buf_covs_out.shape[0] != N \
        or buf_covs_out.shape[1] != 2 \
        or buf_covs_out.shape[2] != 2)
        throw std::runtime_error("Wrong shape of covs_out.");

    double *ptr_znormed = (double *) buf_znormed.ptr,
           *ptr_mean = (double *) buf_mean.ptr,
           *ptr_cov = (double *) buf_cov.ptr,
           *ptr_means_out = (double *) buf_means_out.ptr,
           *ptr_covs_out = (double *) buf_covs_out.ptr;
    double ptr_cov_[6*6] = {0};
    for (int i=0; i<buf_cov.size; i++)
        ptr_cov_[i] = ptr_cov[i];

    for (int i=0; i<N; i++)
        calc_uncertainty_points_bev(
            ptr_mean, ptr_cov_, mask,
            &ptr_znormed[i*BEV_DIM],
            &ptr_means_out[i*BEV_DIM],
            &ptr_covs_out[i*BEV_DIM*BEV_DIM]);

    means_out.resize({N, 2});
    covs_out.resize({N ,2, 2});
}

// TODO: Expose more functions
// TODO: Use the same base utils with jiou_3d.cpp
// TODO: Change file name as jiou_bev.cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("jiou_eval_bev_cpp", &jiou_eval_bev_cpp, "jiou_eval_bev_cpp (CPP)");
  m.def("jacobian_z_normed_x0_cpp", &jacobian_z_normed_x0_cpp, "jacobian_z_normed_x0_cpp(CPP)");
  m.def("jacobian_z_normed_feat0_cpp", &jacobian_z_normed_feat0_cpp, "jacobian_z_normed_feat0_cpp(CPP)");
  m.def("sample_prob_cpp", &sample_prob_cpp, "sample_prob_cpp(CPP)");
  m.def("calc_uncertainty_points_cpp", &calc_uncertainty_points_cpp, "calc_uncertainty_points(CPP)");
}
