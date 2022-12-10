/*
 * File Created: Wed Mar 04 2020

 * This is converted from the python code:
 * https://github.com/traveller59/second.pytorch/blob/master/second/core/non_max_suppression/
 * https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

inline int divup(int a, int b) 
{
    return (a % b != 0) ? (a / b + 1) : (a / b); 
}

__device__ float trangle_area(float* a, float* b, float* c)
{
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0;
}

__device__ float area(
    float* int_pts,
    int num_of_inter)
{
    float area_val = 0.0;
    for(int i=0; i<num_of_inter-2; i++)
    {
        float a[2] = {0};
        float b[2] = {0};
        float c[2] = {0};
        a[0] = int_pts[0]; a[1] = int_pts[1];
        b[0] = int_pts[2 * i + 2]; b[1] = int_pts[2 * i + 3];
        c[0] = int_pts[2 * i + 4]; c[1] = int_pts[2 * i + 5];
        area_val += abs(trangle_area(a, b, c));
    }
    return area_val;
}

__device__ void sort_vertex_in_convex_polygon(
    float* int_pts,
    int num_of_inter)
{
    if (num_of_inter > 0)
    {
        float center[2] = {0};
        for(int i=0; i<num_of_inter; i++)
        {
            center[0] += int_pts[2 * i];
            center[1] += int_pts[2 * i + 1];
        }
        center[0] /= num_of_inter;
        center[1] /= num_of_inter;
        float v[2] = {0};
        float vs[16] = {0};
        for(int i=0; i<num_of_inter; i++)
        {
            v[0] = int_pts[2 * i] - center[0];
            v[1] = int_pts[2 * i + 1] - center[1];
            float d = sqrt(v[0] * v[0] + v[1] * v[1]);
            v[0] = v[0] / d;
            v[1] = v[1] / d;
            if (v[1] < 0)
                v[0] = -2 - v[0];
            vs[i] = v[0];
        }
        int j = 0;
        float temp = 0;
        for(int i=1; i<num_of_inter; i++)
        {
            if (vs[i - 1] > vs[i])
            {
                temp = vs[i];
                float tx = int_pts[2 * i];
                float ty = int_pts[2 * i + 1];
                j = i;
                while ((j > 0) && (vs[j - 1] > temp))
                {
                    vs[j] = vs[j - 1];
                    int_pts[j * 2] = int_pts[j * 2 - 2];
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1];
                    j -= 1;
                }
                vs[j] = temp;
                int_pts[j * 2] = tx;
                int_pts[j * 2 + 1] = ty;
            }
        }
    }
}

template <typename scalar_t>
__device__ bool line_segment_intersection(
    scalar_t* pts1,
    scalar_t* pts2,
    int i,
    int j,
    float* temp_pts)
{
    float A[2] = {0};
    float B[2] = {0};
    float C[2] = {0};
    float D[2] = {0};

    A[0] = pts1[2 * i];
    A[1] = pts1[2 * i + 1];

    B[0] = pts1[2 * ((i + 1) % 4)];
    B[1] = pts1[2 * ((i + 1) % 4) + 1];

    C[0] = pts2[2 * j];
    C[1] = pts2[2 * j + 1];

    D[0] = pts2[2 * ((j + 1) % 4)];
    D[1] = pts2[2 * ((j + 1) % 4) + 1];
    float BA0 = B[0] - A[0];
    float BA1 = B[1] - A[1];
    float DA0 = D[0] - A[0];
    float CA0 = C[0] - A[0];
    float DA1 = D[1] - A[1];
    float CA1 = C[1] - A[1];
    bool acd = DA1 * CA0 > CA1 * DA0;
    bool bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0]);
    if (acd != bcd)
    {
        bool abc = CA1 * BA0 > BA1 * CA0;
        bool abd = DA1 * BA0 > BA1 * DA0;
        if (abc != abd)
        {
            float DC0 = D[0] - C[0];
            float DC1 = D[1] - C[1];
            float ABBA = A[0] * B[1] - B[0] * A[1];
            float CDDC = C[0] * D[1] - D[0] * C[1];
            float DH = BA1 * DC0 - BA0 * DC1;
            float Dx = ABBA * DC0 - BA0 * CDDC;
            float Dy = ABBA * DC1 - BA1 * CDDC;
            temp_pts[0] = Dx / DH;
            temp_pts[1] = Dy / DH;
            return true;
        }
    }
    return false;
}


template <typename scalar_t>
__device__ bool point_in_quadrilateral(scalar_t pt_x, scalar_t pt_y, scalar_t* corners)
{
    scalar_t ab0 = corners[2] - corners[0];
    scalar_t ab1 = corners[3] - corners[1];

    scalar_t ad0 = corners[6] - corners[0];
    scalar_t ad1 = corners[7] - corners[1];

    scalar_t ap0 = pt_x - corners[0];
    scalar_t ap1 = pt_y - corners[1];

    scalar_t abab = ab0 * ab0 + ab1 * ab1;
    scalar_t abap = ab0 * ap0 + ab1 * ap1;
    scalar_t adad = ad0 * ad0 + ad1 * ad1;
    scalar_t adap = ad0 * ap0 + ad1 * ap1;

    scalar_t eps = -1e-6;
    return abab - abap >= eps and abap >= eps and adad - adap >= eps and adap >= eps;

}

template <typename scalar_t>
__device__ int quadrilateral_intersection(scalar_t* pts1, scalar_t* pts2, float* int_pts)
{
    int num_of_inter = 0;
    for(int i=0; i<4; i++)
    {
        if (point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2))
        {
            int_pts[num_of_inter * 2] = pts1[2 * i];
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
            num_of_inter += 1;
        }
        if (point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1))
        {
            int_pts[num_of_inter * 2] = pts2[2 * i];
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
            num_of_inter += 1;
        }
    }
    float temp_pts[2] = {0};
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
        {
            bool has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts);
            if (has_pts)
            {
                int_pts[num_of_inter * 2] = temp_pts[0];
                int_pts[num_of_inter * 2 + 1] = temp_pts[1];
                num_of_inter += 1;
            }
        }
    return num_of_inter;
}

template <typename scalar_t>
__device__ void rbbox_to_corners(float* corners, scalar_t* rbbox)
{
    scalar_t angle = -rbbox[4];
    scalar_t a_cos = cos(angle);
    scalar_t a_sin = sin(angle);
    scalar_t center_x = rbbox[0]; scalar_t center_y = rbbox[1];
    scalar_t x_d = rbbox[2]; scalar_t y_d = rbbox[3];
    scalar_t corners_x[4] = {-x_d / 2, -x_d / 2, x_d / 2, x_d / 2};
    scalar_t corners_y[4] = {-y_d / 2, y_d / 2, y_d / 2, -y_d / 2};
    for(int i=0; i<4; i++)
    {
        corners[2 * i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x;
        corners[2 * i + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y;
    }
}

template <typename scalar_t>
__device__ scalar_t inter(scalar_t* rbbox1, scalar_t* rbbox2)
{
    //local memory
    float corners1[8]= {0};
    float corners2[8] = {0};
    float intersection_corners[16] = {0};
    rbbox_to_corners(corners1, rbbox1);
    rbbox_to_corners(corners2, rbbox2);
    int num_intersection = quadrilateral_intersection(corners1, corners2, intersection_corners);
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection);
    return area(intersection_corners, num_intersection);
}

template <typename scalar_t>
__global__ void compute_intersect_2drot_cuda_kernel(
    const int N,
    const int K,
    scalar_t* __restrict__ boxes,
    scalar_t* __restrict__ quey_boxes,
    scalar_t* __restrict__ iou
    ) 
{
    const int threadsPerBlock = 8 * 8;
    const int row_start = blockIdx.x;
    const int col_start = blockIdx.y;
    const int tx = threadIdx.x;
    const int row_size = min(N - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size = min(K - col_start * threadsPerBlock, threadsPerBlock);
    __shared__ scalar_t block_boxes[64 * 5];
    __shared__ scalar_t block_qboxes[64 * 5];
    const int dev_query_box_idx = threadsPerBlock * col_start + tx;
    const int dev_box_idx = threadsPerBlock * row_start + tx;
    if (tx < col_size)
    {        
        block_qboxes[tx * 5 + 0] = quey_boxes[dev_query_box_idx * 5 + 0];
        block_qboxes[tx * 5 + 1] = quey_boxes[dev_query_box_idx * 5 + 1];
        block_qboxes[tx * 5 + 2] = quey_boxes[dev_query_box_idx * 5 + 2];
        block_qboxes[tx * 5 + 3] = quey_boxes[dev_query_box_idx * 5 + 3];
        block_qboxes[tx * 5 + 4] = quey_boxes[dev_query_box_idx * 5 + 4];
    }
    if (tx < row_size)
    {
        block_boxes[tx * 5 + 0] = boxes[dev_box_idx * 5 + 0];
        block_boxes[tx * 5 + 1] = boxes[dev_box_idx * 5 + 1];
        block_boxes[tx * 5 + 2] = boxes[dev_box_idx * 5 + 2];
        block_boxes[tx * 5 + 3] = boxes[dev_box_idx * 5 + 3];
        block_boxes[tx * 5 + 4] = boxes[dev_box_idx * 5 + 4];
    }
    __syncthreads();
    if (tx < row_size)
        // printf("e%d %d", row_start, col_start);
        for(int i=0; i<col_size; i++)
        {
            const int offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i;
            const scalar_t rbbox1[] = {block_qboxes[i * 5], block_qboxes[i * 5 + 1],
                block_qboxes[i * 5 + 2], block_qboxes[i * 5 + 3], block_qboxes[i * 5 + 4]};
            const scalar_t rbbox2[] = {block_boxes[tx * 5], block_boxes[tx * 5 + 1],
                block_boxes[tx * 5 + 2], block_boxes[tx * 5 + 3], block_boxes[tx * 5 + 4]};
            iou[offset] = inter(rbbox1, rbbox2);
        }
}

torch::Tensor compute_intersect_2drot_cuda(torch::Tensor boxes, torch::Tensor query_boxes)
{
    auto boxes_dtype = boxes.dtype();
    auto boxes_dev = boxes.to(torch::kFloat32);
    auto query_boxes_dev = query_boxes.to(torch::kFloat32);
    const int N = boxes.size(0);
    const int K = query_boxes.size(0);
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCUDA, 0);
    auto iou = torch::zeros({N, K}, options);
    if (N == 0 or K == 0)
        return iou.to(boxes_dtype);
    const int threadsPerBlock = 8 * 8;
    const dim3 blockspergrid(divup(N, threadsPerBlock), divup(K, threadsPerBlock));
    // printf("e%d %d %d\n", N, threadsPerBlock, divup(N, threadsPerBlock));
    // printf("e%d %d %d\n", K, threadsPerBlock, divup(K, threadsPerBlock));
    AT_DISPATCH_FLOATING_TYPES(boxes_dev.type(), "compute_intersect_2drot_cuda", ([&]{
        compute_intersect_2drot_cuda_kernel<scalar_t><<<blockspergrid, threadsPerBlock>>>(
            /*N=*/N,
            /*K=*/K,
            /*boxes=*/boxes_dev.data<scalar_t>(),
            /*quey_boxes=*/query_boxes_dev.data<scalar_t>(),
            /*iou=*/iou.data<scalar_t>()
            );
    }));
    return iou.to(boxes_dtype);
}