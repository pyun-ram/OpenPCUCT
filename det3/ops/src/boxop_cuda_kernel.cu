/*
 * File Created: Thu Mar 05 2020

*/
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

inline int divup(int a, int b) 
{
    return (a % b != 0) ? (a / b + 1) : (a / b); 
}

template <typename scalar_t>
__device__ bool is_pt_in_box(scalar_t* pt, scalar_t* box)
{
    // scalar_t x = pt[0];
    // return x > 0;
    float center_x = box[0]; float center_y = box[1]; float center_z = box[2] + box[5]/2.0;
    float d_x = box[3]; float d_y = box[4]; float d_z = box[5];
    float ry = -box[6]; float c = cos(ry); float s = sin(ry);
    float pt_x = pt[0]; float pt_y = pt[1]; float pt_z = pt[2];

    // translate pt with -(center_x, center_y, center_z)
    pt_x -= center_x; pt_y -= center_y; pt_z -= center_z;
    // rotate pt with -ry
    float final_pt_x = c * pt_x - s * pt_y;
    float final_pt_y = s * pt_x + c * pt_y;
    float final_pt_z = pt_z;

    bool bool_x = (-d_x/2.0<=final_pt_x) and (final_pt_x<=d_x/2.0);
    bool bool_y = (-d_y/2.0<=final_pt_y) and (final_pt_y<=d_y/2.0);
    bool bool_z = (-d_z/2.0<=final_pt_z) and (final_pt_z<=d_z/2.0);
    return bool_x and bool_y and bool_z;
}

template <typename scalar_t>
__global__ void crop_pts_3drot_cuda_kernel(
    const int M,
    const int N,
    scalar_t* __restrict__ boxes,
    scalar_t* __restrict__ pts,
    bool* __restrict__ masks
)
{
    // config row, col
    // copy boxes
    // copy pts
    // syncthreads
    // change mask
    const int threadsPerBlock = 8 * 8;
    const int row_start = blockIdx.x;
    const int col_start = blockIdx.y;
    const int tx = threadIdx.x;
    const int row_size = min(M - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size = min(N - col_start * threadsPerBlock, threadsPerBlock);
    __shared__ scalar_t block_pts[64 * 3];
    __shared__ scalar_t block_boxes[64 * 7];
    const int dev_pt_idx = threadsPerBlock * col_start + tx;
    const int dev_box_idx = threadsPerBlock * row_start + tx;
    if (tx < col_size)
    {
        block_pts[tx * 3 + 0] = pts[dev_pt_idx * 3 + 0];
        block_pts[tx * 3 + 1] = pts[dev_pt_idx * 3 + 1];
        block_pts[tx * 3 + 2] = pts[dev_pt_idx * 3 + 2];
    }
    if (tx < row_size)
    {
        block_boxes[tx * 7 + 0] = boxes[dev_box_idx * 7 + 0];
        block_boxes[tx * 7 + 1] = boxes[dev_box_idx * 7 + 1];
        block_boxes[tx * 7 + 2] = boxes[dev_box_idx * 7 + 2];
        block_boxes[tx * 7 + 3] = boxes[dev_box_idx * 7 + 3];
        block_boxes[tx * 7 + 4] = boxes[dev_box_idx * 7 + 4];
        block_boxes[tx * 7 + 5] = boxes[dev_box_idx * 7 + 5];
        block_boxes[tx * 7 + 6] = boxes[dev_box_idx * 7 + 6];
    }
    __syncthreads();
    if (tx < row_size)
        for(int i=0; i<col_size; i++)
        {
            const int offset = row_start * threadsPerBlock * N + col_start * threadsPerBlock + tx * N + i;
            const scalar_t pt[] = {block_pts[i * 3], block_pts[i * 3 + 1], block_pts[i * 3 + 2]};
            const scalar_t box[] = {block_boxes[tx * 7], block_boxes[tx * 7 + 1],
                block_boxes[tx * 7 + 2], block_boxes[tx * 7 + 3], block_boxes[tx * 7 + 4],
                block_boxes[tx * 7 + 5], block_boxes[tx * 7 + 6]};
            masks[offset] = is_pt_in_box(pt, box);
        }
}

torch::Tensor crop_pts_3drot_cuda(torch::Tensor boxes, torch::Tensor pts)
{
    //boxes: [M, 7]
    //pts: [N, 3]
    //one thread proc one point with M times in for loop
    const int M = boxes.size(0);
    const int N = pts.size(0);
    auto boxes_dev = boxes.to(torch::kFloat32);
    auto pts_dev = pts.to(torch::kFloat32);
    auto options = torch::TensorOptions()
        .dtype(torch::kBool)
        .layout(torch::kStrided)
        .device(torch::kCUDA, 0);
    auto masks = torch::zeros({M, N}, options);
    if (M == 0 or N == 0)
        return masks;
    const int threadsPerBlock = 8 * 8;
    const dim3 blockspergrid(divup(M, threadsPerBlock),divup(N, threadsPerBlock));
    AT_DISPATCH_FLOATING_TYPES(boxes_dev.type(), "crop_pts_3drot_cuda", ([&]{
        crop_pts_3drot_cuda_kernel<scalar_t><<<blockspergrid, threadsPerBlock>>>(
        /*M=*/M,
        /*N=*/N,
        /*boxes=*/boxes_dev.data<scalar_t>(),
        /*pts=*/pts_dev.data<scalar_t>(),
        /*mask=*/masks.data<bool>()
        );
    }));
    return masks;
}