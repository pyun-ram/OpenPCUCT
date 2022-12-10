/*
 * File Created: Thu Mar 05 2020

*/
#include <torch/extension.h>
#include <vector>

// CUDA declarations
torch::Tensor crop_pts_3drot_cuda(
    torch::Tensor boxes,
    torch::Tensor pts);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor crop_pts_3drot(
    torch::Tensor boxes,
    torch::Tensor pts)
{
  CHECK_INPUT(boxes);
  CHECK_INPUT(pts);
  return crop_pts_3drot_cuda(boxes, pts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("crop_pts_3drot", &crop_pts_3drot, "crop_pts_3drot (CUDA)");
}