/*
 * File Created: Wed Mar 04 2020

 * This is converted from the python code:
 * https://github.com/traveller59/second.pytorch/blob/master/second/core/non_max_suppression/
 * https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation
*/

#include <torch/extension.h>
#include <vector>

// CUDA declarations
torch::Tensor compute_intersect_2drot_cuda(
    torch::Tensor boxes,
    torch::Tensor query_boxes);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor compute_intersect_2drot(
    torch::Tensor boxes,
    torch::Tensor query_boxes)
{
  CHECK_INPUT(boxes);
  CHECK_INPUT(query_boxes);
  return compute_intersect_2drot_cuda(boxes, query_boxes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_intersect_2drot", &compute_intersect_2drot, "compute_intersect_2drot (CUDA)");
}