/*
 * File Created: Mon Mar 02 2020

*/

#include <torch/extension.h>

torch::Tensor get_corner_box_2drot(torch::Tensor boxes){
    // auto device = boxes.device();
    // auto dtype = boxes.dtype();
    // auto options = torch::TensorOptions().device(device).dtype(dtype);
    // auto M = boxes.size(0);
    auto l = boxes.narrow(1, 2, 1);
    auto w = boxes.narrow(1, 3, 1);
    auto theta = boxes.narrow(1, 4, 1);
    auto p1 = torch::stack({-l/2.0, w/2.0}, /*dim=*/1);
    auto p2 = torch::stack({ l/2.0, w/2.0}, /*dim=*/1);
    auto p3 = torch::stack({ l/2.0,-w/2.0}, /*dim=*/1);
    auto p4 = torch::stack({-l/2.0,-w/2.0}, /*dim=*/1);
    auto pts = torch::stack({p1, p2, p3, p4}, /*dim=*/0)
        .transpose(0, 1).squeeze();
    auto tr_vecs = boxes.narrow(1, 0, 2).unsqueeze(1).repeat({1, 4, 1});
    auto cry = torch::cos(theta); auto sry = torch::sin(theta);
    auto R_r0 = torch::stack({cry, -sry}, /*dim=*/1).squeeze();
    auto R_r1 = torch::stack({sry,  cry}, /*dim=*/1).squeeze();
    auto R = torch::stack({R_r0,  R_r1}, /*dim=*/1);
    pts = torch::bmm(R, pts.transpose(-2, -1)).transpose(-2, -1);
    pts += tr_vecs;
    return pts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_corner_box_2drot", &get_corner_box_2drot, "get_corner_box_2drot");
}