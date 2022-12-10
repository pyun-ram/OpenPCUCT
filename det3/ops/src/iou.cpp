/*
 * File Created: Mon Mar 02 2020

*/

#include <torch/extension.h>

torch::Tensor compute_intersect_2d(torch::Tensor box, torch::Tensor others) {
    auto box_x = box[0]; auto box_y = box[1];
    auto box_l = box[2]; auto box_w = box[3];
    auto box_xmin = box_x - box_l/2.0;
    auto box_xmax = box_x + box_l/2.0;
    auto box_ymin = box_y - box_w/2.0;
    auto box_ymax = box_y + box_w/2.0;
    auto others_x = others.narrow(1, 0, 1);
    auto others_y = others.narrow(1, 1, 1);
    auto others_l = others.narrow(1, 2, 1);
    auto others_w = others.narrow(1, 3, 1);
    auto others_xmin = others_x - others_l/2.0;
    auto others_ymin = others_y - others_w/2.0;
    auto others_xmax = others_x + others_l/2.0;
    auto others_ymax = others_y + others_w/2.0;
    auto xx1 = torch::max(box_xmin, others_xmin);
    auto yy1 = torch::max(box_ymin, others_ymin);
    auto xx2 = torch::min(box_xmax, others_xmax);
    auto yy2 = torch::min(box_ymax, others_ymax);
    auto w = torch::clamp(xx2 - xx1, 0.0, std::numeric_limits<double>::infinity());
    auto h = torch::clamp(yy2 - yy1, 0.0, std::numeric_limits<double>::infinity());
    return (w * h).flatten();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_intersect_2d", &compute_intersect_2d, "compute_intersect_2d");
}