#ifndef CV_RS_CUDAOPTFLOW_H
#define CV_RS_CUDAOPTFLOW_H

#include "../common.h"
#include <opencv2/core.hpp>

extern "C" {

void cuda_calc_optical_flow_dtvl1(cv::Mat* from,
                                  cv::Mat* to,
                                  cv::Mat* out,
                                  double tau = 0.25,
                                  double lambda = 0.15,
                                  double theta = 0.3,
                                  int nscales = 5,
                                  int warps = 5,
                                  double epsilon = 0.01,
                                  int iterations = 30,
                                  double scaleStep = 0.8,
                                  double gamma = 0.0,
                                  bool useInitialFlow = false);

}
#endif
