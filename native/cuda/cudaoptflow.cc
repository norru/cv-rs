#include "cudaoptflow.h"
#include <opencv2/cudaoptflow.hpp>
#include "../utils.h"

extern "C" {

void cuda_calc_optical_flow_dtvl1(cv::Mat* from,
                                  cv::Mat* to,
                                  cv::Mat* out,
                                  double tau,
                                  double lambda,
                                  double theta,
                                  int nscales,
                                  int warps,
                                  double epsilon,
                                  int iterations,
                                  double scaleStep,
                                  double gamma,
                                  bool useInitialFlow) {

    auto optical_flow = cv::cuda::OpticalFlowDual_TVL1::create(tau,
                                                               lambda,
                                                               theta,
                                                               nscales,
                                                               warps,
                                                               epsilon,
                                                               iterations,
                                                               scaleStep,
                                                               gamma,
                                                               useInitialFlow);

    cv::cuda::GpuMat gpuFrom;
    cv::cuda::GpuMat gpuTo;
    cv::cuda::GpuMat gpuOut;

    gpuFrom.upload(*from);
    gpuTo.upload(*to);

    optical_flow->calc(gpuFrom, gpuTo, gpuOut);
    gpuOut.download(*out);
}

}
