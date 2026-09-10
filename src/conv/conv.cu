
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <mma.h>

#include <string>
#include <vector>

#include "conv.h"


DT_conv2d::DT_conv2d(int C, int OC, int ks, int stride, int padding)
    : C(C), OC(OC), ks(ks), stride(stride), padding(padding) {
    d_workspace=nullptr;
    d_workspace_w_back=nullptr;
    d_workspace_y_back=nullptr;
    workspace_size=0;
    workspace_size_w_back=0;
    workspace_size_y_back=0;
}

extern "C" DT_conv2d *conv2d_Create(Scope_Struct *ctx, int M, int N, int ks, int stride, int padding) {
    DT_conv2d *conv= new DT_conv2d(M,N,ks,stride,padding);

    return conv;
}


extern "C" void *conv2d_forward(Scope_Struct *ctx, DT_conv2d *conv, void *x, void *weight, int b, int h, int w) {

    float *ret = conv->Forward(*(float**)x, *(float**)weight, b, h, w);

    void **ptr = (void**)allocate_pool(ctx, 8, data_name_to_type()["float_pp"]);
    *ptr = ret;
    return ptr;
}

void DT_conv2d::SetDescriptors(int B, int H, int W) {
  this->B = B;
  this->H = H;
  this->W = W;


  out_H = std::floor((H - ks + 2 * padding) / stride) + 1;
  out_W = std::floor((W - ks + 2 * padding) / stride) + 1;
  // std::cout << "Out H: " << out_H << " out W: " << out_W << "\n";

  checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, C, H, W));
  
  // Initialize filter descriptor
  cudnnFilterDescriptor_t filter_desc;
  checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
  checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OC, C, ks, ks));
  this->filter_desc = filter_desc;

  // Initialize convolution descriptor
  cudnnConvolutionDescriptor_t conv_desc;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
                                           CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
  this->conv_desc = conv_desc;

  // Initialize output tensor descriptor
  cudnnTensorDescriptor_t output_desc;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, B, OC, out_H, out_W));
  this->output_desc = output_desc;

  
  int requested_algo_count;
  int algo_count;




  checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        requested_algo_count,
        &algo_count,
        perf_results.data()
  ));

  this->fwd_algo = perf_results.front().algo;


  
  if (d_workspace!=nullptr)
      cudaFree(d_workspace);
    // move_to_pool_pow2(0, workspace_size, d_workspace, "d workspace");

  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        fwd_algo,
        &workspace_size
  ));
  // d_workspace = get_from_pool_pow2(0, workspace_size, "d workspace");
  cudaMalloc(&d_workspace, workspace_size);
  
  




  // Backward to input
  checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_back_y(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn,
        filter_desc,
        output_desc,
        conv_desc,
        input_desc,
        requested_algo_count,
        &algo_count,
        perf_results_back_y.data()
  ));

  y_bwd_algo = perf_results_back_y.front().algo;

  
  if(d_workspace_y_back!=nullptr)
      cudaFree(d_workspace_y_back);
    // move_to_pool_pow2(0, workspace_size_y_back, d_workspace_y_back, "d workspace y back");

  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn,
        filter_desc,
        output_desc,
        conv_desc,
        input_desc,
        y_bwd_algo,
        &workspace_size_y_back
  ));
  cudaMalloc(&d_workspace_y_back, workspace_size_y_back);
  // d_workspace_y_back = get_from_pool_pow2(0, workspace_size_y_back, "d workspace y back");
  




  // Backward to weight
  checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn, &requested_algo_count));
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_back_w(requested_algo_count);
  checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnn,
        input_desc,
        output_desc,
        conv_desc,
        filter_desc,
        requested_algo_count,
        &algo_count,
        perf_results_back_w.data()
  ));

  w_bwd_algo = perf_results_back_w.front().algo;

  
  
  if (d_workspace_w_back!=nullptr)
      cudaFree(d_workspace_w_back);
    // move_to_pool_pow2(0, workspace_size_w_back, d_workspace_w_back, "conv d workspace w back");

  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn,
        input_desc,
        output_desc,
        conv_desc,
        filter_desc,
        w_bwd_algo,
        &workspace_size_w_back
  ));
  cudaMalloc(&d_workspace_w_back, workspace_size_w_back);
  // d_workspace_w_back = get_from_pool_pow2(0, workspace_size_w_back, "conv d workspace w back");
}







float *DT_conv2d::Forward(float *x, float *weight, int B, int H, int W) {
  if (H != this->H || W != this->W || B != this->B)
    this->SetDescriptors(B, H, W);
  
  
  float *d_output = tarena.Allocate<float>(B*out_H*out_W*OC);

  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
 
  // std::cout << "now forward" << "\n";
  // std::cout << "workspace size " << workspace_size << "\n";

  checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &one,
        input_desc,
        x,
        filter_desc,
        weight,
        conv_desc,
        fwd_algo,
        d_workspace,
        workspace_size,
        &zero,
        output_desc,
        d_output
    ));


  return d_output;
}

//void Conv2dCPP::InitFilters() {
//  std::vector<float> h_filter;
//  float *filter;
//  for (std::size_t idx = 0; idx < C * OC; ++idx) {

//    if (Init=="xavu_relu")
//      filter = make_xavier_uniform_float_relu(ks*ks, ks*ks*C, ks*ks*OC);
//    if (Init == "xavu_tanh")
//      filter = make_xavier_uniform_float_tanh(ks*ks, ks*ks*C, ks*ks*OC);
//    if (Init=="he_normal_relu")
//      filter = make_he_normal_float_relu(ks*ks, ks*ks*C);
//    if (Init == "init_gpt")
//      filter = make_gpt_init(ks*ks);
//    if (Init=="xavu")
//      filter = make_xavier_uniform_float(ks*ks, ks*ks*C, ks*ks*OC);
//    if (Init=="zeros")
//      filter = make_zeros_float(ks*ks);
//    if (Init=="ones")
//      filter = make_ones_float(ks*ks);
//    if (Init=="randu")
//      filter = make_random_float_uniform(ks*ks);


//    for (int i=0; i < ks*ks; i++)
//      h_filter.emplace_back(filter[i]);

//    delete[] filter;
//    //for (const auto& val : filter) 
//    //  h_filter.emplace_back(val);
//  }
    
//  float* d_filter = nullptr;
//  const std::size_t filter_size = h_filter.size();
//  cudaCheck(cudaMalloc(&d_filter, round_to_nearest_pow2(filter_size) * sizeof(float)));

//  cudaCheck(cudaMemcpy(d_filter, h_filter.data(), filter_size * sizeof(float), cudaMemcpyDefault));
//  this->d_filter = d_filter;
  


//  std::vector<int> kernel_dims = {OC, C, ks, ks}; 

//  Filter_Tensor = createTensor(d_filter, kernel_dims, DimsProd(kernel_dims), true, Name+"W");
//  Filter_Tensor->SetIsWeight();
//}





extern "C" int conv2d_backward(Scope_Struct *ctx, DT_conv2d *conv, void *x, void *w, void *dx, void *dw, void *dy, bool x_has_grad) {
    conv->Backward(x, w, dx, dw, dy, x_has_grad);
    return 0;
}

void DT_conv2d::Backward(void *x, void *w, void *dx, void *dw, void *dy, bool x_has_grad) {
  // std::cout << "\nConv2d Backward with H: " << H << " W: " << W << "\n";


  constexpr float one = 1.0f;
  constexpr float zero = 0.0f;
  

  
  // Backward to input
  if (x_has_grad)
      checkCUDNN(cudnnConvolutionBackwardData(
        cudnn,
        &one,
        filter_desc, // input tensor descriptor
        *(float**)w,
        output_desc, // output grad tensor descriptor
        *(float**)dy,
        conv_desc, // convolution descriptor
        y_bwd_algo, //Obtained with getConvolutionBackwardDataAlgorithm
        d_workspace_y_back, 
        workspace_size_y_back, //Obtained with getConvolutionBackwardDataWorkspaceSize
        &zero,
        input_desc, // filter descriptor
        *(float**)dx
      ));


  // if (!x_has_grad)
      // return;
  

  // Backward to weight
  checkCUDNN(cudnnConvolutionBackwardFilter(
    cudnn,
    &one,
    input_desc, // input tensor descriptor
    *(float**)x,
    output_desc, // output grad tensor descriptor
    *(float**)dy,
    conv_desc, // convolution descriptor
    w_bwd_algo, //Obtained with getConvolutionBackwardFilterAlgorithm
    d_workspace_w_back, 
    workspace_size_w_back, //Obtained with getConvolutionBackwardFilterWorkspaceSize
    &one,
    filter_desc, // filter descriptor
    *(float**)dw
  ));

}
