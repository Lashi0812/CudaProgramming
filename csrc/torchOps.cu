#include "ATen/ATen.h"

int main()
{
    at::TensorOptions options = at::TensorOptions().device(at::kCUDA,0).dtype(at::kFloat);
    at::Tensor input = at::arange(64,options);
    at::Tensor sum_out = input + 10;
    at::Tensor mean_out = input.mean();
    at::Tensor var_out = input.pow(2).mean();
}