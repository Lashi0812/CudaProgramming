#include <torch/torch.h>
#include <iostream>

int main()
{
    auto options = torch::TensorOptions().device(torch::kCUDA,0);
    auto input = torch::rand({128,10},options);
    auto linear = torch::nn::Linear(torch::nn::LinearOptions(10,20).bias(false));
    linear->to(torch::kCUDA,0);
    std::cout << linear->weight.sizes();
    std::cout << linear(input).sizes();
}