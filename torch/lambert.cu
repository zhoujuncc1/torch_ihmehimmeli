#include <torch/extension.h>
#include <vector>



#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

// C++ Python interface

// Modified from https://github.com/thomasluu/plog
template <class T>
__global__ void plog(T* input, T* output, unsigned size)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index>=size)
    return;
  
  if (input[index] == 0.0) {
    output[index]=0.0;
    return;
  }

  x = input[index];
  T w0, w1;
  if (x > 0.0) {
    w0 = log(1.2 * x / log(2.4 * x / log1p(2.4 * x)));
  } else {
    T v = 1.4142135623730950488 * sqrt(1 + 2.7182818284590452354 * x);
    T N2 = 10.242640687119285146 + 1.9797586132081854940 * v;
    T N1 = 0.29289321881345247560 * (1.4142135623730950488 + N2);
    w0 = -1.0 + v * (N2 + v) / (N2 + v + N1 * v);
  }

  while (true) {
    T e = exp(w0);
    T f = w0 * e - x;
    w1 = w0 + ((f+f) * (1.0 + w0)) / (f * (2.0 + w0) - (e+e) * (1.0 + w0) * (1.0 + w0));
    if (fabs(w0 / w1 - 1.0) < 1.4901161193847656e-8) {
      break;
    }
    w0 = w1;
  }
  output[index]=w1
  return;
}


torch::Tensor lambertCuda(
	torch::Tensor input)
{
	CHECK_INPUT(input);

	auto output = torch::empty_like(input);

	cudaSetDevice(input.device().index());

	unsigned size = 1;
	for(int i = 0; i < input.ndimension(); i++)
		size *= input.size(i);
	if(input.dtype() == torch::kFloat32)
		plog<float>(input.data<float>(), output.data<float>(), size);
	else
		plog<double>(input.data<double>(), output.data<double>(), size);
	return output;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("lambertw", &lambertCuda, "Get lambert (CUDA)");

}