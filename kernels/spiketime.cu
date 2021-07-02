#include <torch/extension.h>
#include <vector>



#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

// C++ Python interface

// Modified from https://github.com/thomasluu/plog
template <class T>
__global__ void spiketime_cuda_kernel(T* output, T* input, T* spiketime, long* spiketime_index, T* kNoSpike, unsigned nNeuron, unsigned n_in)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= nNeuron)
    return;
  T* output_pt = output+index*n_in;
  T* input_pt = input+index*n_in;
  T v = *kNoSpike;
  int vi= n_in-1;
  for(unsigned i = 0; i < n_in-1; i++){
    if(output_pt[i] < *kNoSpike && output_pt[i] > input_pt[i] && output_pt[i] <= input_pt[i+1]){
      v = output_pt[i];
      vi=i;
      break;
    }
  }
  if (vi==n_in-1)
    v = output_pt[vi];
  spiketime[index] = v>0?v:*kNoSpike;
  spiketime_index[index] = vi;
}

template <class T>
void spiketime_cpp(T* output, T* input, T* spiketime, long* spiketime_index, T* kNoSpike, unsigned nNeuron, unsigned n_in)
{
  for(unsigned index = 0 ; index < nNeuron; index++){
    T* output_pt = output+index*n_in;
    T* input_pt = input+index*n_in;
    T v = *kNoSpike;
    int vi= n_in-1;
    for(unsigned i = 0; i < n_in-1; i++){
      if(output_pt[i] < *kNoSpike && output_pt[i] > input_pt[i] && output_pt[i] <= input_pt[i+1]){
        v = output_pt[i];
        vi=i;
        break;
      }
    }
    if (vi==n_in-1)
      v = output_pt[vi];
    spiketime[index] = v>0?v:*kNoSpike;
    spiketime_index[index] = vi;
  }
}

std::vector<torch::Tensor> spiketime(
	torch::Tensor output, torch::Tensor input, torch::Tensor kNoSpike)
{

	CHECK_CONTIGUOUS(output);
	CHECK_CONTIGUOUS(input);

	unsigned n_in = output.size(-1);
	unsigned nNeurons = 1;
	for(int i = 0; i < output.ndimension()-1; i++)
		nNeurons *= output.size(i);

  torch::Tensor spike_time = torch::ones({output.size(0), output.size(1),1}, output.options());
  torch::Tensor spike_time_index = torch::ones({output.size(0), output.size(1),1}, output.options().dtype(torch::kInt64));
  if(output.is_cuda()){
    cudaSetDevice(output.device().index());
    unsigned threads = 256;
    unsigned blocks  = ceil(1.0f * nNeurons / threads);
    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "spiketime_cuda", ([&] {
    spiketime_cuda_kernel<scalar_t><<<blocks, threads>>>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        spike_time.data_ptr<scalar_t>(),
        spike_time_index.data_ptr<long>(),
        kNoSpike.data_ptr<scalar_t>(), nNeurons, n_in);
  }));
  }
  else {
     AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "spiketime_cpp", ([&] {
    spiketime_cpp<scalar_t>(
        output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        spike_time.data_ptr<scalar_t>(),
        spike_time_index.data_ptr<long>(),
        kNoSpike.data_ptr<scalar_t>(), nNeurons, n_in);
  }));
  }
	return {spike_time, spike_time_index};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("spiketime", &spiketime, "Get spike time");

}