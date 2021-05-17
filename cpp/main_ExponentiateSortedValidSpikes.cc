#include "functions.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char* argv[])
{
  VectorXd activations;
  for (int i = 1; i < argc; ++i){
    activations.push_back(atof(argv[i]));
  }
  std::vector<size_t> sorted_indices = GetSortedIndices(activations);
  // Precompute exp(activations).
  VectorXd exp_activations = ExponentiateSortedValidSpikes(activations, sorted_indices, DECAY_RATE);
  for (int i = 0; i < exp_activations.size(); ++i)
  {
    printf("%.10f ", exp_activations[i]);
  }
}