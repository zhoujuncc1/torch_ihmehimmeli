#include "functions.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char* argv[])
{
  VectorXd activations;
  VectorXd targets;
  for (int i = 1; i < argc-1; ++i){
    activations.push_back(atof(argv[i]));
    targets.push_back(0);
  }
  int target = atoi(argv[argc-1]);
  targets[target] = 1;
  // Precompute exp(activations).
  double loss = CrossEntropyLoss(activations, targets);
  printf("%.10f ", loss);

  VectorXd derivative = CrossEntropyLossDerivative(activations, targets);
  for (int k = 0; k < derivative.size(); ++k) {
    printf("%.10f ", derivative[k]);
  }

}