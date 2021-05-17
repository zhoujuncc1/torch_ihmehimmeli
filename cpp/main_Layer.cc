#include "functions.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
int main(int argc, char *argv[])
{
  char *weight_f;
  char *activation_f;
  char *A_f;
  char *B_f;
  char *W_f;
  char *causal_set_f;
  int m, n;
  double theta;
  int n_pulse;

  if (argc < 11)
  {
    printf("Not enough arguments\n");
    exit(1);
  }

  weight_f = argv[1];
  activation_f = argv[2];
  A_f = argv[3];
  B_f = argv[4];
  W_f = argv[5];
  causal_set_f = argv[6];
  m = atoi(argv[7]);
  n = atoi(argv[8]);
  theta = atof(argv[9]);
  n_pulse = atoi(argv[10]);

  VectorXd activations(m);
  VectorXd activations_next(n);
  VectorXXd weights(n);
  VectorXd A(n);
  VectorXd B(n);
  VectorXd W(n);
  VectorXXb causal_set(n);

  // activations.resize(m);
  // activations_next.resize(n);
  // weights.resize(n);
  // A.resize(n);
  // B.resize(n);
  // W.resize(n);
  // causal_set.resize(n);

  DecayParams decay_param;
  decay_param.set_decay_rate(DECAY_RATE);

  std::ifstream fin(activation_f, std::ios::binary);
  double f;
  for (int i = 0; i < m; ++i)
  {
    fin.read(reinterpret_cast<char *>(&f), sizeof(double));
    activations[i] = f;
  }
  fin.close();

  std::ifstream fin2(weight_f, std::ios::binary);


  weights.assign(n, VectorXd(m+n_pulse));
  causal_set.assign(n, VectorXb(m+n_pulse, 0));

  for (int i = 0; i < n; ++i)
  {
    //weights[i].resize(m);
    for (int j = 0; j < m+n_pulse; ++j)
    {
      fin2.read(reinterpret_cast<char *>(&f), sizeof(double));
      weights[i][j]=f;
    }
  }
  fin2.close();

  double max_v = *std::max_element(activations.begin(), activations.end());
  double min_v = *std::min_element(activations.begin(), activations.end());
  std::pair<double, double> input_range(min_v, max_v);
  VectorXd pulse = GeneratePulses(n_pulse, input_range);
  activations.insert(activations.end(), pulse.begin(), pulse.end());


  std::vector<size_t> sorted_indices = GetSortedIndices(activations);
  // Precompute exp(activations).
  VectorXd exp_activations = ExponentiateSortedValidSpikes(activations, sorted_indices, decay_param.rate());
  for (int i = 0; i < n; ++i)
  {
    //printf("%d\n", i);
    //printf("%.2f, %.2f\n", weights[i][0], activations[0]);
    //printf("%.2f, %.2f, %.2f, %.2f\n", A[i], B[i], W[i], causal_set[i][0]);
    activations_next[i] = ActivateNeuronAlpha(
        weights[i], activations, exp_activations,
        sorted_indices, theta, &(causal_set[i]),
        &(A[i]), &(B[i]), &(W[i]), decay_param);
  }
  activations_next.insert(activations_next.end(), pulse.begin(), pulse.end());

  std::ofstream fout(activation_f, std::ios::binary);
  for (int i = 0; i < n+n_pulse; ++i)
  {
    fout.write(reinterpret_cast<const char *>(&activations_next[i]), sizeof(double));
  }
  fout.close();
  fout.open(A_f, std::ios::binary);
  for (int i = 0; i < n; ++i)
  {
    fout.write(reinterpret_cast<const char *>(&A[i]), sizeof(double));
  }
  fout.close();

  fout.open(B_f, std::ios::binary);
  for (int i = 0; i < n; ++i)
  {
    fout.write(reinterpret_cast<const char *>(&B[i]), sizeof(double));
  }
  fout.close();

  fout.open(W_f, std::ios::binary);
  for (int i = 0; i < n; ++i)
  {
    fout.write(reinterpret_cast<const char *>(&W[i]), sizeof(double));
  }
  fout.close();

  fout.open(causal_set_f, std::ios::binary);

  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < m+n_pulse; ++j){
      fout.write(reinterpret_cast<const char *>(&causal_set[i][j]), sizeof(unsigned char));
    }
  }
  fout.close();
}