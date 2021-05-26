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
  char *d_loss_f;
  char *d_w_f;
  char *d_x_f;
  int m, n;
  double theta;
  int n_pulse;
  bool forward_only = false;
  if (argc == 11)
  {
    forward_only = true;
  }
  else if (argc < 14)
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
  if (!forward_only)
  {
    d_loss_f = argv[11];
    d_w_f = argv[12];
    d_x_f = argv[13];
  }
  VectorXd activations(m);
  VectorXd activations_next(n);
  VectorXXd weights(n);
  VectorXd A(n);
  VectorXd B(n);
  VectorXd W(n);
  VectorXXb causal_set(n);

  VectorXXd grad_weights(n);
  VectorXd grad_activation(m+n_pulse);
  VectorXd loss(n+n_pulse);

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

  fin.open(weight_f, std::ios::binary);
  weights.assign(n, VectorXd(m + n_pulse));
  causal_set.assign(n, VectorXb(m + n_pulse, 0));
  for (int i = 0; i < n; ++i)
  {
    //weights[i].resize(m);
    for (int j = 0; j < m + n_pulse; ++j)
    {
      fin.read(reinterpret_cast<char *>(&f), sizeof(double));
      weights[i][j] = f;
    }
  }
  fin.close();
  if (!forward_only)
  {
    fin.open(d_loss_f, std::ios::binary);
    for (int i = 0; i < n+n_pulse; ++i)
    {
      fin.read(reinterpret_cast<char *>(&f), sizeof(double));
      loss[i] = f;
    }
    fin.close();
  }
  // Forward
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

  // Backward
  if (!forward_only)
  {
    grad_weights.assign(n, VectorXd(m + n_pulse, 0.0));
    grad_activation.assign(m + n_pulse, 0.0);
    // Update weights.
    for (int k = 0; k < n; ++k)
    {
      for (int j = 0; j < m + n_pulse; ++j)
      {
        // Update weight derivative between current layers.
        if (activations[k] == K_NO_SPIKE)
        {
          // no spike, negative grad
          grad_weights[k][j] += -PENALTY_NO_SPIKE;
        }
        else
        {
          double derivative;

          derivative = loss[k] *
                       WeightDerivativeAlpha(activations, causal_set, A, B,
                                             W, k, j, decay_param);

          grad_weights[k][j] +=
              ClipDerivative(derivative, CLIP_DERIVATIVE);
        }
        // Update activation derivative in presynaptic wrt postsynaptic layer.
        double derivative;

        derivative = loss[k] *
                     ActivationDerivativeAlpha(activations, weights, causal_set, A, B,
                                               W, k, j, decay_param);

        grad_activation[j] += ClipDerivative(derivative, CLIP_DERIVATIVE);
      }
    }
  }
  std::ofstream fout(activation_f, std::ios::binary);
  for (int i = 0; i < n + n_pulse; ++i)
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
    for (int j = 0; j < m + n_pulse; ++j)
    {
      fout.write(reinterpret_cast<const char *>(&causal_set[i][j]), sizeof(unsigned char));
    }
  }
  fout.close();

  if (!forward_only)
  {
    fout.open(d_w_f, std::ios::binary);
    for (int i = 0; i < n; ++i)
    {
      for (int j = 0; j < m + n_pulse; ++j)
      {
        fout.write(reinterpret_cast<char *>(&grad_weights[i][j]), sizeof(double));
      }
    }
    fout.close();

    fout.open(d_x_f, std::ios::binary);
    for (int i = 0; i < m + n_pulse; ++i)
    {
      fout.write(reinterpret_cast<const char *>(&grad_activation[i]), sizeof(double));
    }
    fout.close();
  }
}