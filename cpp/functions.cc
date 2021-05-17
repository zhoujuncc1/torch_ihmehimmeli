#include <algorithm>
#include <numeric>
#include <cmath>

#include "functions.h"
#include "lambertw.h"

std::vector<size_t> GetSortedIndices(const VectorXd &activations)
{
  std::vector<size_t> sorted_indices(activations.size());
  iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&activations](size_t i, size_t j) {
              return activations[i] < activations[j];
            });
  return sorted_indices;
}

VectorXd ExponentiateSortedValidSpikes(
    const VectorXd &activations, const std::vector<size_t> &sorted_indices,
    const double decay_rate)
{
  VectorXd exp_activations(activations.size(), K_NO_SPIKE);
  for (int i = 0; i < sorted_indices.size() &&
                  activations[sorted_indices[i]] < K_NO_SPIKE;
       ++i)
  {
    exp_activations[sorted_indices[i]] =
        exp(decay_rate * activations[sorted_indices[i]]);
  }
  return exp_activations;
}

VectorXd GeneratePulses(const int n_pulses,
                        const std::pair<double, double> input_range)
{
  VectorXd pulses_per_layer(n_pulses);
  const double pulse_spacing =
      (input_range.second - input_range.first) / (pulses_per_layer.size() + 1);
  for (int i = 0; i < pulses_per_layer.size(); ++i)
  {
    pulses_per_layer[i] = input_range.first + pulse_spacing * (i + 1);
  }
  return pulses_per_layer;
}


double ComputeCrossEntropyLossWithPenalty(
    const VectorXd& outputs, const VectorXd& targets,
    const VectorXd& spike_times) {
  double total_loss = 0.0;
  const double kEps = 1e-8;
  double penalty_output_spike_time_ = 0.0;
  for (int i = 0; i < targets.size(); ++i) {
    total_loss -= targets[i] * log(outputs[i] + kEps);
  }

  for (int i = 0; i < spike_times.size(); ++i) {
    total_loss += penalty_output_spike_time_ * spike_times[i] * spike_times[i];
  }

  return total_loss;
}

double CrossEntropyLoss(const VectorXd &activations, VectorXd &targets)
{
  VectorXd exp_outputs(activations.size());  // softmaxed outputs
  double min_output = *min_element(activations.begin(), activations.end());
  for (int i = 0; i < activations.size(); ++i)
    exp_outputs[i] = exp(-activations[i] + min_output);
  double exp_sum = accumulate(exp_outputs.begin(), exp_outputs.end(), 0.0);
  for (int i = 0; i < exp_outputs.size(); ++i)
    exp_outputs[i] /= exp_sum;
  return ComputeCrossEntropyLossWithPenalty(exp_outputs, targets, activations);
}

VectorXd CrossEntropyLossDerivative(const VectorXd &activations, VectorXd &targets)
{
  VectorXd exp_outputs(activations.size());  // softmaxed outputs
  double min_output = *min_element(activations.begin(), activations.end());
  for (int i = 0; i < activations.size(); ++i)
    exp_outputs[i] = exp(-activations[i] + min_output);
  double exp_sum = accumulate(exp_outputs.begin(), exp_outputs.end(), 0.0);
  for (int i = 0; i < exp_outputs.size(); ++i)
    exp_outputs[i] /= exp_sum;
  VectorXd d_activations_pre(activations.size());

  // Accumulate derivative of cross-entropy loss at each output node.
  for (int k = 0; k < activations.size(); ++k) {
    d_activations_pre[k] = -(exp_outputs[k] - targets[k]);
  }
  return d_activations_pre;
  
  }
