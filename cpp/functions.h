#ifndef TEST_FUNCTIONS_H
#define TEST_FUNCTIONS_H
#include <vector>
#include <cmath>
#include <utility>

typedef std::vector<double> VectorXd;
typedef std::vector<std::vector<double>> VectorXXd;
typedef std::vector<unsigned char> VectorXb;
typedef std::vector<std::vector<unsigned char>> VectorXXb;

#define K_NO_SPIKE 1000
#define DECAY_RATE 0.18176949150701854
#define PENALTY_NO_SPIKE 48.3748
#define CLIP_DERIVATIVE 539.7


static double ClipDerivative(double val, double limit) {
  if (limit == 0.0) return val;
  if (val < -limit) val = -limit;
  if (val > limit) val = limit;
  return val;
}

class DecayParams
{
public:
  // The decay rate K of the potential function: f(t) = t * exp(-K * t)
  double rate() const { return rate_; }

  // Inverse of the decay rate (1 / decay_rate). Precomputed for speed.
  double rate_inverse() const { return rate_inverse_; }

  // Updates the decay rate and the minimum significant weight.
  void set_decay_rate(const double decay_rate)
  {
    rate_ = decay_rate;
    rate_inverse_ = 1.0 / rate_; // allow zero decay rate for experiments
    dual_exponential_scale_ = 4 * rate_inverse_ / M_E;
  }

  // Factor to multiply the dual exponential by to make it have the same maximum
  // as the alpha function.
  double dual_exponential_scale() const { return dual_exponential_scale_; }

private:
  double rate_ = 1.0;
  double rate_inverse_ = 1.0;
  double dual_exponential_scale_ = 4 / M_E;
};

std::vector<size_t> GetSortedIndices(const VectorXd &activations);
VectorXd ExponentiateSortedValidSpikes(const VectorXd &activations, const std::vector<size_t> &sorted_indices,
                                       const double decay_rate);
double ActivateNeuronAlpha(const VectorXd &weights, const VectorXd &inputs, const VectorXd &exp_inputs,
                           const std::vector<size_t> &sorted_ind, double fire_threshold,
                           VectorXb *causal_set, double *A, double *B, double *W,
                           const DecayParams decay_params);
VectorXd GeneratePulses(const int n_pulses, const std::pair<double, double> input_range);
double CrossEntropyLoss(const VectorXd &activations, VectorXd &targets);
double ComputeCrossEntropyLossWithPenalty(
    const VectorXd& outputs, const VectorXd& targets,
    const VectorXd& spike_times);
VectorXd CrossEntropyLossDerivative(const VectorXd &activations, VectorXd &targets);
double WeightDerivativeAlpha(const VectorXd& activations,
                                        const VectorXXb& causal_sets,
                                        const VectorXd& a, const VectorXd& b,
                                        const VectorXd& w, int post,
                                        int pre, DecayParams decay_params);

double ActivationDerivativeAlpha(const VectorXd& activations, const VectorXXd& weights,
                                            const VectorXXb& causal_sets,
                                            const VectorXd& a,
                                            const VectorXd& b,
                                            const VectorXd& w,
                                            int post, int pre, DecayParams decay_params);
#endif