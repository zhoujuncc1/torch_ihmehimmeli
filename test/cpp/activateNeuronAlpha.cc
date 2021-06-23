#include "functions.h"
#include <cmath>
#include "lambertw.h"
double ActivateNeuronAlpha(
    const VectorXd& weights, const VectorXd& inputs, const VectorXd& exp_inputs,
    const std::vector<size_t>& sorted_ind, double fire_threshold,
    VectorXb* causal_set, double* A, double* B, double* W,
    const DecayParams decay_params) {
  // Initialise causal variables.
  double spike_time = K_NO_SPIKE;
  *A = 0.0;
  *B = 0.0;


  // Process incoming spikes one by one.
  for (const size_t spike_ind : sorted_ind) {
    // Check if neuron spiked before incoming spike (or incoming is kNoSpike).
    if (spike_time <= inputs[spike_ind]) {
      return spike_time;
    }
    // Reset spike time, in case an inhibitory input cancels a potential spike.
    spike_time = K_NO_SPIKE;

    // Update causal set because presynaptic input precedes postsynaptic spike.
    (*causal_set)[spike_ind] = 1;

    const double w_exp_z = weights[spike_ind] * exp_inputs[spike_ind];
    *A += w_exp_z;
    *B += w_exp_z * inputs[spike_ind];

    // The value of the first derivative of the activation function in the
    // intersection point with the fire threshold is given by *A multiplied by a
    // never-negative value. Thus, if *A is negative the intersection will be in
    // a decreasing-potential area, and thus not a spike.
    if (*A < 0) continue;

    const double b_over_a = *B / *A;

    // Compute Lambert W argument for solving the threshold crossing.
    const double lambert_arg = -decay_params.rate() * fire_threshold / *A *
                               exp(decay_params.rate() * b_over_a);
    // Minimum argument for the main branch of the Lambert W function.
    constexpr double kMinLambertArg = -1.0 / M_E;
    // Maximum argument for which gsl_sf_lambert_W0 produces a valid result.
    constexpr double kMaxLambertArg = 1.7976131e+308;
    if (lambert_arg >= kMinLambertArg && lambert_arg <= kMaxLambertArg) {
      double val;
      LambertW0(lambert_arg, &val);

      *W = val;
      spike_time = b_over_a - *W * decay_params.rate_inverse();

      // For inhibitory weights, this might be a false alarm.
      // This is not the same as spike_time < inputs[spike_ind]: it is also true
      // for NaNs.
      if (!(spike_time >= inputs[spike_ind])) spike_time = K_NO_SPIKE;
    }
  }
  // If we get here, either there is no spike, in which case
  // all presynaptic neurons are to blame, or there is eventually
  // a spike caused by all presynaptic inputs.
  if (spike_time == K_NO_SPIKE) {
    causal_set->assign(causal_set->size(), true);
  }
  return spike_time;
}

