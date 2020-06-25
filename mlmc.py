""" APMLMC - An implementation of multilevel Monte Carlo for AP schemes
    Copyright (C) 2020 Emil Loevbak (emil.loevbak@kuleuven.be)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""Contains the high level methods for running Multi-level Monte Carlo simulations in an abstract setting."""




import random
import numpy as np
import math
import pickle
import time

def MLMC_sim(levels, fcn, cost_fcn, alpha, epsilon, default_samples=50, max_work=5000, seed=None, convergence_fcn=None, checkpoint=None):
    """A function for Multilevel Monte Carlo simulation.
    Variable types:
        levels: A list of integer values indicating the initial target number of samples at each level.
        fcn: A function with an interface of the form 'sample_data = fcn(new_samples, level)' with
             new_samples: An integer number of samples to generate.
             level: An integer indicating the level at  which sampling is taking place, 0 being the roughest level.
             sample_data: A tuple of format (sum_samples, sum_squares, sum_sample_diffs, sum_squared_diffs) representing
                          respectively the sums of QOI's of the samples, the sums of squares of the QOI's the sums of the
                          differences between correlated samples and their squares, in Numpy arrays. These returned sums do not
                          necessarily need to be numbers but support addition/multiplication with their own kind and numerical
                          values, division by greater than / equality / less than comparison with numbers.
        cost_fcn: A function with as interface 'cost = cost_fcn(level)' that returns a numerical value for the cost that
                  computing a sample at the given level has.
        epsilon: A numerical value indicating the user specified RMS accuracy.
        default_samples: The default number of samples with which to initialize a new level.
        max_work: The maximal number of samples multiplied by cost that will be done in the simulation.
        seed: A seed for the random number generator. If no seed is given then the RNG is not seeded, making each run different.
    """

    if checkpoint is None:
        # Setup seed, default of None does nothing
        random.seed(seed)
        data = MLMC_data(levels, cost_fcn, alpha, epsilon, max_work)

        print("Sampling for level 0")
        data.generate_samples(fcn)
        data.update_opt_samples()
    else:
        print("Loading checkpoint...")
        data = load_checkpoint(checkpoint)

    while(not converged(data, convergence_fcn)):
        print("Adding level number " + str(data.num_levels))
        data.add_level(default_samples, cost_fcn)
        print("Updating samples to the distribution: " + str(data._opt_samples))
        data.generate_samples(fcn)
        if not data.update_opt_samples():
            print("Computation limit reached!")
            break
        save_checkpoint(data)
        print(data.mean())
        print(data.diff_variance())
    return data


def converged(data, convergence_fcn):
    """Tests the provided simulation data against the weak convergence criterion in Mike Giles, Acta Numerica, unless another 
       function is given.
    """
    if convergence_fcn is None:
        # First check if we have done too much work
        if data.total_work >= data.max_work or min(data._opt_samples) < 0:
            print("Computation limit reached!")
            return True

        # Otherwise the convergence criterion will not make sense
        if data.num_levels <= 3:
            return False

        errors = data.diff_mean()
        weak_error = np.amax(abs(np.concatenate(errors[-3:])))
        weak_error_extrapolated = weak_error / (2**data.alpha - 1)
        if weak_error_extrapolated < data.epsilon / np.sqrt(2):
            print("Weak convergence achieved!")
            return True

        return False
    else:
        return convergence_fcn(data)


def save_checkpoint(data):
    """Save data to a timestamped checkpoint file."""
    timestamp = str(round(time.time() * 1000))
    with open(timestamp + ".chk", 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)


def load_checkpoint(checkpoint):
    """Read checkpoint from file and return it."""
    with open(checkpoint, 'rb') as input:
        data = pickle.load(input)
    return data


class MLMC_data:

    """Contains the relevant data for the levels of a Multilevel Monte Carlo simulation.
    Member variables:
        num_levels: The number of levels in the simulation.
        alpha: A numerical value indicating the expected rate of weak convergence
        epsilon: A numerical value indicating the user specified RMS accuracy.
        samples: A list of integers listing the number of computed samples at each level.
        max_work: The maximal number of samples multiplied by cost that may be done.
        total_work: The amount of work already done.
        _opt_samples: A list of integers listing the optimal number of samples estimated for each level.
        _cost: A list of costs for a sample at each level.
        _sum_samples: A list of the sums of the samples generated at each level (P_L).
        _sum_squares: A list of the sums of squares of the samples generated at each level.
        _diff_samples: A list of the differences of samples generated in correlated simulations at each level (Y_L).
        _diff_squares: Squares of the differences mentioned in the line above.
    For type information of the last four see the documentation of MLMC_sim.
    """

    def __init__(self, levels, cost_fcn, alpha, epsilon, max_work):
        self.num_levels = len(levels)
        self.alpha = alpha
        self.epsilon = epsilon
        self.samples = [0]*self.num_levels
        self.max_work = max_work
        self.total_work = 0
        self._opt_samples = levels
        self._cost = [cost_fcn(i) for i in range(0, self.num_levels)]
        self._sum_samples = [0]*self.num_levels
        self._sum_squares = [0]*self.num_levels
        self._diff_samples = [0]*self.num_levels
        self._diff_squares = [0]*self.num_levels

    def mean(self, level=None):
        """Returns the means of the sampled QOI's ath the requested level.
        If no level is specified then all levels are returned in array format. Expects an integer argument.
        """

        if level is None:
            return [self.mean(i) for i in range(0, self.num_levels)]
        else:
            if self.samples[level] > 0:
                return self._sum_samples[level]/self.samples[level]
            else:
                return 0

    def variance(self, level=None, only_max=True):
        """Returns the maximal variance of the sampled QOI's at the requested level.
        If no level is specified then all levels are returned in array format. Expects an integer argument.
        """

        if level is None:
            return [self.variance(i, only_max) for i in range(0, self.num_levels)]
        else:
            if self.samples[level] > 0:
                if(only_max):
                    return np.max(self._sum_squares[level]/self.samples[level] - self.mean(level)*self.mean(level))
                else:
                    return self._sum_squares[level]/self.samples[level] - self.mean(level)*self.mean(level)
            else:
                return 0

    def diff_mean(self, level=None):
        """Returns the means of the differences of the sampled QOI's at the requested level."""

        if level is None:
            return [self.diff_mean(i) for i in range(0, self.num_levels)]
        else:
            if self.samples[level] > 0:
                return self._diff_samples[level]/self.samples[level]
            else:
                return 0

    def telescopic_sum(self, level=None):
        """Returns the sum of the means of the differences of the sampled QOI's up to the requested level."""

        if level is None:
            return [self.diff_mean(i) for i in range(0, self.num_levels)]
        else:
            telescopic_sum_result = 0
            for i in range(min(self.num_levels, level+1)):
                if self.samples[i] > 0:
                    telescopic_sum_result += self._diff_samples[i] / \
                        self.samples[i]
            return telescopic_sum_result

    def diff_variance(self, level=None, only_max=True):
        """Returns the maximal variance of the difference of sampled QOI's at the requested level.
        If no level is specified then all levels are returned in array format. Expects an integer argument.
        """

        if level is None:
            return [self.diff_variance(i, only_max) for i in range(0, self.num_levels)]
        else:
            if self.samples[level] > 0:
                variances = self._diff_squares[level]/self.samples[level] - \
                    self.diff_mean(level)*self.diff_mean(level)
                if(only_max):
                    return np.max(variances)
                else:
                    return variances
            else:
                return 0

    def update_opt_samples(self, level=None):
        """Performes an update of the optimal number of samples at a given level.
        If no level is given then all levels are updated. Expects an integer argument.
        """
        if (level is None):
            for i in range(0, self.num_levels):
                if not self.update_opt_samples(i):
                    return False
        else:
            sum_prod = sum([np.sqrt(self.diff_variance(i)*self._cost[i])
                            for i in range(0, self.num_levels)])
            optimal_samples = 2 * \
                self.epsilon**(-2) * np.sqrt(self.diff_variance(level) /
                                             self._cost[level]) * sum_prod
            self._opt_samples[level] = int(np.ceil(optimal_samples))

            oversample = int(math.ceil((sum(np.multiply(
                self._opt_samples, self._cost)) - self.max_work) / self._cost[level]))
            if oversample >= 0:
                self._opt_samples[level] -= oversample
                return False
        return True

    def add_level(self, num_samples, cost_fcn):
        """Adds a new level to the data structure with the given initial number of expected samples.
        Expects an integer argument for num_samples and a function argument for cost_fcn as specified in the documentation for
        MLMC_sim.
        """

        self.num_levels += 1
        self.samples += [0]
        self._opt_samples += [num_samples]
        self._cost += [cost_fcn(self.num_levels - 1)]
        self._sum_samples += [0]
        self._sum_squares += [0]
        self._diff_samples += [0]
        self._diff_squares += [0]

    def generate_samples(self, fcn, level=None):
        """Increases the number of samples at the given level so that this is at least as many as the current optimal value.
        If no level is specified then all levels are updated.
        Variable types:
            fcn: A function which matches the specification given for the mlmc function.
            level: An integer value.
        """

        if (level is None):
            for i in range(0, self.num_levels):
                self.generate_samples(fcn, i)
        else:
            # Number of new samples needed
            new_samples = self._opt_samples[level] - self.samples[level]
            oversamples = sum(np.multiply(self.samples, self._cost)
                              ) + new_samples*self._cost[level]
            if oversamples > self.max_work:
                new_samples -= int(math.ceil(oversamples / self._cost[level]))
            if (new_samples > 0):
                sample_data = fcn(new_samples, level)
                self._sum_samples[level] += sample_data[0]
                self._sum_squares[level] += sample_data[1]
                self._diff_samples[level] += sample_data[2]
                self._diff_squares[level] += sample_data[3]
                self.samples[level] += new_samples
                self.total_work += new_samples*self._cost[level]
