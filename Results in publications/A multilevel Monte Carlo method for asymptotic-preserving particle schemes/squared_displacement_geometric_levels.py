"""
 This file is part of the simulation code accompanying the article
 "A multilevel Monte Carlo method for asymptotic-preserving particle schemes".
 Copyright (C) 2020 Emil Loevbak emil.loevbak@kuleuven.be

 This simulation code is free software: you can redistribute it and/or
 modify it under the terms of the GNU General Public License as published
 by the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This simulation code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this simulation code.  If not, see <https://www.gnu.org/licenses/>.
"""

"""Script used for generating Tables 1-3 of the paper.
This script takes three arguments in the general case:
    1) The value of epsilon.
    2) The value of E (the requested RMSE).
    3) The initial number of samples per level.
If restarting a simulation from a checkpoint file, set epsilon to 0 and provide a 4th argument which is the path to the checkpoint file.
"""

import numpy as np
from random import random, gauss
from mlmc import MLMC_sim
import genericParticle
from GoldsteinTaylor import GTparticle
import multiprocessing as mp
import math
import sys
sys.path.append("../../")


def sample_wrapper(samples, level, particles_per_sample, delta_t0, epsilon, stop_times, bound_L, bound_R, bound_L_type, bound_R_type, dist_fcn, scale_factor, threads):

    print("Level")
    print(level)
    (delta_t_f, scale_factor) = level_timestep(
        level, epsilon, delta_t0, scale_factor)
    print(delta_t_f)

    output = mp.Queue()
    processes = [mp.Process(target=sample_fcn, args=(numSamples(thread, samples, threads), level, particles_per_sample, delta_t_f, epsilon,
                                                     stop_times, bound_L, bound_R, bound_L_type, bound_R_type, dist_fcn, scale_factor, output)) for thread in range(threads)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    sample_sum = np.zeros((1, len(stop_times)))
    sample_sum2 = np.zeros((1, len(stop_times)))
    sample_diff_sum = np.zeros((1, len(stop_times)))
    sample_diff_sum2 = np.zeros((1, len(stop_times)))

    for p in processes:
        (ss, ss2, sds, sds2) = output.get()
        sample_sum += ss
        sample_sum2 += ss2
        sample_diff_sum += sds
        sample_diff_sum2 += sds2

    return (sample_sum, sample_sum2, sample_diff_sum, sample_diff_sum2)


def numSamples(thread, samples, threads):
    div = samples//threads
    rem = samples % threads
    if thread < rem:
        return div+1
    else:
        return div


def sample_fcn(samples, level, particles_per_sample, delta_t_f, epsilon, stop_times, bound_L, bound_R, bound_L_type, bound_R_type, dist_fcn, scale_factor, output):

    particle_mass = 1/particles_per_sample
    sample_sum = np.zeros(len(stop_times))
    sample_sum2 = np.zeros(len(stop_times))
    sample_diff_sum = np.zeros(len(stop_times))
    sample_diff_sum2 = np.zeros(len(stop_times))

    if level > 0:
        delta_t_r = delta_t_f*scale_factor
    else:
        delta_t_r = delta_t_f

    for _ in range(samples):
        # Generate distribution of particles
        position_random = [dist_fcn(random())
                           for _ in range(particles_per_sample)]
        particles = [GTparticle(particle_mass, bound_L, bound_R,
                                epsilon, delta_t_r, pos) for pos in position_random]
        if level != 0:
            particles_f = [GTparticle(
                particle_mass, bound_L, bound_R, epsilon, delta_t_f, pos) for pos in position_random]
            for i in range(particles_per_sample):
                particles_f[i].speed *= np.sign(particles_f[i].speed) * \
                    np.sign(particles[i].speed)
            particles = (particles_f, particles)

        t = 0
        for stop_time_ind in range(len(stop_times)):
            stop_time = stop_times[stop_time_ind]
            # Simulate until we are interested in taking a measurement
            while t < stop_time:
                if level == 0:
                    genericParticle.make_time_step(
                        particles, delta_t_r, epsilon, bound_L, bound_R, bound_L_type, bound_R_type)
                else:
                    genericParticle.make_correlated_time_steps(particles, delta_t_f, delta_t_r, epsilon,
                                                               bound_L, bound_R, bound_L_type, bound_R_type)
                t += delta_t_r

            # Calculate moments
            this_sample = 0.0
            this_sample_diff = 0.0
            for i in range(particles_per_sample):
                if level == 0:
                    this_sample += particles[i].position**2
                    this_sample_diff = this_sample
                else:
                    this_sample += particles[0][i].position**2
                    this_sample_diff += particles[0][i].position**2
                    this_sample_diff -= particles[1][i].position**2
            this_sample /= particles_per_sample
            this_sample_diff /= particles_per_sample
            sample_sum[stop_time_ind] += this_sample
            sample_sum2[stop_time_ind] += this_sample**2
            sample_diff_sum[stop_time_ind] += this_sample_diff
            sample_diff_sum2[stop_time_ind] += this_sample_diff**2

    output.put((sample_sum, sample_sum2, sample_diff_sum, sample_diff_sum2))


def level_timestep(level, epsilon, dt0, scale_factor):
    return (dt0/scale_factor**(level), scale_factor)


def cost_fcn(level, dt0, epsilon, scale_factor):
    if level == 0:
        return 1
    else:
        return 2**(level-1)*(1+scale_factor)


def distribution(norm):
    return 0


particles_per_sample = 1
delta_t0 = 0.5
epsilon = float(sys.argv[1])
t_end = 0.51
stop_times = np.arange(0, t_end, delta_t0)
bound_L = -1000000
bound_R = 1000000
bound_L_type = "circular"
bound_R_type = "circular"
scale_factor = 2
threads = 20


def fcn(samples, level): return sample_wrapper(samples, level, particles_per_sample, delta_t0, epsilon,
                                               stop_times, bound_L, bound_R, bound_L_type, bound_R_type, distribution, scale_factor, threads)


def cost_fun(level): return cost_fcn(level, delta_t0, epsilon, scale_factor)


alpha = 1
eps = float(sys.argv[2])
max_samples = sys.maxsize
seed = None
convergence_fcn = None
default_samples = int(sys.argv[3])
levels = [default_samples]

if epsilon == 0:
    checkpoint = sys.argv[4]
else:
    checkpoint = None


simulation_data = MLMC_sim(levels, fcn, cost_fun, alpha, eps,
                           default_samples, max_samples, seed, convergence_fcn, checkpoint)

print("Mean")
print(simulation_data.mean())
print("Telescopic sum")
print(simulation_data.telescopic_sum(simulation_data.num_levels))
print("Variance of differences")
print(simulation_data.diff_variance())
print("Number of samples")
print(simulation_data.samples)
print("Total work")
print(simulation_data.total_work)
