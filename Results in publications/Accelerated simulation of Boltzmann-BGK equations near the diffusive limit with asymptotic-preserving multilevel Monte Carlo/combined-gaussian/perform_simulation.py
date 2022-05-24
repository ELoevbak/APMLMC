import sys
import numpy as np
from random import random, gauss
sys.path.append("../../../Multilevel_AP_simulation_code/")
from mlmc import MLMC_sim
import GaussianVelocity
from GaussianVelocity import gaussianParticle
import genericParticle
import multiprocessing as mp
import math


def sample_wrapper(samples, level, particles_per_sample, delta_t0, epsilon, moments, stop_times, bound_L, bound_R, bound_L_type, bound_R_type, dist_fcn, scale_factor, threads, level_1_refinement):

    print("sample fun")
    print(level)
    (delta_t_f, scale_factor) = level_timestep(level, epsilon, delta_t0, scale_factor, level_1_refinement)
    print(delta_t_f)
    
    output = mp.Queue()
    processes = [mp.Process(target=sample_fcn, args=(numSamples(thread, samples, threads), level, particles_per_sample, delta_t_f, epsilon, moments, stop_times, bound_L, bound_R, bound_L_type, bound_R_type, dist_fcn, scale_factor, output)) for thread in range(threads)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    sample_sum = np.zeros((moments, len(stop_times)))
    sample_sum2 = np.zeros((moments, len(stop_times)))
    sample_diff_sum = np.zeros((moments, len(stop_times)))
    sample_diff_sum2 = np.zeros((moments, len(stop_times)))

    for p in processes:
        (ss, ss2, sds, sds2) = output.get()
        sample_sum += ss
        sample_sum2 += ss2
        sample_diff_sum += sds
        sample_diff_sum2 += sds2
    
    return (sample_sum, sample_sum2, sample_diff_sum, sample_diff_sum2)
    
def numSamples(thread, samples, threads):
    div = samples//threads
    rem = samples%threads
    if thread < rem:
        return div+1
    else:
        return div


def sample_fcn(samples, level, particles_per_sample, delta_t_f, epsilon, moments, stop_times, bound_L, bound_R, bound_L_type, bound_R_type, dist_fcn, scale_factor, output):
    
    particle_mass = 1/particles_per_sample
    sample_sum = np.zeros((moments, len(stop_times)))
    sample_sum2 = np.zeros((moments, len(stop_times)))
    sample_diff_sum = np.zeros((moments, len(stop_times)))
    sample_diff_sum2 = np.zeros((moments, len(stop_times)))

    if level > 0:
        delta_t_r = delta_t_f*scale_factor
    else:
        delta_t_r = delta_t_f
        
    for sample in range(samples):
        #Generate distribution of particles
        position_random = [dist_fcn(random()) for _ in range(particles_per_sample)]
        particles = [gaussianParticle(particle_mass, bound_L, bound_R, epsilon, delta_t_r, pos) for pos in position_random]
        if level != 0:
            particles_f = [gaussianParticle(particle_mass, bound_L, bound_R, epsilon, delta_t_f, pos) for pos in position_random]
            for i in range(particles_per_sample):
                particles_f[i].speed = particles[i].speed*(epsilon**2+delta_t_r)/(epsilon**2+delta_t_f)
            particles = (particles_f, particles)

        t = 0
        for stop_time_ind in range(len(stop_times)):
            stop_time = stop_times[stop_time_ind]
            #Simulate until we are interested in taking a measurement
            while t < stop_time-delta_t_r/2:
                if level == 0:
                    genericParticle.make_time_step(particles, delta_t_r, epsilon, bound_L, bound_R, bound_L_type, bound_R_type)
                else:
                    genericParticle.make_correlated_time_steps(particles, delta_t_f, delta_t_r, epsilon, \
                                                               bound_L, bound_R, bound_L_type, bound_R_type, correlation_type="combined")
                t += delta_t_r
                
            #Calculate moments
            this_sample = np.zeros(moments)
            this_sample_diff = np.zeros(moments)
            for i in range(particles_per_sample):
                for moment in range(1, moments+1):
                    if level == 0:
                        this_sample[moment-1] += particles[i].position**moment
                        this_sample_diff[moment-1] = this_sample[moment-1]
                    else:
                        this_sample[moment-1] += particles[0][i].position**moment
                        this_sample_diff[moment-1] += particles[0][i].position**moment
                        this_sample_diff[moment-1] -= particles[1][i].position**moment
            this_sample /= particles_per_sample
            this_sample_diff /= particles_per_sample
            sample_sum[:, stop_time_ind] += this_sample
            sample_sum2[:, stop_time_ind] += this_sample**2
            sample_diff_sum[:, stop_time_ind] += this_sample_diff
            sample_diff_sum2[:, stop_time_ind] += this_sample_diff**2

    output.put((sample_sum, sample_sum2, sample_diff_sum, sample_diff_sum2))

def level_timestep(level, epsilon, dt0, scale_factor, level_1_refinement):
    epsilon = epsilon/np.sqrt(level_1_refinement)
    ratio = dt0/epsilon**2
    if level==0:
        #Clean up any rounding errors
        return (epsilon**2 * round(ratio), 0)
    if level==1:
        return (epsilon**2, round(ratio))
    return (epsilon**2 / scale_factor**(level-1), scale_factor)

def cost_fcn(level, dt0, epsilon, scale_factor, level_1_refinement):
    start_ratio = dt0/epsilon**2*level_1_refinement
    if level == 0:
        return 1
    if level == 1:
        return 1 + start_ratio
    else:
        return scale_factor**(level-2)*(1+scale_factor)*start_ratio

def distribution(norm):
    return 0

particles_per_sample = 1
delta_t0 = 0.5
epsilon = float(sys.argv[1])
moments = 2
t_end = 0.51
stop_times = np.arange(0, t_end, delta_t0)
bound_L = -1000000
bound_R = 1000000
bound_L_type = "circular"
bound_R_type = "circular"
scale_factor = 2
threads = 35

default_samples = 1000#2*threads
levels = [default_samples]
fcn = lambda samples, level: sample_wrapper(samples, level, particles_per_sample, delta_t0, epsilon, moments, stop_times, bound_L, bound_R, bound_L_type, bound_R_type, distribution, scale_factor, threads, level_1_refinement)
cost_fun = lambda level: cost_fcn(level, delta_t0, epsilon, scale_factor, level_1_refinement)
alpha = 1
eps = float(sys.argv[2])
max_samples = sys.maxsize
seed = None
convergence_fcn = None

level_1_refinement = int(sys.argv[3])

if len(sys.argv) == 5:
    chkpnt = sys.argv[4]
else:
    chkpnt = None

simulation_data = MLMC_sim(levels, fcn, cost_fun, alpha, eps, default_samples, max_samples, seed, convergence_fcn, chkpnt)

print(simulation_data.mean())
print(simulation_data.telescopic_sum(simulation_data.num_levels))
print(simulation_data.diff_variance())
print(simulation_data.samples)
print(simulation_data.total_work)
