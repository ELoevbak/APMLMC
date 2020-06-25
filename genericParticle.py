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

"""Contains generic methods and a class for simulating (ensembles of) particle(s)."""




import numpy as np
import math
import random
import copy
def make_time_step(particles, delta_t, epsilon, boundary_L, boundary_R, boundary_type_L, boundary_type_R, emulate_correlation=None, M=2):
    """Perform time step for an array of particles.
    Variable types:
        particles: A list of particle objects which inherit from the class genericParticle.
        delta_t: Numeric value
        epsilon: Numeric value
        boundary_L, boundary_R: Numeric values representing the x-coordinates of the boundaries.
        boundary_type_L, boundary_type_R: String values that can be either "no flux", "circular" or "outflow".
        emulate_correlation: Set to None if this simulation is as specified by the model. Set to "combined" if this is a level 0 simulation where the combined correlation is used at level 1.
        M: If emulate_correlation is set to "combined", then this must be set equal to the value of M at level 1.
    """

    for particle in particles:
        particle.make_time_step(delta_t, epsilon, boundary_L, boundary_R,
                                boundary_type_L, boundary_type_R, None, emulate_correlation, M=M)


def make_correlated_time_steps(particles, delta_t_fine, delta_t_coarse, epsilon, boundary_L, boundary_R, boundary_type_L, boundary_type_R, correlation_type="independent"):
    """Perform correlated time steps for an array of tuples of particles.
    Variable types:
        particles: A tuple of lists of particle objects which inherit from the class genericParticle. All particles with the same
                   list index undergo correlated simulations with the time_steps given in delta_t. The tuple should be ordered
                   from finest to roughest step size.
        delta_t: A tuple of numeric values corresponding to the time_steps of the pairs of particles. Each subsequent value should
                 be an integer multiple of the previous.
        epsilon: Numeric value
        boundary_L, boundary_R: Numeric values representing the x-coordinates of the boundaries.
        boundary_type_L, boundary_type_R: String values that can be either "no flux", "circular" or "outflow".
        correlation_type: A string indicating how to correlate simulations.
    """

    random_numbers = [()]*len(particles[0])

    # This block computes a bunch of constants from the matching papers
    M = int(delta_t_coarse / delta_t_fine)
    p_nc = epsilon**2/(delta_t_fine+epsilon**2)
    p_c = 1-p_nc
    vc_fine = epsilon / (delta_t_fine+epsilon**2)
    D_coarse = delta_t_coarse / (delta_t_coarse+epsilon**2)
    D_fine = delta_t_fine / (delta_t_fine+epsilon**2)
    V_sum_variance = M + 2 * p_nc / \
        p_c**2 * (p_nc**M+M*p_c-1)
    V_sum_product_expectation = M - 2*p_nc/p_c - 1 + \
        (2*(M-2)/p_c + (M-1)*(M + 2/p_c - 1))*p_nc**M + \
        2/p_c**2 * (3*p_nc**M - (M+1)*p_nc**2 + (M-2)*p_nc)
    C2_1 = 4*delta_t_coarse**2 * D_coarse*D_fine
    C2_2 = delta_t_fine**2 * vc_fine**2 * 2*delta_t_coarse * \
        D_coarse/V_sum_variance*V_sum_product_expectation**2
    theta = C2_1 / (C2_1 + C2_2)

    for i in range(0, len(particles[0])):
        particle = particles[0][i]
        for _ in range(0, M):
            random_numbers[i] += (particle.make_time_step(delta_t_fine, epsilon, boundary_L, boundary_R, boundary_type_L,
                                                          boundary_type_R, None, None, M, theta),)
        particle = particles[1][i]
        particle.make_correlated_big_step(delta_t_coarse, epsilon, boundary_L, boundary_R, boundary_type_L, boundary_type_R,
                                          random_numbers[i], V_sum_variance, correlation_type, theta)


class genericParticle:

    """Contains non model-specific particle behavior."""

    def __init__(self, mass, left_x, right_x, epsilon, delta_t, start_position=None):
        """Init
        Variable types:
            mass: A numeric value representing the particle mass.
            left_x, right_x: The bounds of the domain containing the particle.
            epsilon: Numeric value
            delta_t: Numeric value
            start_position: Optional. The initial position. If not provided then a random position is chosen.
        """

        self.mass = mass
        if start_position is None:
            self.position = random.random() * (right_x-left_x) + left_x
        else:
            self.position = start_position
        speed_dimless = self._sample_velocity_distribution()
        self.speed = speed_dimless * epsilon / (epsilon**2 + delta_t)
        self.is_active = True

    def __str__(self):

        return "Position:" + str(self.position) + " Speed:" + str(self.speed) + " Active:" + str(self.is_active)

    def make_time_step(self, delta_t, epsilon, boundary_L, boundary_R, boundary_type_L, boundary_type_R, random_numbers=None, emulate_correlation=None, M=2, theta=1):
        """Perform time step for this particle.
        Variable types:
            delta_t: Numeric value
            epsilon: Numeric value
            boundary_L, boundary_R: Numeric values representing the x-coordinates of the boundaries.
            boundary_type_L, boundary_type_R: String values that can be either "no flux", "circular" or "outflow".
            random_numbers: A tuple of the random number information to be passed to the called methods.
            emulate_correlation: Set to None if this simulation is as specified by the model. Set to "combined" if this is a level 0 simulation where the combined correlation is used at level 1.
            M: If emulate_correlation is set to "combined", then this must be set equal to the value of M at level 1.
            theta: If emulate correlation is set to "combined", then this must be equal to the value of theta at level 1.
        returns: The random numbers used in the called methods.
        """
        if random_numbers is None:
            transport_random = None
            collision_random = None
        else:
            transport_random = random_numbers[0]
            collision_random = random_numbers[1]
        # Simulate particle
        transport_step = self._transport_diffusion_step(
            delta_t, epsilon, emulate_correlation, transport_random, M=M, theta=theta)
        collision_step = self._collision_step(
            delta_t, epsilon, collision_random)
        self._check_bounds(boundary_L, boundary_R,
                           boundary_type_L, boundary_type_R)

        return (transport_step, collision_step)

    def make_correlated_big_step(self, delta_t, epsilon, boundary_L, boundary_R, boundary_type_L, boundary_type_R, random_numbers, V_sum_variance, correlation_type=None, theta=1):
        """Perform time step which is correlated with the time steps of the finer simulation.
        Variable types:
            delta_t: Numeric value, is the big time step.
            epsilon: Numeric value
            boundary_L, boundary_R: Numeric values representing the x-coordinates of the boundaries.
            boundary_type_L, boundary_type_R: String values that can be either "no flux", "circular" or "outflow".
            random_numbers: A tuple of the random number information returned by the steps of the finer simulation.
            V_sum_variance: The variance of the sum of fine velocities, computed in the calling function.
            correlation_type: A string which takes the value "independent" or "combined".
            theta: If emulate correlation is set to "combined", then this parameter indicates how much of the coarse diffusion comes from fine diffusion.
        returns: The random numbers used in the called methods.
        """
        M = len(random_numbers)

        # Calculate new random numbers
        transport_random = [0]

        # Generate browian increment
        for sub_step in random_numbers:
            # Ugly because clean code was too slow
            transport_random[0] += sub_step[0][0]
        # Rescaling to preserve variance of rougher model
        transport_random[0] /= np.sqrt(M)

        # Generate sum of velocities, if combined correlation_type
        if correlation_type == "independent":
            pass  # Do nothing, but included so we have a full list here
        elif correlation_type == "combined":
            if delta_t > epsilon**2:
                # Calculate influence of fine transport increments
                V_sum = 0.0
                fine_speed = self._sample_velocity_distribution()
                counter = 0
                for sub_step in random_numbers:
                    V_sum += fine_speed
                    counter += 1
                    collision_occurred = sub_step[1][0] > epsilon**2 / \
                        (epsilon**2 + delta_t/M)
                    if collision_occurred:
                        fine_speed = sub_step[1][1]
                        counter = 0
                V_sum += self._sample_velocity_distribution()*counter
                V_sum -= fine_speed*counter
                V_sum /= np.sqrt(V_sum_variance)
                transport_random[0] = transport_random[0] * \
                    np.sqrt(theta) + V_sum*np.sqrt(1-theta)
        else:
            raise ValueError("Correlation type should be either 'independent' or 'combined'. Received '" +
                             correlation_type + "'!")

        # Random initialization of second element so that Kolmogorov-Smirnov test will pass if no collisions happen
        collision_random = [0, self._sample_velocity_distribution()]
        for sub_step in random_numbers:
            # At least one jump occurs based on largest u in the fine simulation
            if collision_random[0] < sub_step[1][0]:
                collision_random[0] = sub_step[1][0]
            # The last jump that occurs will influence the speed
            jumped = sub_step[1][0] > epsilon**2 / (epsilon**2 + delta_t/M)
            if jumped:
                collision_random[1] = sub_step[1][1]
        # Float in exponent to make computation cheaper for level 1
        collision_random[0] **= float(M)
        return self.make_time_step(delta_t, epsilon, boundary_L, boundary_R, boundary_type_L, boundary_type_R, random_numbers=(transport_random, collision_random), emulate_correlation=None)

    def _transport_diffusion_step(self, delta_t, epsilon, emulate_correlation, random_numbers=None, M=2, theta=1):
        """Perform diffusion step for a single particle.
        Variable types:
            particle: A particle object
            delta_t: Numeric value
            epsilon: Numeric value
            emulate_correlation: Should be either None or "combined". The latter case applies when this is a level 0 simulation and the corresponding level 1 simulation uses the combined approach.
            random_numbers: The random numbers to be used in this step.
            M: Numeric, the number of fine steps in a coarse step.
            theta: If emulate_correlation is set to "combined", then this should be equal to the theta in the level 1 simulation.
        returns: The random numbers used.
        """

        if random_numbers is None:
            # Here we generate random numbers which are distributed as if they are correlated with a fine simulation
            if emulate_correlation is None:
                xi = random.gauss(0, 1)
            elif emulate_correlation == "combined":
                xi = self._virtual_fine_simulation(
                    delta_t, delta_t/M, epsilon, theta)
            else:
                raise ValueError(
                    "If provided, emulate_correlation should be set to 'combined'. Received '" + emulate_correlation + "'!")
        else:
            xi = random_numbers[0]

        sigma = np.sqrt(2 * delta_t**2 / (epsilon**2 + delta_t))
        pos = self.position
        speed = self.speed
        self.position = pos + speed*delta_t + sigma*xi
        return [xi]

    def _collision_step(self, delta_t, epsilon, random_numbers):
        """"Perform collision step for a single particle and a boolean indicating if the particle is active.
        Variable types:
            particle: A particle object
            delta_t: Numeric value
            epsilon: Numeric value
            random_numbers: The random numbers to be used in this step.
        returns: The random numbers used.
        """

        # Always generate two random numbers for simplicity in other methods, may become a bottleneck later.
        if random_numbers is None:
            u = random.random()
            V = self._sample_velocity_distribution()
        else:
            u = random_numbers[0]
            V = random_numbers[1]

        collision_occurred = u > epsilon**2 / (epsilon**2 + delta_t)

        if(collision_occurred):
            characteristic_velocity = epsilon / (epsilon**2 + delta_t)
            self.speed = V * characteristic_velocity
        return [u, V]

    def _virtual_fine_simulation(self, dt_coarse, dt_fine, epsilon, theta):
        """Generate coarse diffusion as if it comes from a combined correlation."""
        raise NotImplementedError("No method implemented in subclass!")

    def _sample_velocity_distribution(self):
        """Returns a sample from the steady-state velocity distribution."""
        raise NotImplementedError("No method implemented in subclass!")

    def _apply_boundary_condition(self, boundary_L, boundary_R, boundary_type, side):
        """Correct particle position if in has strayed out of range.
        Variable types:
            boundary_L, boundary_R: Space coordinates of the boundaries.
            boundary_type: A string giving the type of boundary. Can be either "no flux", "circular" or "outflow".
            side: A string giving which side the boundary is on. Can be either "L" or "R".
        """
        # Convert boundary to a left side boundary at zero
        if(side == "R"):
            shift = -boundary_R
            rotation = -1
        elif(side == "L"):
            shift = -boundary_L
            rotation = 1
        else:
            raise ValueError(
                "Boundary side should be either 'L' or 'R'. Received '" + side + "'!")
        pos = self.position
        speed = self.speed
        pos = rotation * (pos + shift)
        intervalLength = boundary_R - boundary_L
        boundary_L = rotation * (boundary_L + shift)
        boundary_R = rotation * (boundary_R + shift)
        speed = rotation*speed
        self.position = pos
        self.speed = speed

        # Apply condition to generic case
        self._apply_boundary_condition_L(intervalLength, boundary_type)

        # Convert back to specific case
        pos = self.position
        speed = self.speed
        pos = rotation * pos - shift
        boundary_L = rotation * boundary_L - shift
        boundary_R = rotation * boundary_R - shift
        speed = rotation*speed
        self.position = pos
        self.speed = speed

    def _apply_boundary_condition_L(self, intervalLength, boundary_type):
        """Applies a specific boundary condition to a particle in the generic case of a left hand boundary at zero.
        Variable types:
            intervalLength: A numeric value indicating the distance between the left and right bounds.
            boundary_type: A string giving the type of boundary. Can be either "no flux", "circular" or "outflow".
        """

        if(boundary_type == "no flux"):
            # Keep the following line from triggering a bug in pylint
            # https://github.com/PyCQA/pylint/issues/1472
            # pylint: disable=invalid-unary-operand-type
            self.position = -self.position
            # pylint: enable=invalid-unary-operand-type
            self.speed = -self.speed
        elif(boundary_type == "circular"):
            self.position = self.position + intervalLength
        elif(boundary_type == "outflow"):
            self.is_active = False
        else:
            raise ValueError(
                "Boundary type should be either 'no flux', 'circular' or 'outflow'! Received '" + boundary_type + "'!")

    def _check_bounds(self, boundary_L, boundary_R, boundary_type_L, boundary_type_R):
        """Check whether the particle is out of bounds.
        Variable types:
            boundary_L, boundary_R: Numeric values representing the x-coordinates of the boundaries.
            boundary_type_L, boundary_type_R: String values that can either be "no flux", "circular" or "outflow".
        """

        position_valid = False
        while (not position_valid) and self.is_active:
            if(self.position < boundary_L):
                self._apply_boundary_condition(
                    boundary_L, boundary_R, boundary_type_L, 'L')
            elif(self.position > boundary_R):
                self._apply_boundary_condition(
                    boundary_L, boundary_R, boundary_type_R, 'R')
            else:
                position_valid = True

    def filtered_mass(self, boundary_L, boundary_R):
        """Return the mass contribution of this particle in the simulation in region of space.
        Variable types:
            bound_L, bound_R: The bounds of the area over which the mass is being calculated.
        """

        if (self.position > boundary_L) and (self.position < boundary_R) and self.is_active:
            return self.mass
        else:
            return 0
