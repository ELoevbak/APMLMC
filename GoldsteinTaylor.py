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

"""Containts code for computing a single step of the AP(ML)MC scheme for the Goldstein-Taylor model in one dimension for a given distribution of particles."""




import random
import numpy as np
from genericParticle import genericParticle
def compute_E_cum_dist(n, k_max, p):
    """Implementation of the calculation of the cumulative distribution of E_{n,k}, using Fu and Koutras, Distribution Theory of Runs."""

    l_max = (1+n)//2
    dist = np.zeros((k_max, l_max+1))
    for k in range(1, k_max+1):
        l = (n+1)//(k+1)
        # Build matrix
        transition_matrix = build_transition_matrix_E(l, k, p)
        # Perform computation
        state = np.zeros((l+1)*(k+2)-1)
        state[1] = 1
        transition_matrix = np.linalg.matrix_power(transition_matrix, n)
        state = np.matmul(state, transition_matrix)
        # Calculate probability distribution
        dist[k-1, :] = np.sum(state[:k+1])
        for x in range(1, l+1):
            dist[k-1, x:] += np.sum(state[(k+2)*(x-1)+k+1:(k+2)*x+k+1])
    return dist


def compute_G_cum_dist(n, k_max, p):
    """Implementation of the calculation of the cumulative distribution of G_{n,k}, using Fu and Koutras, Distribution Theory of Runs."""

    l_max = (1+n)//2
    dist = np.zeros((k_max, l_max+1))
    for k in range(1, k_max+1):
        l = (n+1)//(k+1)
        # Build matrix
        transition_matrix = build_transition_matrix_G(l, k, p)
        # Perform computation
        state = np.zeros((l+1)*(k+1)-1)
        state[0] = 1
        transition_matrix = np.linalg.matrix_power(transition_matrix, n)
        state = np.matmul(state, transition_matrix)
        # Calculate probability distribution
        dist[k-1, :] = np.sum(state[:k])
        for x in range(1, l+1):
            dist[k-1, x:] += np.sum(state[(k+1)*(x-1)+k:(k+1)*x+k])
    return dist


def build_transition_matrix_E(l, k, p):
    """Helper function for compute_E_dist"""

    q = 1-p
    transition_matrix = np.zeros(((l+1)*(k+2)-1, (l+1)*(k+2)-1))
    transition_matrix[0, 0] = p
    transition_matrix[0, 1] = q
    for x in range(l):
        for i in range(k):
            transition_matrix[x*(k+2)+i+1, x*(k+2)+1] = q
            transition_matrix[x*(k+2)+i+1, x*(k+2)+i+2] = p
        transition_matrix[x*(k+2)+k+1, x*(k+2)] = p
        transition_matrix[(x+1)*(k+2), (x+1)*(k+2)] = p
        transition_matrix[x*(k+2)+k+1, (x+1)*(k+2)+1] = q
        transition_matrix[(x+1)*(k+2), (x+1)*(k+2)+1] = q
    for i in range(k-1):
        transition_matrix[l*(k+2)+i+1, l*(k+2)+1] = q
        transition_matrix[l*(k+2)+i+1, l*(k+2)+i+2] = p
    transition_matrix[(l+1)*(k+2)-2, (l+1)*(k+2)-2] = 1
    return transition_matrix


def build_transition_matrix_G(l, k, p):
    """Helper function for compute_G_dist"""

    q = 1-p
    transition_matrix = np.zeros(((l+1)*(k+1)-1, (l+1)*(k+1)-1))
    for x in range(l):
        for i in range(k):
            transition_matrix[x*(k+1)+i, x*(k+1)] = q
            transition_matrix[x*(k+1)+i, x*(k+1)+i+1] = p
        transition_matrix[x*(k+1)+k, x*(k+1)+k] = p
        transition_matrix[x*(k+1)+k, (x+1)*(k+1)] = q
    for i in range(k-1):
        transition_matrix[l*(k+1)+i, l*(k+1)] = q
        transition_matrix[l*(k+1)+i, l*(k+1)+i+1] = p
    transition_matrix[(l+1)*(k+1)-2, (l+1)*(k+1)-2] = 1
    return transition_matrix


class GTparticle(genericParticle):

    """Represents a single particle under the Goldstein-Taylor model.
    Member variables:
        position: Numeric value
        speed: Numeric value
        is_active: Boolean indicating whether the particle has non-zero mass in the simulation.
    """

    def _virtual_fine_simulation(self, delta_t_coarse, delta_t_fine, epsilon, theta):
        """Implements the independent coarse simulation as developed in the text written on 16/10/2019
        Variables as above, we assume that the fine dt is at epsilon^2
        """
        p_c = delta_t_fine/(delta_t_fine+epsilon**2)
        p_nc = epsilon**2/(delta_t_fine+epsilon**2)
        # Generate gaussian value
        xi_W = random.gauss(0, 1)
        # Generate value as described in the document, representing transport increments
        M = int(delta_t_coarse/delta_t_fine)
        Lam = 20
        xi_T = 0

        # We opt for global values here so that the calling script does not need break abstraction in order
        # to manage this computation. As a safeguard we append a random sequence of numbers to the variable names.
        global run_table125697541, run_table265844854
        if not 'run_table125697541' in globals():
            # Number of collision moments is one more than number of time steps
            run_table125697541 = compute_G_cum_dist(M-1, Lam, p_nc)
            run_table265844854 = compute_E_cum_dist(M-1, Lam, p_nc)

        run_sum = 0
        for lam in range(Lam, 1, -1):
            u = np.random.uniform()
            if(run_sum == 0):
                phi_larger = next(idx for idx, value in enumerate(
                    run_table265844854[lam-2, :]) if value >= u)
            else:
                phi_larger = next(idx for idx, value in enumerate(
                    run_table125697541[lam-2, :]) if value >= u)
            phi = max(phi_larger-run_sum, 0)
            M -= lam*phi
            while M < 0:
                M += lam
                phi -= 1
            run_sum += phi
            tmp = self._sample_sum_distribution(phi)
            xi_T += lam * tmp

        phi = M
        tmp = self._sample_sum_distribution(phi)
        xi_T += tmp

        # +2**(2-M) Left out as easily neglected and expensive to compute
        xi_T /= np.sqrt(M + 2*p_nc*(p_nc**M+M*p_c-1)/(p_c**2))
        # Return a weighted combination
        return xi_W*np.sqrt(theta) + xi_T*np.sqrt(1-theta)

    def _sample_sum_distribution(self, phi):
        """Sample the sum of phi random values +-1. For sampling we use a table which we extend when needed."""

        # We opt for global values here so that the calling script does not need break abstraction in order
        # to manage this computation. As a safeguard we append a random sequence of numbers to the variable names.
        global PDF_table556215985, CDF_table421586995
        if not 'CDF_table421586995' in globals():
            PDF_table556215985 = [[1.0]]
            CDF_table421586995 = [[1.0]]

        # Ensure table contains the values that we need
        if len(CDF_table421586995) <= phi:
            for row_index in range(len(CDF_table421586995), phi+1):
                new_row = np.empty((row_index+1, 1))
                for i in range(1, len(new_row)-1):
                    new_row[i] = (PDF_table556215985[row_index-1][i] +
                                  PDF_table556215985[row_index-1][i-1]) / 2
                new_row[0] = PDF_table556215985[row_index-1][0]/2
                new_row[-1] = PDF_table556215985[row_index-1][-1]/2
                PDF_table556215985.append(new_row)
                sum_new_row = np.cumsum(new_row)
                CDF_table421586995.append(sum_new_row)

        # Sample from table
        u = np.random.uniform()
        return 2*next(idx for idx, value in enumerate(CDF_table421586995[int(phi)]) if value >= u) - phi

    def _sample_velocity_distribution(self):

        return np.sign(random.random()-0.5)
