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

import random
from genericParticle import genericParticle


class gaussianParticle(genericParticle):

    """Implementation of a 1D particle with Gaussian velocities."""

    def _virtual_fine_simulation(self, dt_coarse, dt_fine, epsilon, theta):
        """Simply sample a velocity and ignore parameters."""

        return random.gauss(0, 1)

    def _sample_velocity_distribution(self):
        """Gaussian velocity."""

        return random.gauss(0, 1)
