The folders here contain simulation checkpoints corresponding with the tables in the supplementary materials of the paper as well as a script which can be used to re-run the simulations. If you wish to re-compute one of the results. you can run the script from the folder at hand with

python3 perform_simulation.py <epsilon> <RMSE> <skipped_level_factor>

, where skipped_level_factor is the integer ratio epsilon^2 / Delta t_1. The last generated .chk file corresponds with program state at the end of the simulation. The number of threads used for the simulation is hard-coded at the bottom of the script. Note that some simulations will take a long time, especially for small epsilon.

Latex tables can be generated from the provided checkpoints by running the script table_iterator.sh with one of the four folders here as an argument. This script assumes that the checkpoints have filenames following the convention <epsilon>_<skipped_level_factor>.chk.