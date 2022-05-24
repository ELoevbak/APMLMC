In order to generate the in the tables in the paper from scratch, use the following commands for, respectively, Tables 1-3 and 5-7:
* python3 squared_displacement_geometric_levels.py 0.1 0.1 40
* python3 squared_displacement_geometric_levels.py 0.1 0.01 500
* python3 squared_displacement_geometric_levels.py 0.1 0.001 1000
* python3 squared_displacement_coarse_level.py 0.1 0.1 40
* python3 squared_displacement_coarse_level.py 0.1 0.01 500
* python3 squared_displacement_coarse_level.py 0.1 0.001 1000

The provided parameters are epsilon, the RMSE and initial samples per level. Note that some simulations will take a long time, especially for small RMSE. For this reason you can also find .chk files for each simulation as performed when preparing the paper.

Most data is printed to the terminal when a simulation is run, but if you want to look at non-printed metrics, you can read in the last .chk file produced by a simulation.

Reading in .chk files is done as follows:
1) Open up a python3 shell.
2) Import mlmc.py (To do this you will either need to copy it to the current directory or add ../../ to the python path, e.g. as is done at the top of the python scripts in this folder.)
3) Load the checkpoint by typing data = mlmc.load_checkpoint("<checkpoint-file-name>").
4) The resulting object is of type mlmc and you can call any instance method and read internal state values as defined in mlmc.py. (E.g. data.samples, data.mean(), ...).
