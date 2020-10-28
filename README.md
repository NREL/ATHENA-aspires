# ASPIRES: Airport Shuttle Planning and Improved Routing Event-driven Simulation
## Contributors:
Qichao Wang <Qichao.Wang@nrel.gov>
<br>
Devon Sigler <Devon.Sigler@nrel.gov>
<br>
Andrew Kotz <Andrew.Kotz@nrel.gov>
<br>
Zhaocai Liu <Zhaocai.Liu@nrel.gov>
<br>
Kenneth Kelly <Kenneth.Kelly@nrel.gov>
<br>
Caleb Phillips <Caleb.Phillips@nrel.gov>

### Purpose
To fast simulate airport shuttle operations with the data collected.
The simulation inputs are passenger arrival rates, shuttle routes and frequencies, and simulation configurations. The outputs include: time dependent shuttle energy level, time dependent charging station usage, time dependent number of passengers on each bus, time dependent number of passengers at each bus stop, history of number of passengers left at each bus stop, route history of fleet shuttle buses and on-demand buses, and time dependent bus distance traveled.
### This repo contains the following items
- Core code for the software
	* DES_shuttle.py
	* environment.yml
- Input data
	* [data/](https://github.com/NREL/ATHENA-aspires/tree/master/data_2020)
	* /data_2020/arrRate.pickle: the estimated arrival rate of passengers at each shuttle bus stop.
	* /data_2020/Time_n_Energy_Dictionary_Nested_Full: the time and energy cost of each link between two nodes from bus logger data
	* /data_2020/Time_n_Energy_Dictionary_Nested_Full_p: the time and energy cost of each link between two nodes from combined bus logger data and simulation data
	* /data_2020/FleetSize_MixedRoute/: the estimated hourly number of buses needed for each route for each day
	* /data_2020/SPOT/: folder to put the SPOT data
- Exapmle scripts to run simulation
	* On HPC: Asim.slurm
- Exaplme outputs:
	* Asim.log.log
	* result_baseline.pckl
## Installation instructions
### 1. Clone this repo
In terminal, type
``` bash
git clone https://github.com/NREL/ATHENA-aspires.git
```
### 2. Setup conda environment

Go to the repo
``` bash
cd ATHENA-aspires
```
Then create the environment
``` bash
conda env create --file environment.yml
```
Activate the environment
``` bash
conda activate ASPIRES
```
### 3. Run the simulation
The simulation command includes several optional parameters. The code is in DES_shuttle.py.
One example is
First go to the directory that has DES_shuttle
``` bash
cd <path to ATHENA-aspires>
```
Then type the following command
``` bash
python DES_shuttle.py --StartingDayOfWeek 1 --SimTime 8 --doHotShot True --outputName baseline --maxqueue 150
```
Asim.slurm is the script to run ASPIRES on HPC.

### 4. Update data (Optional)
- When new SPOT data is available, put the SPOT data csv files into /data_2020/SPOT/. Then remove /data_2020/arrRate.pickle.
- When new simulation data from SUMO is available, name the simulated data as SUMO_AverageDayBusOutput.csv, place it under /data_2020/, and remove /data_2020/Time_n_Energy_Dictionary_Nested_Full_p.npy.
- The optimized route data can be found on eagle under this path: /projects/athena/bus_opt/bus_opt_csvs/.
