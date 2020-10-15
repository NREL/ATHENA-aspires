#!/usr/bin/env bash
#SBATCH --job-name=ASPIRES_test
#SBATCH --output=Asim.log.log
#SBATCH --account=athena
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --partition=debug

module purge
module load conda 

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=5_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=5_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=5_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=5_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=5_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=5_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=7_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=7_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=7_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=7_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=7_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=7_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=10_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=10_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=10_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=10_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=10_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=10_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=15_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=15_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=15_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=15_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=15_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=15_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=20_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=20_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=20_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=20_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=20_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=20_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=25_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=25_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=25_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=25_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=15_hw=25_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=15_hw=25_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=5_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=5_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=5_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=5_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=5_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=5_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=7_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=7_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=7_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=7_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=7_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=7_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=10_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=10_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=10_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=10_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=10_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=10_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=15_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=15_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=15_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=15_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=15_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=15_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=20_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=20_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=20_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=20_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=20_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=20_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=25_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=25_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=25_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=25_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=20_hw=25_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=20_hw=25_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=5_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=5_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=5_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=5_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=5_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=5_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=7_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=7_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=7_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=7_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=7_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=7_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=10_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=10_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=10_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=10_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=10_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=10_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=15_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=15_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=15_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=15_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=15_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=15_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=20_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=20_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=20_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=20_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=20_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=20_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=25_sd=0_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=25_sd=0_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=25_sd=1_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=25_sd=1_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

srun -N 1 -n 1 ~/.conda-envs/aspires/bin/python /home/qwang3/athena/ASPIRES/DES_shuttle.py --RoutesFile mr=25_hw=25_sd=2_routes=5.csv --StartingDayOfWeek 0 --SimTime 29 --doHotShot False --outputName '/projects/athena/Qichao/shuttleOptimization/result_noHotshot_mr=25_hw=25_sd=2_routes=5.csv' --RoutesFileFolder '/projects/athena/bus_opt/bus_opt_csvs/CSV_files_5route_60min_43' --maxqueue 150 &

wait
