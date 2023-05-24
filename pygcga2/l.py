from ase.io.trajectory import Trajectory 
import glob 

name = glob.glob('*.traj')
name=name[0]

traj = Trajectory(name)
print(len(traj))
