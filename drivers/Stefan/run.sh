#
# Run script for HeatTransientDriver
#

# Test norm
#norm=Graph
norm=ManualGraph

# Formulation
#form=Ultraweak
form=Primal
#form=Galerkin

# Other
prob=Manufactured
p=1
dp=2

theta=1.0 # Backward Euler
#theta=0.5 # Crank Nicolson
#theta=0.0 # Forward Euler

# Time convergence study (32x32 mesh)
#dt=0.1
#dt=0.05
#dt=0.025
#dt=0.0125
#dt=0.00625
#dt=0.003125
#dt=0.0015625
#dt=0.00078125
#dt=0.000390625
#dt=0.0001953125

T=0.1
nrElems=32
vis=novis
#vis=vis
run=runOne
#run=runAll

mpirun -np 1 ./HeatTransientDriver --norm=${norm} --formulation=${form} --problem=${prob} --meshWidth=${nrElems} --polyOrder=${p} --deltaP=${dp} --dt=${dt} --T=${T} --theta=${theta} --${vis} --${run} --x0=0.0 --x1=1.0 --y0=0.0 --y1=1.0
