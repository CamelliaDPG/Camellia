# start with clean environment
module --force purge

# StdEnv is required for detecting compiler modules
module load StdEnv

# ... but StdEnv autoloads compilers and MPI; we unload those
module unload mvapich2/2.3.7
module unload intel-classic/2021.6.0-magic

# now we load the compilers and MPI we actually want
module load gcc/13.3.1
module load openmpi/4.1.2
