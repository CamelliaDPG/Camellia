Help("""
       Type: 'scons' to build the production library,
             'scons -c' to clean the build,
             'scons debug=1' to build the debug version,
             'scons parallel=1' to build the parallel version (default).
       """)

import os

env = Environment(ENV=os.environ)

# Compiler Settings
CC_OPTS     = '-O3'
DEBUG_OPTS  = '-g -Wall'
CPP_COMP    = 'mpicxx'
C_COMP      = 'mpicc'

env['CXX'] = CPP_COMP
env['CC'] = C_COMP

camellia_dir = '#build/'
build_message = 'Building '
# Serial version
parallel = int(ARGUMENTS.get('parallel', 1))
if parallel:
    camellia_dir += 'mpi'
    build_message += 'parallel '
else:
    env.Replace(CXX = 'cpp')
    camellia_dir += 'serial'
    build_message += 'serial '

# Debug options
debug = int(ARGUMENTS.get('debug', 1))
if debug:
    env.Append(CCFLAGS = DEBUG_OPTS)
    camellia_dir += '-debug'
    build_message += 'debug version'
else:
    env.Append(CCFLAGS = CC_OPTS)
    camellia_dir + '-release'
    build_message += 'optimized version'
print build_message

Export('env')

SConscript('SConscript')

SConscript('drivers/NavierStokes/SConscript')

