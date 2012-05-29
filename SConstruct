Help("""
       Type: 'scons' to build the production library,
             'scons -c' to clean the build,
             'scons debug=1' to build the debug version,
             'scons parallel=1' to build the parallel version (default).
       """)

import os

env = Environment(ENV=os.environ)

# Library locations
Mumps_Dir = '/workspace/jchan/MUMPS_4.9.2/'
Trilinos_Dir = '/workspace/jchan/trilinos_builds/mpi_release/'
Scalapack_lib = '/workspace/jchan/lib/scalapack'

# Compiler Settings
CC_OPTS     = '-O3'
DEBUG_OPTS  = '-g -Wall'
CPP_COMP    = 'mpicxx'
C_COMP      = 'mpicc'

env['CXX'] = CPP_COMP
env['CC'] = C_COMP

build_dir = 'build/'

# Serial version
parallel = int(ARGUMENTS.get('parallel', 1))
if parallel:
    build_dir = build_dir + 'mpi'
    print 'Building parallel, '
else:
    env.Replace(CXX = 'cpp')
    build_dir = build_dir + 'serial'
    print 'Building serial, '

# Debug options
debug = int(ARGUMENTS.get('debug', 0))
if debug:
    env.Append(CCFLAGS = DEBUG_OPTS)
    build_dir = build_dir + '-debug'
    print 'debug version'
else:
    env.Append(CCFLAGS = CC_OPTS)
    build_dir = build_dir + '-release'
    print 'optimized version'

# Append path locations for libraries
# Camellia
env.Append(CPPPATH = [Dir('src/include')])
env.Append(LIBPATH = [build_dir+'/lib'])
env.Append(LIBS = 'Camellia')

# Trilinos
TrilinosLibs = ['intrepid', 'ml', 'ifpack', 'pamgen_extras', 'pamgen', 
    'amesos', 'galeri', 'aztecoo', 'isorropia', 'epetraext', 'tpetraext', 
    'tpetrainout', 'tpetra', 'triutils', 'shards', 'zoltan', 'epetra', 
    'kokkoslinalg', 'kokkosnodeapi', 'kokkos', 'sacado', 'tpi', 'teuchos', 
    'lapack','blas','pthread']
env.Append(CPPPATH = [Trilinos_Dir+'include'])
env.Append(LIBPATH = [Trilinos_Dir+'lib'])
env.Append(LIBS = TrilinosLibs)

# Mumps
MumpsLibs = ['dmumps', 'mumps_common', 'pord', 'scalapack']
env.Append(CPPPATH = [Mumps_Dir+'include', Mumps_Dir+'PORD/include'])
env.Append(LIBPATH = [Mumps_Dir+'lib', Mumps_Dir+'PORD/lib', Scalapack_lib])
env.Append(LIBS = MumpsLibs)

Export('env', 'debug', 'parallel')
SConscript('src/SConscript', variant_dir=build_dir, duplicate=0)
