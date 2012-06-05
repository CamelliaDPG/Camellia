Import('*')

# Library locations
Mumps_Dir = '/workspace/jchan/MUMPS_4.9.2/'
Trilinos_Dir = '/workspace/jchan/trilinos_builds/mpi_release/'
Scalapack_lib = '/workspace/jchan/lib/scalapack'

# Append path locations for libraries
# Camellia
env.Append(CPPPATH = [Dir('src/include'), Dir('drivers/DPGTests')])
env.Append(LIBPATH = [Dir('build/lib')])
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

# Build Camellia Library
src_obj = env.Object(Glob('src/*/*.cpp'))
camellia = env.Library('build/lib/Camellia', src_obj)
# Default(camellia)
