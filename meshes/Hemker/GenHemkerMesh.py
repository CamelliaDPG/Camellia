from pylab import *

f = open('Hemker.poly', 'w')

nseg = 32

xs = [-3, 9, 9, -3]
ys = [-3, -3, 3, 3]

angle = linspace(0,2*pi*(nseg-1)/nseg,nseg)

xc = cos(angle)
yc = sin(angle)

npts = nseg + 4
# Vertices
f.write(str(npts)+' 2 0 0\n')
c = 1
for i in range(0,4):
  f.write(str(c)+' '+str(xs[i])+' '+str(ys[i])+'\n')
  c = c+1
for i in range(0,nseg):
  f.write(str(c)+' '+str(xc[i])+' '+str(yc[i])+'\n')
  c = c+1
f.write(str(npts)+' 1\n')
# Connectivity
c = 1
for i in range(0,3):
  f.write(str(c)+' '+str(c)+' '+str(c+1)+' 1\n')
  c = c+1
f.write(str(c)+' '+str(c)+' '+str(1)+' 1\n')
c = c+1
for i in range(0,nseg-1):
  f.write(str(c)+' '+str(c)+' '+str(c+1)+' 1\n')
  c = c+1
f.write(str(c)+' '+str(c)+' '+str(5)+' 1\n')
# Holes
f.write('1\n')
f.write('1 0 0\n')
f.close()
