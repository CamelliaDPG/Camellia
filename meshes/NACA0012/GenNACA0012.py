from pylab import *

f = open('NACA0012.poly', 'w')

nseg = 32

xs = [-1, 2, 2, -1]
ys = [-1, -1, 1, 1]

t = .12
chord = 1

xa = linspace(0, chord, nseg)
ya = t*chord/0.2*(0.2969*sqrt(xa/chord)
    -0.1260*xa/chord
    -0.3516*(xa/chord)**2
    +0.2843*(xa/chord)**3
    -0.1015*(xa/chord)**4)

# plot(xa,ya,xa,-ya)
# axis('equal')
# 
# show()

npts = 2*nseg+4
# Vertices
f.write(str(npts)+' 2 0 0\n')
c = 1
for i in range(0,4):
  f.write(str(c)+' '+str(xs[i])+' '+str(ys[i])+'\n')
  c = c+1
for i in range(0,nseg):
  f.write(str(c)+' '+str(xa[i])+' '+str(ya[i])+'\n')
  c = c+1
for i in range(nseg-1,-1,-1):
  f.write(str(c)+' '+str(xa[i])+' '+str(-ya[i])+'\n')
  c = c+1
f.write(str(npts)+' 1\n')
# Connectivity
c = 1
for i in range(0,3):
  f.write(str(c)+' '+str(c)+' '+str(c+1)+' 1\n')
  c = c+1
f.write(str(c)+' '+str(c)+' '+str(1)+' 1\n')
c = c+1
for i in range(0,2*nseg-1):
  f.write(str(c)+' '+str(c)+' '+str(c+1)+' 1\n')
  c = c+1
f.write(str(c)+' '+str(c)+' '+str(5)+' 1\n')
# Holes
f.write('1\n')
f.write('1 0.5 0\n')
f.close()
