from pylab import *

f = open('HighLift.poly', 'w')

nseg = 32

xs = [-1, 2, 2, -1]
ys = [-1, -1, 1, 1]

SlatData = loadtxt('SlatData.txt').flatten()[0:-2]
WingData = loadtxt('WingData.txt').flatten()[0:-4]
FlapData = loadtxt('FlapData.txt').flatten()[0:-2]
SlatX = SlatData[0::2]
SlatY = SlatData[1::2]
WingX = WingData[0::2]
WingY = WingData[1::2]
FlapX = FlapData[0::2]
FlapY = FlapData[1::2]

npts = SlatX.size + WingX.size + FlapX.size + 4
# Vertices
f.write(str(npts)+' 2 0 0\n')
c = 1
# Outer Rectangle
for i in range(0,4):
  f.write(str(c)+' '+str(xs[i])+' '+str(ys[i])+'\n')
  c = c+1
# Slat
slat_start = c
for i in range(0,SlatX.size):
  f.write(str(c)+' '+str(SlatX[i])+' '+str(SlatY[i])+'\n')
  c = c+1
# Wing
wing_start = c
for i in range(0,WingX.size):
  f.write(str(c)+' '+str(WingX[i])+' '+str(WingY[i])+'\n')
  c = c+1
# Flap
flap_start = c
for i in range(0,FlapX.size):
  f.write(str(c)+' '+str(FlapX[i])+' '+str(FlapY[i])+'\n')
  c = c+1
f.write(str(npts)+' 1\n')
# Connectivity
c = 1
# Outer Rectangle
for i in range(0,3):
  f.write(str(c)+' '+str(c)+' '+str(c+1)+' 1\n')
  c = c+1
f.write(str(c)+' '+str(c)+' '+str(1)+' 1\n')
c = c+1
# Slat
for i in range(0,SlatX.size-1):
  f.write(str(c)+' '+str(c)+' '+str(c+1)+' 1\n')
  c = c+1
f.write(str(c)+' '+str(c)+' '+str(slat_start)+' 1\n')
c = c+1
# Wing
for i in range(0,WingX.size-1):
  f.write(str(c)+' '+str(c)+' '+str(c+1)+' 1\n')
  c = c+1
f.write(str(c)+' '+str(c)+' '+str(wing_start)+' 1\n')
c = c+1
# Flap
for i in range(0,FlapX.size-1):
  f.write(str(c)+' '+str(c)+' '+str(c+1)+' 1\n')
  c = c+1
f.write(str(c)+' '+str(c)+' '+str(flap_start)+' 1\n')
c = c+1
# Holes
f.write('3\n')
f.write('1 -0.05 -0.05\n')
f.write('2 0.5 0\n')
f.write('3 1 -0.04\n')
f.close()

plot(SlatX,SlatY)
plot(WingX,WingY)
plot(FlapX,FlapY)
axis('equal')
show()
