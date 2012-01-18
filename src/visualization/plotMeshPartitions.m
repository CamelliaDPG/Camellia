clear
MeshPartitions;
colorMat = hsv(numPartitions);
for i = 1:numPartitions
   patch(verts{i,1},verts{i,2},colorMat(i,:)); 
end
axis off