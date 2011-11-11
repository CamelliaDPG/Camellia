function plotSolution(filePath)

figure
view(3)
hold on;

solnValues = load(filePath); % format: cellIndex patchIndex x y z

numRows = size(solnValues,1);

i=1;
cellNumber=0;
while (i < numRows)
  start_i = i;
  cellID = solnValues(i,1);
  patchID = solnValues(i,2);
  cellNumber = cellNumber + 1;

  currentCell = cellID;
  currentPatch = patchID;
  numPatches = 0;
  verticesPerPatch = 0;
  while currentCell==cellID && i < numRows
    while currentPatch == patchID && currentCell==cellID && i < numRows
        i=i+1;
        currentCell = solnValues(i,1);
        currentPatch = solnValues(i,2);
    end
    if verticesPerPatch == 0
        if i < numRows
            verticesPerPatch = i - start_i;
        else
            verticesPerPatch = i - start_i + 1;
        end
    end
    numPatches = numPatches + 1;
    patchID = solnValues(i,2);
        
    %currentCell = solnValues(i,1);
  end
  X = zeros(verticesPerPatch,numPatches);
  Y = zeros(verticesPerPatch,numPatches);
  Z = zeros(verticesPerPatch,numPatches);
  patch_start_i = start_i;
  for patchIndex=1:numPatches
    X(1:verticesPerPatch,patchIndex) = solnValues(patch_start_i:patch_start_i+verticesPerPatch-1,3);
    Y(1:verticesPerPatch,patchIndex) = solnValues(patch_start_i:patch_start_i+verticesPerPatch-1,4);
    Z(1:verticesPerPatch,patchIndex) = solnValues(patch_start_i:patch_start_i+verticesPerPatch-1,5);
    patch_start_i = patch_start_i + verticesPerPatch;
  end
  patch(X,Y,Z,Z);
  %patch(X,Y,Z,Z,'EdgeColor','none');
end

view(10,20)