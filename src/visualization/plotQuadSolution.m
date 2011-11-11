function plotSolution(filePath)

figure
view(3)
hold on;

solnValues = load(filePath); % format: cellIndex xIndex yIndex x y z

numRows = size(solnValues,1);

i=1
cellNumber=0;
while (i < numRows)
  start_i = i;
  cellID = solnValues(i,1);
  cellNumber = cellNumber + 1;
  %let's count the number of points we have for this element
  numPoints = 0;
  currentCell = cellID;
  while currentCell==cellID && i < numRows
    i=i+1;
    currentCell = solnValues(i,1);
  end
  maxXIndex = max(solnValues(start_i:i-1,2));
  maxYIndex = max(solnValues(start_i:i-1,3));
  X = zeros(maxXIndex,1);
  Y = zeros(maxYIndex,1);
  Z = zeros(maxXIndex,maxYIndex);

  % repeat the above loop, filling in X, Y, Z values
  i = start_i;
  currentCell = cellID;
  while currentCell==cellID && i <= numRows
    xIndex = solnValues(i,2)+1;
    yIndex = solnValues(i,3)+1;
    x = solnValues(i,4);
    y = solnValues(i,5);
    z = solnValues(i,6);
    X(xIndex,1) = x;
    Y(yIndex,1) = y;
    Z(xIndex,yIndex) = z;
    i=i+1;
    if i <= numRows
      currentCell = solnValues(i,1);
    end
  end
  surf(X,Y,Z','EdgeAlpha',0,'FaceColor','interp');
end

view(10,20)