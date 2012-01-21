function plotSoln(filepath,fluxFilepath)
run(filepath);
figure;hold on;
for i=1:numCells
    surf(x{i},y{i},z{i}')
end
shading interp;
if (nargin>1)
    A=load(fluxFilepath);
    plot3(A(:,1),A(:,2),A(:,3),'r');
end
