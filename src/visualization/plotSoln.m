function plotSoln(filepath,fluxFilepath)

extLocation = strfind(filepath,'.m');
if isempty(extLocation)
    disp('Error: filepath specified is not of .m extension.')
    return
end
filepath = filepath(1:extLocation-1);% cut out extension 

run(filepath);
figure;hold on;
for i=1:numCells
    surf(x{i},y{i},z{i}')
end
colorbar
shading interp;
if (nargin>1)
    A=load(fluxFilepath);
    plot3(A(:,1),A(:,2),A(:,3),'k','linewidth',1.5);
end
