function h = plotSoln(filepath,fluxFilepath)

extLocation = strfind(filepath,'.m');
if isempty(extLocation)
    disp('Error: filepath specified is not of .m extension.')
    return
end
filepath = filepath(1:extLocation-1);% cut out extension 

run(filepath);
h = figure;hold on;
thresh = 5000; % if we have more than 3000 cells
if (numCells<thresh)
    for i=1:numCells        
        surf(x{i},y{i},z{i}')
    end
else
    for i=1:numCells
        pcolor(x{i},y{i},z{i}')
    end

end
colorbar
shading interp;
if (nargin>1)
    A=load(fluxFilepath);
    epsilon = .05*max(A(:,3));  
    plot3(A(:,1),A(:,2),A(:,3)+epsilon,'k','linewidth',1);
end
