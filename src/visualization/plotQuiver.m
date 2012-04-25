function h = plotQuiver(filepath1,filepath2)

extLocation1 = strfind(filepath1,'.m');
extLocation2 = strfind(filepath2,'.m');
filepath1 = filepath1(1:extLocation1-1);% cut out extension 
filepath2 = filepath2(1:extLocation2-1);% cut out extension 

run(filepath1);
z1 = z;
run(filepath2);

h = figure;hold on;
thresh = 5000; % if we have more than 3000 cells
for i=1:numCells
    quiver(x{i},y{i},z1{i}',z{i}','b')
end
 