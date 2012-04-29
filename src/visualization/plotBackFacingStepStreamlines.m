function h = plotBackFacingStepStreamlines(filepath)

extLocation1 = strfind(filepath,'.m');
filepath = filepath(1:extLocation1-1);% cut out extension 

run(filepath);

options = [1,1000];
[sx,sy] = meshgrid(0:0,1:.0404:2,options);
[sx2,sy2] = meshgrid(4.00:.081:4.5,0:.0404:.6,options);
h = figure;hold on;
streamline(X,Y,U',V',sx,sy);
streamline(X,Y,U',V',sx2,sy2);