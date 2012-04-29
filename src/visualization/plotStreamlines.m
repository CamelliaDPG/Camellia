function h = plotStreamlines(filepath)

extLocation1 = strfind(filepath,'.m');
filepath = filepath(1:extLocation1-1);% cut out extension 

run(filepath);

h = figure;hold on;
options = [0.0001,5];
[sx,sy] = meshgrid(0:.02:.1,0:.02:.1,options);
streamline(X,Y,U',V',sx,sy);
[sx,sy] = meshgrid(.9:.02:1.0,0:.02:.1,options);
streamline(X,Y,U',V',sx,sy);
[sx,sy] = meshgrid(0.5:.5,0:.1:1.0,options);
streamline(X,Y,U',V',sx,sy);
