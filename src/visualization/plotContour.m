function h = plotContour(filepath1)

extLocation1 = strfind(filepath1,'.m');
filepath1 = filepath1(1:extLocation1-1);% cut out extension 

run(filepath1);

xSize = size(X,1)
n = min(xSize,25);
stepSize=xSize/n;
levels = zeros(n,1);
for i=1:n
    levels(i) = U(i*stepSize,i*stepSize);
end
% for i=1:n/2
%     levels(2*i) = U(2*i*stepSize,2*i*stepSize);
%     levels(2*i+1) = U(2*i*stepSize,xSize-2*i*stepSize+1);
% end

h = figure;hold on;
%set(h,'LevelList',levels);
contour(X,Y,U',levels, 'b');