function plotConvergence(A,name,best)
% % First order solution
% A{1} = [1 0.015625000000000 0.009092198566947
% 1 0.007812500000000   0.001941095878590
% 1 0.003906250000000   0.000472314346570
% 1 0.001953125000000   0.000115962509411];
% % Second order solution
% A{2} = [2 0.015625000000000 0.001529334669987
% 2 0.007812500000000   0.000358791388418
% 2 0.003906250000000   0.000042735296594
% 2 0.001953125000000   0.000005086547954];
% % Third order solution 
% A{3} = [3 0.015625000000000 6.590037046239085e-04
% 3 0.007812500000000   0.000049080023736
% 3 0.003906250000000   0.000002413575104
% 3 0.001953125000000   0.000000150814174];
% % Fourth order solution
% A{4} = [4 0.015625000000000 1.326513627798934e-04
% 4 0.007812500000000   0.000002402383781
% 4 0.003906250000000   0.000000169556135
% 4 0.001953125000000   0.000000005378503];
% % Fifth order solution
% A{5} = [5 0.015625000000000 1.465040597678935e-05
% 5 0.007812500000000   0.000000956316574
% 5 0.003906250000000   0.000000013931646
% 5 0.001953125000000   0.000000000196333];
%%%%%%%%%%%%%%%------------h convergence----------------------------
N = length(A);
order = zeros(N,1);
figure
axes('fontsize',14);
hold on;
kValues = zeros(N,1); % will hold 1,2,3,...
min_x = 1;
max_x = 1;
for n=1:N
    kValues(n) = n;
  h = A{n}(:,2); err = sqrt(sum(A{n}(:,3:end).^2,2));
  bestErr = sqrt(sum(best{n}(:,3:end).^2,2));
  % slope
  order(n) = (log(err(end))-log(err(end-1)))/(log(h(end))-log(h(end-1)));
  loglog(h,err,'linewidth',1.0)
  if (nargin>2)
      loglog(h,bestErr,'--');
  end
  
  % Draw the solution order
  text(h(1)+5.e-2,err(1),strcat('k = ',num2str(A{n}(1,1))),'FontSize',14)
  
  % drawing the slope, working between log and linear scale
  x1 = h(end); x2 = h(end-1)-0.5*(h(end-1)-h(end));
  y1 = err(end) + err(end)*0.2;
  b = log(y1) - order(n)*log(x1);
  y2 = exp(order(n)*log(x2) + b);
  x3 = x1;
  y3 = y2;
  px = [x1,x2,x3,x1];
  py = [y1,y2,y3,y1];
  loglog(px,py,'r','linewidth',1.0)
  
  % plot the order of convergence
  xm = x1; xm = xm - 0.2*xm;
  min_x = min(min_x,xm);
  ym = y1 + 0.3*(y3-y1); 
  text(xm,ym,num2str(order(n),3),'FontSize',14)
  
  xm = x1 + 0.3*(x2-x1);
  ym = y2 + 0.5*y2; 
  text(xm,ym,'1','FontSize',14)
end
min_x = min_x - 0.2 * min_x;
xlim([min_x max_x]);
%xlim([0.04 0.5])
%axis([0.02 0.45 1.e-10 1])
set(gca,'XScale','log')
set(gca,'YScale','log')
xlabel('h'); ylabel('error in L^2-norm');
legend('actual','best','Location','SouthEast');
print(gcf,'-depsc',strcat(name, '_h'))
print(gcf,'-dpng',strcat(name, '_h'))