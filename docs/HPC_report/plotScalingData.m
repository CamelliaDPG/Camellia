close all
clear
load hsfcMats
load cyclicMats
cyclic_assemble = [hsfc_assemble(:,1) cyclic_assemble];
cyclic_local = [hsfc_local(:,1) cyclic_local];
cyclic_solve = [hsfc_solve(:,1) cyclic_solve];

%% weak scaling plots
figure;hold on
plot(1:length(numMpiProcs),diag(cyclic_assemble),'r*-','linewidth',2)
plot(1:length(numMpiProcs),diag(hsfc_assemble),'bo-','linewidth',2)
legend('Cyclic partitioning','HSFC partitioning')
ylim([0,.7])
title('Global stiffness matrix assembly time','fontsize',14)
%set(gca,'XtickLabel',['4^0';'4^1';'4^2';'4^3'])
xtickVec =['1 ';'  ';' 4';'  ';'16';'  ';'64'];
set(gca,'XTickLabel',xtickVec)
xlabel('MPI nodes used','fontsize',14)
ylabel('Runtime (in seconds)','fontsize',14)
grid on

figure;hold on
plot(1:length(numMpiProcs),diag(cyclic_local),'r*-','linewidth',2)
plot(1:length(numMpiProcs),diag(hsfc_local),'bo-','linewidth',2)
legend('Cyclic partitioning','HSFC partitioning')
ylim([0,ceil(max(diag(cyclic_local)))])
title('Local stiffness matrix computation time','fontsize',14)
%set(gca,'XTick',numMpiProcs)
set(gca,'XTickLabel',xtickVec)
xlabel('MPI nodes used','fontsize',14)
ylabel('Runtime (in seconds)','fontsize',14)

grid on

%% strong scaling plots
numInitElems = 202; % from data file
figure(3);
figure(4);

figure(5);
figure(6);

cmat = hsv(4);
for numUnifRefs = 0:3
    figure(3)
    semilogy(1:3,hsfc_assemble(numUnifRefs+1,(2:end)),'*-','color',cmat(numUnifRefs+1,:),'linewidth',2);
    hold on
    figure(4)
    semilogy(1:3,cyclic_assemble(numUnifRefs+1,(2:end)),'*-','color',cmat(numUnifRefs+1,:),'linewidth',2);
    hold on
    figure(5)
    semilogy(1:3,hsfc_local(numUnifRefs+1,(2:end)),'*-','color',cmat(numUnifRefs+1,:),'linewidth',2);
    hold on
    figure(6)
    semilogy(1:3,cyclic_local(numUnifRefs+1,(2:end)),'*-','color',cmat(numUnifRefs+1,:),'linewidth',2);
    hold on
end
xtickVec =[' 4';'  ';'16';'  ';'64';'  '];
xticks =1:.5:3;
figure(3)
legend('202 elements','808 elements','3232 elements','12928 elements')
title('Global stiffness matrix assembly for HSFC partitioning','fontsize',14)
set(gca,'XTick',xticks)
set(gca,'XTickLabel',xtickVec)
xlabel('MPI nodes used','fontsize',14)
ylabel('Runtime (in seconds)','fontsize',14)
grid on

figure(4)
legend('202 elements','808 elements','3232 elements','12928 elements')
title('Global stiffness matrix assembly for cyclic partitioning','fontsize',14)
set(gca,'XTick',xticks)
set(gca,'XTickLabel',xtickVec)
xlabel('MPI nodes used','fontsize',14)
ylabel('Runtime (in seconds)','fontsize',14)
grid on

figure(5)
legend('202 elements','808 elements','3232 elements','12928 elements')
title('Local stiffness matrix computation for HSFC partitioning','fontsize',14)
set(gca,'XTick',xticks)
set(gca,'XTickLabel',xtickVec)
xlabel('MPI nodes used','fontsize',14)
ylabel('Runtime (in seconds)','fontsize',14)
grid on

figure(6)
legend('202 elements','808 elements','3232 elements','12928 elements')
title('Local stiffness matrix computation for cyclic partitioning','fontsize',14)
set(gca,'XTick',xticks)
set(gca,'XTickLabel',xtickVec)
xlabel('MPI nodes used','fontsize',14)
ylabel('Runtime (in seconds)','fontsize',14)
grid on

%% bar graphs
numDofs = [13343 52137 206081 819393];
for i=1:4
    leftover = hsfc_wall_time(i,:) - hsfc_local(i,:) - hsfc_assemble(i,:) - hsfc_solve(i,:);
    barMat = [hsfc_local(i,:)' hsfc_assemble(i,:)' hsfc_solve(i,:)' leftover'];
    figure(7+i-1)
    bar(barMat,'stacked')
    legend('Local stiffness matrix computation', 'Global stiffness matrix assembly','Solve','Other')
    title(['Time spent solving on ' num2str(202*4^(i-1)) ' elements, ' num2str(numDofs(i)) ' dofs'],'fontsize',14)
    set(gca,'XTick',1:4)
    set(gca,'XTickLabel',[' 1';' 4';'16';'64'])
    xlabel('MPI nodes used','fontsize',14)
    ylabel('Runtime (in seconds)','fontsize',14)
end

keyboard % put in debug mode, adjust figures here if necessary, then dbcont to continue to end

nameVector{1} = 'weakScalingAssembly.pdf'
nameVector{2} = 'weakScalingLocal.pdf'
nameVector{3} = 'hsfcStrongScalingAssembly.pdf'
nameVector{4} = 'cyclicStrongScalingAssembly.pdf'
nameVector{5} = 'hsfcStrongScalingLocal.pdf'
nameVector{6} = 'cyclicStrongScalingLocal.pdf'

nameVector{7} = 'bar_ref0.pdf'
nameVector{8} = 'bar_ref1.pdf'
nameVector{9} = 'bar_ref2.pdf'
nameVector{10} = 'bar_ref3.pdf'

for i = 1:10 %number of figures    
    h = figure(i);
    set(gca,'units','centimeters')
    pos = get(gca,'Position');
    ti = get(gca,'TightInset')+.5; % .5 just makes it look better
    set(gcf, 'PaperUnits','centimeters');
    set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
    print(h,['scalingFigs/' nameVector{i}],'-dpdf')
end

close all
clear