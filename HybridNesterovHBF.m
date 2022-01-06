%--------------------------------------------------------------------------
% Matlab M-file Project: HyEQ Toolbox @  Hybrid Systems Laboratory (HSL), 
% https://hybrid.soe.ucsc.edu/software
% http://hybridsimulator.wordpress.com/
% Filename: HybridNessterovHBF.m
%--------------------------------------------------------------------------
% Project: Uniting Nesterov's accelerated gradient descent globally with
% heavy ball locally. Nonstrongly convex version.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%   See also HYEQSOLVER, PLOTARC, PLOTARC3, PLOTFLOWS, PLOTHARC,
%   PLOTHARCCOLOR, PLOTHARCCOLOR3D, PLOTHYBRIDARC, PLOTJUMPS.
%   Copyright @ Hybrid Systems Laboratory (HSL),
%   Revision: 0.0.0.3 Date: 09/29/2020 1:07:00

clear all

set(0,'defaultTextInterpreter','tex');

% global variables
global delta M gamma lambda c_0 c_10 r tauMin tauMax tauMed c zeta cTilde_0 cTilde_10 d_0 d_10 alpha

%%%%%%%%%%%%%%%%%%%%%
% setting the globals
%%%%%%%%%%%%%%%%%%%%%
setMinima();

% Nesterov constants
M = 2;
zeta = 2;

% Heavy Ball constants
gamma = 2/3; % 
lambda = 200; % 

% Uniting parameters for \mathcal{U}_0 and \mathcal{T}_{1,0}:
c_0 = 7000; % \mathcal{U}_0 
c_10 = 6819.67593; % \mathcal{T}_{1,0} 

% These are the same, since L = x^2
alpha = 1;

% eps_0 has to be bigger than eps_10
eps_0 = 10;
eps_10 = 5; % 5 10 12 15

cTilde_0 = eps_0*alpha
cTilde_10 = eps_10*alpha
d_0 = c_0 - gamma*((cTilde_0^2)/alpha)
d_10 = c_10 - (cTilde_10/alpha)^2 - (zeta^2/(M))*((cTilde_10^2)/alpha)

tauMin = (1+sqrt(7))/2;

c = 0.5;
deltaMed = 3300; % 10700 3300
r = 51; 

delta = 0.5;

%%%%%%%%%%%%%%%%%%%%%
% setting the locals
%%%%%%%%%%%%%%%%%%%%%

deltaVecUniting = [0,0,0];
deltaVec = [0,0,0];
deltaVecHBF = [0,0,0];
deltaVecUniting = [0,0,0];

lDeltaNest = 0;
lDeltaHBF = 0;
lDeltaUniting = 0;
lDelta = 0;

% initial conditions
z1_0 = 50; 
z2_0 = 0;
z2_00 = 50;
q_0 = 1;
tau_0 = 0;
tauPN_0 = tauMin; 

tauMed = sqrt(((r^2)/(2*c) + (tauMin^2)*CalculateL(z1_0))/deltaMed) + tauMin
tauMax = tauMed + 1;

% Assign initial conditions to vector
x0 = [z1_0;z2_0;q_0;tau_0];
x00 = [z1_0;z2_00;tauPN_0];
x000 = [z1_0;z2_0;q_0];

% simulation horizon
TSPAN_HBF=[0 700];
TSPAN=[0 10];
JSPAN = [0 20000];

% rule for jumps
% rule = 1 -> priority for jumps
% rule = 2 -> priority for flows
rule = 1;

options = odeset('RelTol',1e-6,'MaxStep',.01);

% simulate
[tNest,jNest,xNest] = HyEQsolver(@fNesterov,@gNesterov,@CNesterov,@DNesterov,...
    x0,TSPAN,JSPAN,rule,options);

% Find the L values for Nesterov:
lNest = zeros(1,length(tNest));
[deltaVecNest lNest lDeltaNest] = timeToConv(xNest,tNest);

lNestAvg = lNest.';
% This is the dotted line indicating the average value:
PO = polyfit(tNest,log10(lNestAvg(:,1)),1);
yO = polyval(PO,tNest);

[tHBF,jHBF,xHBF] = HyEQsolver(@fHBF,@gHBF,@CHBF,@DHBF,...
    x000,TSPAN_HBF,JSPAN,rule,options);

% Find the L values for HBF:
lHBF = zeros(1,length(tHBF));
[deltaVecHBF lHBF lDeltaHBF] = timeToConv(xHBF,tHBF);

deltaVecHBF = timeToConv(xHBF,tHBF);

[tUniting,jUniting,xUniting] = HyEQsolver(@fU,@gU,@CU,@DU,...
    x0,TSPAN,JSPAN,rule,options);

% Find the L values for Uniting:
lUniting = zeros(1,length(tUniting));
[deltaVecUniting lUniting lDeltaUniting] = timeToConv(xUniting,tUniting);

[t,j,x] = HyEQsolver(@f,@g,@C,@D,...
    x00,TSPAN,JSPAN,rule,options);

% Find the L values for HAND-1:
lHAND = zeros(1,length(t));
[deltaVec lHAND lDelta] = timeToConv(x,t);

lHANDAvg = lHAND.';
% This is the dotted line indicating the average value:
PH = polyfit(t,log10(lHANDAvg(:,1)),1);
yH = polyval(PH,t);

minarc = min([length(x),length(xUniting)]);
ta = [tUniting(1:minarc),t(1:minarc)];
ja = [jUniting(1:minarc),j(1:minarc)];
xa = [xUniting(1:minarc,1),x(1:minarc,1)];
xb = [xUniting(1:minarc,2),x(1:minarc,2)];

figure(1) 
clf
modificatorF{1} = '';
modificatorF{2} = 'LineWidth';
modificatorF{3} = 1.5;
modificatorJ{1} = '*--';
modificatorJ{2} = 'LineWidth';
modificatorJ{3} = 1.5;
subplot(2,1,1), plotHarc(ta,ja,xa,[],modificatorF,modificatorJ);
hold on
plot(deltaVec(3),deltaVec(1),'k.','MarkerSize', 20)
strDelta = [num2str(deltaVec(3)), 's'];
text(deltaVec(3),deltaVec(1),strDelta,'HorizontalAlignment','left','VerticalAlignment','top');
plot(deltaVecUniting(3),deltaVecUniting(1),'k.','MarkerSize', 20)
strDeltaUniting = [num2str(deltaVecUniting(3)), 's'];
text(deltaVecUniting(3),deltaVecUniting(1),strDeltaUniting,'HorizontalAlignment','left','VerticalAlignment','bottom');
axis([0 10 -20 80])
grid on
ylabel('z_1','Fontsize',16)
xlabel('t','Fontsize',16)
hold off
subplot(2,1,2), plotHarc(ta,ja,xb,[],modificatorF,modificatorJ);
hold on
plot(deltaVec(3),deltaVec(2),'k.','MarkerSize', 20)
plot(deltaVecUniting(3),deltaVecUniting(2),'k.','MarkerSize', 20)
axis([0 10 -50 70])
grid on
ylabel('z_2','Fontsize',16)
xlabel('t','Fontsize',16)
hold off
saveas(gcf,'Plots\ComparisonPlotsNSC','png')

minarc = min([length(xUniting),length(x),length(xHBF),length(xNest)]);
tc = [tUniting(1:minarc),t(1:minarc),tHBF(1:minarc),tNest(1:minarc)];
jc = [jUniting(1:minarc),j(1:minarc),jHBF(1:minarc),jNest(1:minarc)];
xc = [xUniting(1:minarc,1),x(1:minarc,1),xHBF(1:minarc,1),xNest(1:minarc,1)];
xd = [xUniting(1:minarc,2),x(1:minarc,2),xHBF(1:minarc,2),xNest(1:minarc,2)];

figure(2)
clf
modificatorF{1} = '';
modificatorF{2} = 'LineWidth';
modificatorF{3} = 1.5;
modificatorJ{1} = '*--';
modificatorJ{2} = 'LineWidth';
modificatorJ{3} = 1.5;
subplot(2,1,1), plotHarc(tc,jc,xc,[],modificatorF,modificatorJ);
hold on
plot(deltaVec(3),deltaVec(1),'k.','MarkerSize', 10)
strDelta = [num2str(deltaVec(3)), 's'];
text(deltaVec(3),deltaVec(1),strDelta,'HorizontalAlignment','left','VerticalAlignment','bottom');
plot(deltaVecUniting(3),deltaVecUniting(1),'k.','MarkerSize', 10)
strDeltaUniting = [num2str(deltaVecUniting(3)), 's'];
text(deltaVecUniting(3),deltaVecUniting(1),strDeltaUniting,'HorizontalAlignment','left','VerticalAlignment','bottom');
plot(deltaVecNest(3),deltaVecNest(1),'k.','MarkerSize', 20)
strDeltaNest = [num2str(deltaVecNest(3)), 's'];
text(deltaVecNest(3),deltaVecNest(1),strDeltaNest,'HorizontalAlignment','left','VerticalAlignment','bottom');
axis([0 10 -20 80])
grid on
ylabel('z_1','Fontsize',16)
xlabel('t','Fontsize',16)
axes('Position',[0.7 0.78 0.15 0.08])
box on
hold on
plot(tHBF,xHBF(:,1),'LineWidth',3)
plot(deltaVecHBF(3),deltaVecHBF(1),'k.','MarkerSize', 20)
strDeltaHBF = [num2str(deltaVecHBF(3)), 's'];
text(deltaVecHBF(3),deltaVecHBF(1),strDeltaHBF,'HorizontalAlignment','left','VerticalAlignment','bottom');
hold off
set(gca,'xtick',[0 100 200])
set(gca,'ytick',[-20 25 70])
axis([0 200 -20 70])
grid on
hold off
subplot(2,1,2), plotHarc(tc,jc,xd,[],modificatorF,modificatorJ);
hold on
plot(deltaVec(3),deltaVec(2),'k.','MarkerSize', 10)
plot(deltaVecHBF(3),deltaVecHBF(2),'k.','MarkerSize', 20)
plot(deltaVecNest(3),deltaVecNest(2),'k.','MarkerSize', 20)
plot(deltaVecUniting(3),deltaVecUniting(2),'k.','MarkerSize', 10)
axis([0 10 -50 70])
grid on
ylabel('z_2','Fontsize',16)
xlabel('t','Fontsize',16)
hold off
saveas(gcf,'Plots\ComparisonPlots2','png')
saveas(gcf,'Plots\ComparisonPlots2','epsc')

figure(3)
clf
semilogy(tUniting,lUniting,'LineWidth',3);
hold on
semilogy(tHBF,lHBF,'Color',[0.4660 0.6740 0.1880],'LineWidth',3);
semilogy(tNest,lNest,'Color',[0.6350 0.0780 0.1840],'LineWidth',3,'LineStyle','--');
semilogy(tNest,10.^(yO(:,1)),'k--','LineWidth',3);
semilogy(t,lHAND,'Color',[0.8500 0.3250 0.0980],'LineWidth',3);
semilogy(t,10.^(yH(:,1)),'k:','LineWidth',3);

% Plotting times to convergence:
semilogy(deltaVec(3),lDelta,'k.','MarkerSize', 20)
strDelta = [num2str(deltaVec(3)), 's'];
text(deltaVec(3),lDelta,strDelta,'HorizontalAlignment','left','VerticalAlignment','bottom');
semilogy(deltaVecUniting(3),lDeltaUniting,'k.','MarkerSize', 20)
strDeltaUniting = [num2str(deltaVecUniting(3)), 's'];
text(deltaVecUniting(3),lDeltaUniting,strDeltaUniting,'HorizontalAlignment','left','VerticalAlignment','bottom');
semilogy(deltaVecNest(3),lDeltaNest,'k.','MarkerSize', 20)
strDeltaNest = [num2str(deltaVecNest(3)), 's'];
text(deltaVecNest(3),lDeltaNest,strDeltaNest,'HorizontalAlignment','left','VerticalAlignment','bottom');

hold off
axis([0 10 10^(-30) 10^(6)]);
ylabel('L(z_1)-L^*','FontSize',20)
xlabel('t','FontSize',20)
legend({'Hybrid','Heavy ball','Nesterov','Nesterov, average','HAND-1','HAND-1, average'},'Location','southwest')

axes('Position',[0.7 0.6 0.15 0.1])
box on
hold on
semilogy(tHBF,lHBF,'Color',[0.4660 0.6740 0.1880],'LineWidth',3);
semilogy(deltaVecHBF(3),lDeltaHBF,'k.','MarkerSize', 20)
strDeltaHBF = [num2str(deltaVecHBF(3)), 's'];
text(deltaVecHBF(3),lDeltaHBF,strDeltaHBF,'HorizontalAlignment','right','VerticalAlignment','bottom');
hold off
set(gca,'xtick',[0 350 700])
set(gca,'ytick',[10^(-2) 10^(2) 10^(6)])
axis([0 700 10^(-2) 10^(6)])
hold off

saveas(gcf,'Plots\Semilog','epsc')
saveas(gcf,'Plots\Semilog','png')
