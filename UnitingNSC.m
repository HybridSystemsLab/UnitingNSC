clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%
% setting the variables
%%%%%%%%%%%%%%%%%%%%%%%
setMinimizer();

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
deltaMed = 50000; % 10700 3300 7800
r = 51; 

delta = 0.5;

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
TSPAN=[0 20];
JSPAN = [0 20000];

% rule for jumps
% rule = 1 -> priority for jumps
% rule = 2 -> priority for flows
rule = 1;

options = odeset('RelTol',1e-6,'MaxStep',.01);

%% simulate
[tNest,jNest,xNest] = HyEQsolver(@(x)fNesterov(x,M,zeta),@gNesterov,@CNesterov,@DNesterov,...
    x0,TSPAN,JSPAN,rule,options);

% Find the L values for Nesterov:
lNest = zeros(1,length(tNest));
[deltaVecNest lNest lDeltaNest] = timeToConv(xNest,tNest,delta);

lNestAvg = lNest.';
% This is the dotted line indicating the average value:
PO = polyfit(tNest,log10(lNestAvg(:,1)),1);
yO = polyval(PO,tNest);

[tHBF,jHBF,xHBF] = HyEQsolver(@(x)fHBF(x,lambda,gamma),@gHBF,@CHBF,@DHBF,...
    x000,TSPAN_HBF,JSPAN,rule,options);

% Find the L values for HBF:
lHBF = zeros(1,length(tHBF));
[deltaVecHBF lHBF lDeltaHBF] = timeToConv(xHBF,tHBF,delta);

[tUniting,jUniting,xUniting] = HyEQsolver(@(x)fU(x,lambda,gamma,M,zeta),@gU,@(x)CU(x,cTilde_0,cTilde_10,d_0,d_10),@(x)DU(x,gamma,M,cTilde_10,d_10,alpha,c_0),...
    x0,TSPAN,JSPAN,rule,options);

% Find the L values for Uniting:
lUniting = zeros(1,length(tUniting));
[deltaVecUniting lUniting lDeltaUniting] = timeToConv(xUniting,tUniting,delta);

[t,j,x] = HyEQsolver(@(x)f(x,c),@(x)g(x,tauMin),@(x)C(x,tauMin,tauMax),@(x)D(x,tauMed,tauMax),...
    x00,TSPAN,JSPAN,rule,options);

% Find the L values for HAND-1:
lHAND = zeros(1,length(t));
[deltaVec lHAND lDelta] = timeToConv(x,t,delta);

lHANDAvg = lHAND.';
% This is the dotted line indicating the average value:
PH = polyfit(t,log10(lHANDAvg(:,1)),1);
yH = polyval(PH,t);

minarc = min([length(x),length(xUniting)]);
ta = [tUniting(1:minarc),t(1:minarc)];
ja = [jUniting(1:minarc),j(1:minarc)];
xa = [xUniting(1:minarc,1),x(1:minarc,1)];
xb = [xUniting(1:minarc,2),x(1:minarc,2)];

%% Plots
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
axis([0 10 -80 80])
grid on
ylabel('z_1','Fontsize',16)
xlabel('t','Fontsize',16)
hold off
subplot(2,1,2), plotHarc(ta,ja,xb,[],modificatorF,modificatorJ);
hold on
plot(deltaVec(3),deltaVec(2),'k.','MarkerSize', 20)
plot(deltaVecUniting(3),deltaVecUniting(2),'k.','MarkerSize', 20)
axis([0 10 -100 70])
grid on
ylabel('z_2','Fontsize',16)
xlabel('t','Fontsize',16)
hold off
saveas(gcf,'Plots\ComparisonPlotsNSC','png')

% minarc = min([length(xUniting),length(x),length(xHBF),length(xNest)]);
% tc = [tUniting(1:minarc),t(1:minarc),tHBF(1:minarc),tNest(1:minarc)];
% jc = [jUniting(1:minarc),j(1:minarc),jHBF(1:minarc),jNest(1:minarc)];
% xc = [xUniting(1:minarc,1),x(1:minarc,1),xHBF(1:minarc,1),xNest(1:minarc,1)];
% xd = [xUniting(1:minarc,2),x(1:minarc,2),xHBF(1:minarc,2),xNest(1:minarc,2)];
% 
% figure(2)
% clf
% modificatorF{1} = '';
% modificatorF{2} = 'LineWidth';
% modificatorF{3} = 1.5;
% modificatorJ{1} = '*--';
% modificatorJ{2} = 'LineWidth';
% modificatorJ{3} = 1.5;
% subplot(2,1,1), plotHarc(tc,jc,xc,[],modificatorF,modificatorJ);
% hold on
% plot(deltaVec(3),deltaVec(1),'k.','MarkerSize', 10)
% strDelta = [num2str(deltaVec(3)), 's'];
% text(deltaVec(3),deltaVec(1),strDelta,'HorizontalAlignment','left','VerticalAlignment','bottom');
% plot(deltaVecUniting(3),deltaVecUniting(1),'k.','MarkerSize', 10)
% strDeltaUniting = [num2str(deltaVecUniting(3)), 's'];
% text(deltaVecUniting(3),deltaVecUniting(1),strDeltaUniting,'HorizontalAlignment','left','VerticalAlignment','bottom');
% plot(deltaVecNest(3),deltaVecNest(1),'k.','MarkerSize', 20)
% strDeltaNest = [num2str(deltaVecNest(3)), 's'];
% text(deltaVecNest(3),deltaVecNest(1),strDeltaNest,'HorizontalAlignment','left','VerticalAlignment','bottom');
% axis([0 10 -20 80])
% grid on
% ylabel('z_1','Fontsize',16)
% xlabel('t','Fontsize',16)
% axes('Position',[0.7 0.78 0.15 0.08])
% box on
% hold on
% plot(tHBF,xHBF(:,1),'LineWidth',3)
% plot(deltaVecHBF(3),deltaVecHBF(1),'k.','MarkerSize', 20)
% strDeltaHBF = [num2str(deltaVecHBF(3)), 's'];
% text(deltaVecHBF(3),deltaVecHBF(1),strDeltaHBF,'HorizontalAlignment','left','VerticalAlignment','bottom');
% hold off
% set(gca,'xtick',[0 100 200])
% set(gca,'ytick',[-20 25 70])
% axis([0 200 -20 70])
% grid on
% hold off
% subplot(2,1,2), plotHarc(tc,jc,xd,[],modificatorF,modificatorJ);
% hold on
% plot(deltaVec(3),deltaVec(2),'k.','MarkerSize', 10)
% plot(deltaVecHBF(3),deltaVecHBF(2),'k.','MarkerSize', 20)
% plot(deltaVecNest(3),deltaVecNest(2),'k.','MarkerSize', 20)
% plot(deltaVecUniting(3),deltaVecUniting(2),'k.','MarkerSize', 10)
% axis([0 10 -50 70])
% grid on
% ylabel('z_2','Fontsize',16)
% xlabel('t','Fontsize',16)
% hold off
% saveas(gcf,'Plots\ComparisonPlots2','png')
% saveas(gcf,'Plots\ComparisonPlots2','epsc')

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


%% UnitingAlgorithm (C,f,D,g):

% C for Uniting
function [value] = CU(x,cTilde_0,cTilde_10,d_0,d_10) 

% state
z1 = x(1);
z2 = x(2);
q = x(3);
tau = x(4);

absGradL = abs(GradientL(z1));

if(q == 0)
    halfz2Squared = (1/2)*z2^2;
elseif(q == 1)
    z2Squared = z2^2;
end

if (q == 0 && absGradL <= cTilde_0 && halfz2Squared <= d_0)||(q==1 && (absGradL >= cTilde_10 || z2Squared >= d_10 ))
    value = 1;
else
    value = 0;
end

end

% D for Uniting
function inside = DU(x,gamma,M,cTilde_10,d_10,alpha,c_0) 

% state
z1 = x(1);
z2 = x(2);
q = x(3);
tau = x(4);

absGradL = abs(GradientL(z1));

if(q == 0)
    V0 = gamma*(alpha/(M^2))*absGradL^2 + (1/2)*z2^2;
elseif(q == 1)
    z2Squared = z2^2;
end

if (q == 0 && V0 >= c_0)||(q == 1 && absGradL <= cTilde_10 && z2Squared <= d_10)  
    inside = 1;
else
    inside = 0;
end

end

% f for Uniting
function xdot = fU(x,lambda,gamma,M,zeta)

% state
z1 = x(1);
z2 = x(2);
q = x(3);
tau = x(4);

if (q == 0) 
    u = - lambda*z2 - gamma*GradientL(z1); 
elseif (q == 1)
    dBar = 3/(2*(tau + 2));
    betaBar = (tau - 1)/(tau + 2);
    u = - 2*dBar*z2 - (zeta^2/(M))*GradientL(z1 + betaBar*z2); 
end

xdot = [z2;u;0;q]; 
end

% g for Uniting
function xplus = gU(x)

% state
z1 = x(1);
z2 = x(2);
q = x(3);

xplus = [z1;z2;1-q;0]; 
end  

%% HAND-1 Algorithm (C,f,D,g):

% C for HAND-1:
function [value] = C(x,tauMin,tauMax) 

tau = x(3);

if (tau >= tauMin && tau <= tauMax)
    value = 1;
else
    value = 0;
end

end

% D for HAND-1: 
function inside = D(x,tauMed,tauMax) 

tau = x(3);

if(tau >= tauMed && tau <= tauMax)
    inside = 1; 
else
    inside = 0;
end

end

% f for HAND-1:
function xdot = f(x,c)

% state
z1 = x(1);
z2 = x(2);
tau = x(3);

% z1dot = z2;
% z2dot = -(3/tau)*z2 - 4*c*GradientL(z1);

z1dot = (2/tau)*(z2 - z1);
z2dot = -2*c*tau*GradientL(z1);

xdot = [z1dot;z2dot;1]; 
end

% g for HAND-1:
function xplus = g(x,tauMin)

% state
z1 = x(1);
z2 = x(2);
% tau = x(3);

xplus = [z1;z2;tauMin]; 
end  

%% Nesterov (C,f,D,g):
% C for Nesterov:
function [value] = CNesterov(x) 
value = 1;
end

% D for Nesterov:
function inside = DNesterov(x) 
inside = 0;
end

% f for Nesterov
function xdot = fNesterov(x,M,zeta)

% state
z1 = x(1);
z2 = x(2);
q = x(3);
tau = x(4);

dBar = 3/(2*(tau + 2));
betaBar = (tau - 1)/(tau + 2);
 
u = - 2*dBar*z2 - (zeta^2/(M))*GradientL(z1 + betaBar*z2); 

xdot = [z2;u;0;1]; 
end

% g for Nesterov
function xplus = gNesterov(x)

% state
z1 = x(1);
z2 = x(2);
q = x(3);
tau = x(4);

xplus = [z1;z2;q;tau]; 
end   

%% Heavy Ball Algorithm (C,f,D,g):

% C for heavy ball
function [value] = CHBF(x) 
value = 1;
end

% D for heavy ball
function inside = DHBF(x) 
inside = 0;
end

% f for heavy ball
function xdot = fHBF(x,lambda,gamma)

% state
z1 = x(1);
z2 = x(2);
q = x(3);

u = - lambda*z2 - gamma*GradientL(z1); 

xdot = [z2;u;0]; 
end

% g for heavy ball
function xplus = gHBF(x)
% state
z1 = x(1);
z2 = x(2);
q = x(3);

xplus = [z1;z2;q]; 
end    

%% Functions for setting the minimizer, and for calculating L and its gradient.
function setMinimizer()
global z1Star
z1Star = 0;
end

function L = CalculateLStar()
global z1Star
L = z1Star^2;
end

function L = CalculateL(z1)
L = z1^2;
end

function GradL = GradientL(z1)
GradL = 2*z1;
end

%% Function for finding the time that a solution takes to get to, and stay within, 1% of the minimizer.
% Also for finding a vector of L values from the resulting x vector.
function [deltaVec lValue lDeltaValue] = timeToConv(xvalue,tvalue,delta)
global z1Star

z1delta = 0;
timeToDeltaIdx = 1;

    % Finding time of convergence 
    for i=2:length(xvalue(:,1))
        if (((abs(z1Star - xvalue(i,1)) <= delta) && (abs(z1Star - xvalue(i-1,1)) > delta)))
            timeToDeltaIdx = i;
            z1delta = xvalue(i,1);
        end
    end
    deltaVec(1) = z1delta;
    deltaVec(2) = xvalue(timeToDeltaIdx,2);
    deltaVec(3) = tvalue(timeToDeltaIdx,1); 
    
    % Finding L values:
    lValue = zeros(1,length(tvalue));
    for i=1:length(tvalue(:,1))
        lValue(i) = (CalculateL(xvalue(i,1)) - CalculateLStar());
    end

    lDeltaValue = lValue(timeToDeltaIdx);
end