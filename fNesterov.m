function xdot = fNesterov(x)
%--------------------------------------------------------------------------
% Matlab M-file Project: HyEQ Toolbox @  Hybrid Systems Laboratory (HSL), 
% https://hybrid.soe.ucsc.edu/software
% http://hybridsimulator.wordpress.com/
% Filename: fNesterov.m
%--------------------------------------------------------------------------
% Project: Uniting Nesterov's accelerated gradient descent globally with
% heavy ball locally.
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%   See also HYEQSOLVER, PLOTARC, PLOTARC3, PLOTFLOWS, PLOTHARC,
%   PLOTHARCCOLOR, PLOTHARCCOLOR3D, PLOTHYBRIDARC, PLOTJUMPS.
%   Copyright @ Hybrid Systems Laboratory (HSL),
%   Revision: 0.0.0.3 Date: 09/29/2020 1:07:00
   
% The global variables
global M zeta

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