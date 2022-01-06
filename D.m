function inside = D(x) 
%--------------------------------------------------------------------------
% Matlab M-file Project: HyEQ Toolbox @  Hybrid Systems Laboratory (HSL), 
% https://hybrid.soe.ucsc.edu/software
% http://hybridsimulator.wordpress.com/
% Filename: D.m
%--------------------------------------------------------------------------
% Description: Jump set
% Return 0 if outside of D, and 1 if inside D
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%   See also HYEQSOLVER, PLOTARC, PLOTARC3, PLOTFLOWS, PLOTHARC,
%   PLOTHARCCOLOR, PLOTHARCCOLOR3D, PLOTHYBRIDARC, PLOTJUMPS.
%   Copyright @ Hybrid Systems Laboratory (HSL),
%   Revision: 0.0.0.3 Date: 09/29/2020 1:07:00

global tauMed tauMax

tau = x(3);

if(tau >= tauMed && tau <= tauMax)
    inside = 1; 
else
    inside = 0;
end

end