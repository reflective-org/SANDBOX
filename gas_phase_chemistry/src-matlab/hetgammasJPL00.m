% hetgammasJPL00.m -- adapted from JPL 2000 & Hanson, JPCA 102, p. 4794
% and Tom's clono2h2o.m
% function [Yhocl,Yclnh2o,Yclnhcl]=...
% 	hetrxnsJPL00rev(T,P,h2so4wp,a_W,HCl,ClONO2,radius,sts)
% T in K; P in mbar; HCl in ppbv; ClONO2 in ppbv; radius in cm;
% h2so4wp & h2so4ml calculated by another program; sts [0,1]
%
% Calculate ClONO2 + HCl -> Cl2 + HNO3
% Calculate ClONO2 + H2O -> HOCl + HNO3
% Calculate HOCl + HCl -> Cl2 + H2O
function [Yhocl,Yclnh2o,Yclnhcl] = ...
    hetgammasJPL00(T,P,h2so4wp,a_W,HCl,ClONO2,radius,sts)
sts;

M=0.001.*P.*2.46e19.*298./T;  % P in mbar  2.46e19 mol/cc @ 298 K
p_hcl=HCl.*1e-9.*P./1013;
p_clono2=ClONO2.*1e-9.*P./1013;

h2so4ml=h2so4wp.*1000./(9800-98.*h2so4wp);

% Calculate H2SO4 density from molality (h2so4ml)
Z1=0.12364-5.6e-7.*T.^2;
Z2=-0.02954+1.814e-7.*T.^2;
Z3=2.343e-3-1.487e-6.*T-1.324e-8.*T.^2;
density=1+Z1.*h2so4ml+Z2.*h2so4ml.^1.5+Z3.*h2so4ml.^2;

% Calculate molarity (h2so4M) & mole fraction (X)
h2so4M=density.*h2so4wp./9.8;
X=h2so4wp./(h2so4wp+(100-h2so4wp).*98./18);

% Calculate viscosity
T0=144.11+0.166.*h2so4wp-0.015.*h2so4wp.^2+2.18e-4.*h2so4wp.^3;
A=169.5+5.18.*h2so4wp-0.0825.*h2so4wp.^2+3.27e-3.*h2so4wp.^3;
viscosity=A.*(T.^-1.43).*exp(448./(T-T0));

% Calculate acid activity
a_H1=(60.51-0.095.*h2so4wp+0.0077.*h2so4wp.^2-1.61e-5.*h2so4wp.^3);
a_H2=-(1.76+2.52e-4.*h2so4wp.^2).*T.^0.5;
a_H3=(-805.89+253.05.*h2so4wp.^0.076)./(T.^0.5);
a_H=exp(a_H1+a_H2+a_H3);

% Start uptake stuff
k_H=1.22e12.*exp(-6200./T);
k_h2o=1.95e10.*exp(-2800./T);
k_hydr=k_h2o.*a_W+k_H.*a_H.*a_W;
D_clono2=5e-8.*T./viscosity;
S_clono2=0.306+24./T;

% Effective Henry's law coefficient for ClONO2
H_clono2=1.6e-6.*exp(4710./T).*exp(-S_clono2.*h2so4M);
C_clono2=1474.*T.^0.5;
R=0.082;
uptake_h2o_b=4.*H_clono2.*R.*T.*((D_clono2.*k_hydr).^0.5)./C_clono2;

% HCl uptake. Use measured HCl gas pressure (atm)
% Effective Henry's law coefficient for HCl
H_hcl=real((0.094 - 0.61.*X + 1.2.*X.^2).*exp(-8.68 + (8515 - 10718.*X.^0.7)./T));
M_hcl=H_hcl.*p_hcl;
k_hcl=7.9e11.*a_H.*D_clono2.*M_hcl;

% Reacto-diffusive length
l_clono2=real((D_clono2./(k_hydr + k_hcl)).^0.5);
f_clono2=1./tanh(radius./l_clono2)-l_clono2./radius;
uptake_rxn_clono2=real(f_clono2.*uptake_h2o_b.*(1+k_hcl./k_hydr).^0.5);

if sts==1
	uptake_rxn_clono2=uptake_rxn_clono2./2;
end

uptake_hcl_b=uptake_rxn_clono2.*k_hcl./(k_hcl+k_hydr);
uptake_s=66.12.*H_clono2.*M_hcl.*exp(-1374./T);

if sts==1
	uptake_s=uptake_s./10;
end

% F accounts for the depletion of HCl in the particles due to reaction 
% with ClONO2 inside or on the particle surface.
F_hcl=1./(1+0.612.*(uptake_s+uptake_hcl_b).*p_clono2./p_hcl);
uptake_s_prime=F_hcl.*uptake_s;
uptake_hcl_b_prime=F_hcl.*uptake_hcl_b;
uptake_b=uptake_hcl_b_prime+uptake_rxn_clono2.*k_hydr./(k_hcl+k_hydr);

% Gammas at last
Ycln=1./(1+1./(uptake_s_prime+uptake_b));
Yclnhcl=Ycln.*(uptake_s_prime+uptake_hcl_b_prime)./(uptake_s_prime+uptake_b);
Yclnh2o=Ycln-Yclnhcl;

%?? JPL-00 page 53: if alpha=1 and not using F_hcl correction
%?? Ycln = uptake_h2o_b
D_hocl=6.4e-8.*T./viscosity;
k_hocl=1.25e9.*a_H.*D_hocl.*M_hcl;

% Option Try from Donaldson, et al, JPC, 101, p. 4717.
%k2=exp(2.303.*(6.08-1050../T +0.0747.*h2so4wp));
%k_hocl=k2.*H_hocl.*p_hcl;

S_hocl=0.0776+59.18./T;
H_hocl=1.91e-6.*exp(5862.4./T).*exp(-S_hocl.*h2so4M);
C_hocl=2009.*T.^0.5;
uptake_rxn_hocl=4.*H_hocl.*R.*T.*((D_hocl.*k_hocl).^0.5)./C_hocl;
%l_clono2=real((D_clono2./(k_hydr + k_hcl)).^0.5);
%f_clono2=1./tanh(radius./l_clono2)-l_clono2./radius;
%l_hocl=(D_hocl./k_hocl).^0.5;
l_hocl=real(D_hocl./k_hocl).^0.5;
f_hocl=1./tanh(radius./l_hocl)-l_hocl./radius;

Yhocl=1./(1 + 1./(f_hocl.*uptake_rxn_hocl.*F_hcl));

%??  JPL-00 page 51: if alpha=1 and not using F_hcl correction
%??  Yhocl=uptake_rxn_hocl

