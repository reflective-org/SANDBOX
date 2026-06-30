% h2so4wpATfxn.m
% adapted from Tabazedeh, JPL, 2000
% T (K); P (mbar); H2O (ppmv)
% function [h2so4wpAT,h2so4mlAT,a_WAT] = h2so4wpATfxn(T,P,H2O);
function [h2so4wpAT,h2so4mlAT,a_WAT] = h2so4wpATfxn(T,P,H2O);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% saturation water vapor pressure (mbar)
p0h2o=exp(18.452406985-3505.1578807./T -330918.55082./(T.^2)+...
    12725068.262./(T.^3));
% use h2o mbar to calculated water activity
ph2o=H2O.*1e-6.*P;
a_W=ph2o./p0h2o; a_WAT=a_W;
%
a1=ones(size(a_W));
b1=a1;
c1=a1;
d1=a1;
a2=a1;
b2=a1;
c2=a1;
d2=a1;
%
for i=1:length(a_W);
   if a_W(i)<=0.05
      a1(i)=12.37208932;
      b1(i)=-0.16125516114;
      c1(i)=-30.490657554;
      d1(i)=-2.1133114241;
      a2(i)=13.455394705;
      b2(i)=-0.1921312255;
      c2(i)=-34.285174607;
      d2(i)=-1.7620073078;      
   elseif a_W(i)>0.05 & a_W(i)<0.85
      a1(i)=11.820654354;
      b1(i)=-0.20786404244;
      c1(i)=-4.807306373;
      d1(i)=-5.1727540348;
      a2(i)=12.891938068;
      b2(i)=-0.23233847708;
      c2(i)=-6.4261237757;
      d2(i)=-4.9005471319;
   elseif a_W(i)>=0.85
      a1(i)=-180.06541028;
      b1(i)=-0.38601102592;
      c1(i)=-93.317846778;
      d1(i)=273.88132245;
      a2(i)=-176.95814097;
      b2(i)=-0.36257048154;
      c2(i)=-90.469744201;
      d2(i)=267.45509988;
   end
end
y1=a1.*a_W.^b1+c1.*a_W+d1;
y2=a2.*a_W.^b2+c2.*a_W+d2;
% calculate h2so4 molality mol/kg & h2so4 wt%
h2so4mlAT=y1+(T-190).*(y2-y1)./70;
h2so4wpAT=9800.*h2so4mlAT./(98.*h2so4mlAT+1000);

