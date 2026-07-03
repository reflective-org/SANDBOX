function dCdt = concs_het(t,x)          % this function also calls 2 gamma calc functions
% x = [Cl, ClO, ClOOCl, ClONO2, HCl, Cl2, NO, NO2, O, O2, O3, CH3, CH4, HNO3, H2O, HOCl, N2O5, NO3, 
%      OH, HO2, O1D, HONO, HNO4, OClO, Br, BrO, BrONO2, BrCl, HBr, HOBr,HNO3aq, C2H6, SO2, H2O2]                                 % FK added SO2

global T P M SA WTR SZA Yn2o5 count opt

% DEFINE VARIABLES 
Cl		=	x(1);			ClO     =	x(2);			ClOOCl  =	x(3);				ClONO2  =	x(4);	
HCl     =	x(5);			Cl2     =	x(6);			NO		=	x(7);               NO2     =	x(8);          
O		=	x(9);           O2		=	x(10);          O3		=	x(11);              CH3     =	x(12);          
CH4     =	x(13);          HNO3    =	x(14);          H2O     =	x(15);              HOCl    =	x(16);
N2O5    =	x(17);          NO3     =	x(18);          OH      =	x(19);              HO2     =	x(20);
O1D     =	x(21);          HONO    =	x(22);          HNO4    =	x(23);              OClO    =	x(24);
Br      =	x(25);          BrO     =	x(26);          BrONO2  =	x(27);              BrCl    =	x(28);
HBr     =	x(29);          HOBr    =	x(30);          HNO3aq  =	x(31);              C2H6    =	x(32);
SO2     =   x(33);          H2O2    =   x(34);
SO3     =   x(35);          H2SO4   =   x(36);          % FK: gas-phase sulfur oxidation chain

%opt is set in runconcs_het: 0 to run O3 photolysis new way w/ full rxns; 1 to run old way with workaround as in Science paper  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE GAMMAS FOR HETEROGENEOUS REACTIONS 
% P in mbar; HCl in ppbv; ClONO2 in ppbv; R in cm;
% JPL reference:  P=50; HCL=2.0; CLONO2=0.1; R=0.1e-4; (sts=0;) H2O=5ppm;

Pf=P;
HCLf=HCl./M.*1e9;           % convert from molec/cm3 to ppb
CLONO2f=ClONO2./M.*1e9;     % convert from molec/cm3 to ppb
H2Of=H2O./M.*1e6;           % convert from molec/cm3 to ppm
R=0.1e-4; sts=0;

%T=[185:0.1:235];
  [h2so4wpAT,h2so4mlAT,a_WAT] = h2so4wpATfxn(T,Pf,H2Of);
  [Yhocl,Yclnh2o,Yclnhcl] = hetgammasJPL00(T,Pf,h2so4wpAT,a_WAT,HCLf,CLONO2f,R,sts);
  
clear Pf HCLf CLONO2f H2Of R sts h2so4wpAT h2so4mlAT a_WAT;
% end result of this code is creation of 3 new variables: gamma for ClONO2+HCl, ClONO2+H2O, and HCl+HOCl

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RATE CONSTANTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHLORINE REACTIONS - ClOO and ClNO flight duct chemistry removed (not nec for modeling atmosphere)

% ClO + NO -> NO2 + Cl                                                                      JPL-11
	k1=6.4e-12*exp(290/T);

% ClO + ClO + M -> ClOOCl + M                                                               JPL-11
	kzero=1.6e-32.*((T/300).^(-4.5));
	kinf=3.0e-12.*((T/300).^(-2.0));
	k2=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));

% ClOOCl + M -> ClO + ClO + M
	%Kequil=1.92e-27.*exp(8430../T);                                                       % Plenge 2005
    Kequil=1.72e-27.*exp(8649../T);                                                       % JPL-11
    k3=k2./Kequil;

% ClO + NO2 + M -> ClONO2 + M                                                               JPL-11
	kzero=1.8e-31.*((T/300).^(-3.4));
	kinf=1.5e-11.*((T/300).^(-1.9));
	k4=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));

% ClONO2 + M -> ClO + NO2 + M																Fahey
	k5=6.9e-7*exp(-10909/T);
   %k5=8.3e-6*exp(-11820/T);				%Schonle - factor of 2 faster than Fahey at 510 K
   
% Cl + O3 -> ClO + O2                                                                       JPL-11
	k6=2.3e-11*exp(-200/T);
   
% Cl + CH4 -> HCl + CH3																		JPL-11   
	k7=7.3e-12*exp(-1280/T);

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HETEROGENEOUS REACTIONS - calculate the gas/aerosol collision rate and first-order het rxn rate constant
%   k = khet(T,gasmass,gamma,SA) with units: T (K), gas mass (g/mol), SA = surface area (um^2/cm^3)
%   T and SA are defined in runconcs; gammas are calculated in functions called above

kb=1.3807e-23;          % Boltzmann in J/K
m=1/6.02e23*1/1000;     % mass per atom in kg

% ClONO2 + HCl -> Cl2 + HNO3(aq) 
    gamma=Yclnhcl;                          % gamma for this rxn
    %gamma=Yclnhcl.*1;                          % gamma for this rxn - change to test
    gasmass=97;                             % ClONO2
    v=100.*sqrt(8*kb*T./(pi.*gasmass.*m));	% mean velocity (cm/s)
    k8=0.25.*gamma.*SA.*1e-8.*v;            % 1st order rate constant in units of s-1
    
% ClONO2 + H2O -> HOCl + HNO3(aq)
    gamma=Yclnh2o;                          % gamma for this rxn
    %gamma=Yclnh2o.*1;                          % gamma for this rxn - change to test
    gasmass=97;                             % ClONO2
    v=100.*sqrt(8*kb*T./(pi.*gasmass.*m));	% mean velocity (cm/s)
    k9=0.25.*gamma.*SA.*1e-8.*v;            % 1st order rate constant in units of s-1

% HOCl + HCl -> Cl2 + H2O  
    gamma=Yhocl;                            % gamma for this rxn
    gasmass=52;                             % HOCl
    v=100.*sqrt(8*kb*T./(pi.*gasmass.*m));	% mean velocity (cm/s)
    k10=0.25.*gamma.*SA.*1e-8.*v;           % 1st order rate constant in units of s-1

% N2O5 + H2O -> HNO3(aq) + HNO3(aq) 
    gamma=Yn2o5;                            % gamma for this rxn - defined manually in runconcs
    gasmass=108;                            % N2O5
    v=100.*sqrt(8*kb*T./(pi.*gasmass.*m));	% mean velocity (cm/s)
    k11=0.25.*gamma.*SA.*1e-8.*v;           % 1st order rate constant in units of s-1

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PHOTOLYSIS REACTIONS - use J values at 45 deg SZA - all J values have units of s-1

if SZA==1;          % daytime defined in runconcs_het; if nighttime (SZA=0), set all J's to 0 below
    
% ClONO2 + hv -> Cl + NO3 
    k12a=6.5e-5*(0.9)/1.5;

  % ClONO2 + hv -> ClO + NO2
    k12b=6.5e-5*(0.1)/1.5; 
    
% ClOOCl + hv -> Cl + Cl + O2         % products are actually Cl+ClOO, but ClOO very quickly goes to Cl+O2  
    k13a=2.3e-3*(0.9); 

  % ClOOCl + hv -> ClO + ClO  
    k13b=2.3e-3*(0.1); 
    
% Cl2 + hv -> Cl + Cl
    k14=4.0e-3; 

% HOCl + hv -> OH + Cl
    k15=4.5e-4; 
    
% HNO3 + hv -> OH + NO2
    k16=1.0e-6;         % 1e-6 or 6e-7
    
% NO3 + hv -> NO2 + O 
    k17a=0.23*(0.9);   

  % NO3 + hv -> NO + O2 
    k17b=0.23*(0.1);
    
% NO2 + hv -> NO + O  
    k18=0.014./1;     % divide by 4-5 here to set NO/NO2 ratio to be ~1 in daytime lowermost strat  

% N2O5 + hv -> NO2 + NO3  
    k19=2.7e-5;  

% O3 exclusively photolyzes to O1D, which is rapidly quenched to O in the atm. But a small fraction of O1D reacts with H2O, CH4, and H2 to form OH. 
  % run O3 photolysis new way w/ full chemistry & AbsTol adjusted for O and O1D in runconcs:

% O3 + hv -> O2 + O  
    k20a=0;
    
  % O3 + hv -> O2 + O1D  
    k20b=4.7e-5;

%%%%% Workaround for O3 photolysis from Science paper (r20a/b, r40, r41) %%%%% 
%   The default ode15s solver can't handle the rapid O1D formation and quenching, so just have tiny fraction photolyze directly to O1D and rest go to O. 
%   Checked resulting OH this way and with ode23s before workaround, and this is quite accurate, but not perfect as H2O changes, b/c H2O rxn makes 2OH.
%   At 5 and 10ppm H2O, normally plotted Cly and NOx concs seem to be same to within ~0.2%. OH to within ~3%.
    if opt==1;   
    frc=6.4e-6*WTR;         % typically 6.4e-6*WTR, set assuming CH4=1.6ppm and starting water can vary; 2.4e-5, 6.4e-5
    k20a=4.7e-5*(1-frc); k20b=4.7e-5*(frc); end;     % overwrite k20a/b values above if opt=1    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end; if SZA==0; k12a=0; k12b=0; k13a=0; k13b=0; k14=0; k15=0; k16=0; k17a=0; k17b=0; k18=0; k19=0; k20a=0; k20b=0; end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOx REACTIONS

% NO2 + NO3 + M -> N2O5 + M                                                                 JPL-11
	kzero=2.0e-30.*((T/300).^(-4.4));
	kinf=1.4e-12.*((T/300).^(-0.7));
	k21=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));

% OH + HNO3 -> H2O + NO3                                                                    JPL-11
	kzero=2.4e-14.*exp(460/T);
    ktwo=2.7e-17.*exp(2199/T);
    kthree=6.5e-34.*exp(1335/T);
    k22=kzero + (kthree.*M)./(1+((kthree.*M)./ktwo));

% OH + NO2 + M -> HNO3 + M                                                                  JPL-11
	kzero=1.8e-30.*((T/300).^(-3.0));
	kinf=2.8e-11.*((T/300).^(0.0));
	k23=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));

% O + NO + M -> NO2 + M                                                                     JPL-11
	kzero=9.0e-32.*((T/300).^(-1.5));
	kinf=3.0e-11.*((T/300).^(0.0));
	k24=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));

% O + NO2 + M -> NO3 + M                                                                    JPL-11
	kzero=2.5e-31.*((T/300).^(-1.8));
	kinf=2.2e-11.*((T/300).^(-0.7));
	k25=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));

% O + NO2 -> NO + O2                                                                        JPL-11
	k26=5.1e-12*exp(210/T);
    
% NO + O3 -> NO2 + O2                                                                       JPL-11
	k27=3.0e-12*exp(-1500/T);
    
% NO2 + O3 -> NO3 + O2                                                                      JPL-11
	k28=1.2e-13*exp(-2450/T);
    
% OH + NO + M -> HONO + M                                                                   JPL-11
	kzero=7.0e-31.*((T/300).^(-2.6));
	kinf=3.6e-11.*((T/300).^(-0.1));
	k29=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));

% OH + HONO -> H2O + NO2                                                                    JPL-11
	k30=1.8e-11*exp(-390/T);
    
% NO + NO3 -> NO2 + NO2                                                                     JPL-11
	k31=1.5e-11*exp(170/T);
    
% OH + HNO4 -> H2O + NO2 + O2                                                               JPL-11
	k32=1.3e-12*exp(380/T);
    
% HO2 + NO -> NO2 + OH                                                                      JPL-11
	k33=3.3e-12*exp(270/T);
    
% HO2 + NO2 + M -> HNO4 + M                                                                 JPL-11
	kzero=2.0e-31.*((T/300).^(-3.4));
	kinf=2.9e-12.*((T/300).^(-1.1));
	k34=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HOx, Ox, and O(1D) REACTIONS

% O + O2 + M -> O3 + M                                                                      JPL-11
	kzero=6.0e-34.*((T/300).^(-2.4));
	k35=kzero.*M;

% O + O3 -> O2 + O2                                                                         JPL-11
    k36=8.0e-12*exp(-2060/T);

% OH + O3 -> HO2 + O2                                                                       JPL-11
    k37=1.7e-12*exp(-940/T);

% OH + HO2 -> H2O + O2                                                                      JPL-11
    k38=4.8e-11*exp(250/T);

% HO2 + O3 -> OH + O2 + O2                                                                  JPL-11
    k39=1.0e-14*exp(-490/T);

% O1D + O2 -> O + O2                                                                        JPL-11
    k40=3.3e-11*exp(55/T);
        
% O1D + N2 -> O + N2                                                                        JPL-11
    k41=2.15e-11*exp(110/T);
        
% O1D + H2O -> OH + OH                                                                      JPL-11
    k42=1.63e-10*exp(60/T);

% O1D + CH4 -> CH3 + OH                                                                     JPL-11
    k43=1.31e-10*exp(0/T); 
 
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MORE CHLORINE REACTIONS

% O + ClO -> Cl + O2                                                                        JPL-11
    k44=2.8e-11*exp(85/T);
    
% OH + ClO -> Cl + HO2                                                                      JPL-11
    k45=7.4e-12*exp(270/T);

% OH + ClO -> HCl + O2                                                                      JPL-11
    k46=6.0e-13*exp(230/T);
    
% OH + HCl -> Cl + H2O                                                                      JPL-11
    k47=1.8e-12*exp(-250/T);

% HO2 + ClO -> HOCl + O2                                                                    JPL-11
    k48=2.6e-12*exp(290/T);
    
% HO2 + ClO -> HCl + O3                                                                     ref
    k49=k48*(0.03);    

% BROMINE REACTIONS

% O + BrO -> Br + O2                                                                        JPL-11
    k50=1.9e-11*exp(230/T);

% Br + O3 -> BrO + O2                                                                       JPL-11
    k51=1.6e-11*exp(-780/T);
    
% BrO + NO -> NO2 + Br                                                                      JPL-11
    k52=8.8e-12*exp(260/T);

% BrO + ClO -> Br + Cl + O2        % technically ClOO                                       JPL-11
    k53=2.3e-12*exp(260/T);    

% BrO + ClO -> BrCl + O2                                                                    JPL-11
    k54=4.1e-13*exp(290/T);
    
% BrO + ClO -> Br + OClO                                                                    JPL-11
    k55=9.5e-13*exp(550/T);
  
% BrO + NO2 + M -> BrONO2 + M                                                               JPL-11
	kzero=5.2e-31.*((T/300).^(-3.2));
	kinf=6.9e-12.*((T/300).^(-2.9));
	k56=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MORE PHOTOLYSIS REACTIONS - use J values at 45 deg SZA - all J values have units of s-1

if SZA==1;          % daytime defined in runconcs_het; if nighttime (SZA=0), set all J's to 0 below
    
% HNO4 + hv -> NO2 + HO2 
    k57a=1.3e-5*(0.8);

  % HNO4 + hv -> NO3 + OH
    k57b=1.3e-5*(0.2); 
      
% OClO + hv -> O + ClO
    k58=0.013; 
    
% BrO + hv -> Br + O
    k59=0.060; 

% BrONO2 + hv -> Br + NO3 
    k60a=1.8e-3*(0.85);

  % BrONO2 + hv -> BrO + NO2
    k60b=1.8e-3*(0.15); 
    
% BrCl + hv -> Br + Cl
    k61=0.017;    
    
% O2 + hv -> O + O
    k62=4.0e-13;  
    
% HONO + hv -> OH + NO
    k63=5.0e-4;     % estimate from cross sections: J for HONO should be larger than N2O5 and HOCl and smaller than Cl2 and ClOOCl
   
end; if SZA==0; k57a=0; k57b=0; k58=0; k59=0; k60a=0; k60b=0; k61=0; k62=0; k63=0; end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OTHER

% HNO3aq -> HNO3    % allow for HNO3 to reenter gas phase from liquid phase - at our est sulfate size, >90% comes back 
    k64=1.0e-5;     % 1e-5 is a little high, but compensates for uncertain starting HNO3; 1e-9 off entirely, 1e-3 on entirely
    
% Cl + C2H6 -> HCl + C2H5																		JPL-11   
	k65=7.2e-11*exp(-70/T);    
     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BROMINE HETEROGENEOUS REACTION - calculate the gas/aerosol collision rate and first-order het rxn rate constant
kb=1.3807e-23;          % Boltzmann in J/K
m=1/6.02e23*1/1000;     % mass per atom in kg
  
% BrONO2 + H2O -> HOBr + HNO3(aq)
    Ybrono2=0.8;                            % fix at 0.8 based on JPL (analogous to Yn2o5 above)
    gamma=Ybrono2;                          % gamma for this rxn
    gasmass=142;                            % BrONO2
    v=100.*sqrt(8*kb*T./(pi.*gasmass.*m));	% mean velocity (cm/s)
    k66=0.25.*gamma.*SA.*1e-8.*v;           % 1st order rate constant in units of s-1

% ADDITIONAL BROMINE PHOTOLYSIS REACTION - use J values at 45 deg SZA - all J values have units of s-1
if SZA==1;          % daytime defined in runconcs_het; if nighttime (SZA=0), set J to 0 below

% HOBr + hv -> OH + Br
    k67=1.8e-3;     % estimate from cross sections: J for HOBr should be very similar to BrONO2

end; if SZA==0; k67=0; end;
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% SO2 REACTIONS FK addition

%k68=6e-13;
kzero=2.9e-31.*((T/300).^(-4.1));           % FK: was (298/300) frozen at 298K; now box-T dependent (JPL termolecular, 300K ref)
kinf=1.7e-12.*((T/300).^(0.2));             % FK: was (298/300) frozen at 298K; now box-T dependent
k68=(((kzero.*M)./(1+(kzero.*M./kinf))).*0.6.^((1+(log10(kzero.*M./kinf)).^2).^(-1)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% CH4 REACTIONS FK addition

k69=6.e-16;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REACTION RATES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% HO2 REACTIONS FK addition

k70=3.*10^(-13).*(T/298).*exp(460./T);  
k71=1E-5;                                   %photolysis
k72=1E-18;                                  % SO2 + HO2 -> SO3 + OH (was 5E-18; now OH-producing chain)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REACTION RATES
r1  = k1*ClO*NO;
r2  = k2*ClO*ClO;
r3  = k3*ClOOCl;
r4  = k4*ClO*NO2;
r5  = k5*ClONO2*M;
r6  = k6*Cl*O3;
r7  = k7*Cl*CH4;
r8  = k8*ClONO2; 
r9  = k9*ClONO2; 
r10 = k10*HOCl;  
r11 = k11*N2O5;
r12a= k12a*ClONO2;
r12b= k12b*ClONO2;
r13a= k13a*ClOOCl;
r13b= k13b*ClOOCl;
r14 = k14*Cl2;
r15 = k15*HOCl; 
r16 = k16*HNO3;             % main OH prodcution
r17a= k17a*NO3;
r17b= k17b*NO3;
r18 = k18*NO2;
r19 = k19*N2O5;
r20a= k20a*O3;
r20b= k20b*O3;
r21 = k21*NO2*NO3;
r22 = k22*OH*HNO3; 
r23 = k23*OH*NO2;
r24 = k24*O*NO;
r25 = k25*O*NO2;
r26 = k26*O*NO2;
r27 = k27*NO*O3;
r28 = k28*NO2*O3;
r29 = k29*OH*NO;
r30 = k30*OH*HONO;
r31 = k31*NO*NO3;
r32 = k32*OH*HNO4;
r33 = k33*HO2*NO;            % main OH production
r34 = k34*HO2*NO2;
r35 = k35*O*O2;
r36 = k36*O*O3;
r37 = k37*OH*O3;
r38 = k38*OH*HO2;
r39 = k39*HO2*O3;
r40 = k40*O1D*O2;                % ode15s violates integration tolerances with default AbsTol; odeset in runconcs resets AbsTol for O and O1D so this works
r41 = k41*O1D*0.79*M;            % N2=0.79*M
if opt==1; r40=0; r41=0; end;    % overwrite r40/r41 & set=0 if running ozone photolysis workaround from Science paper (see k20a/b above)
r42 = k42*O1D*H2O;               % important for OH
r43 = k43*O1D*CH4;
r44 = k44*O*ClO;
r45 = k45*OH*ClO;
r46 = k46*OH*ClO;
r47 = k47*OH*HCl;
r48 = k48*HO2*ClO;
r49 = k49*HO2*ClO;
r50 = k50*O*BrO;
r51 = k51*Br*O3;
r52 = k52*BrO*NO;
r53 = k53*BrO*ClO;
r54 = k54*BrO*ClO;
r55 = k55*BrO*ClO;
r56 = k56*BrO*NO2;
r57a= k57a*HNO4;
r57b= k57b*HNO4;
r58 = k58*OClO;
r59 = k59*BrO;
r60a= k60a*BrONO2;
r60b= k60b*BrONO2;
r61 = k61*BrCl;
r62 = k62*O2;
r63 = k63*HONO;                  % important for OH
r64 = k64*HNO3aq;
r65 = k65*Cl*C2H6;
r66 = k66*BrONO2;
r67 = k67*HOBr;
r68 = k68*SO2*OH;                                                                   % FK added
r69 = k69*CH4*OH;                                                                   % FK added
r70 = k70*HO2*HO2;
r71 = k71*H2O2;
r72 = k72*SO2*HO2;                                                                  % FK: SO2+HO2 -> SO3+OH
r73 = 8.5e-41.*exp(6540./T).*SO3.*H2O.^2;                                            % FK: SO3+H2O -> H2SO4 (JPL 19-5 I79)
%r1=0; 
%r2=0; 
%r3=0; 
%r4=0; 
%r5=0; 
%r6=0; 
%r7=0; 
%r8=0; 
%r9=0; 
%r10=0; 
%r11=0;
%r12a=0; r12b=0;
%r13a=0; r13b=0; 
%r14=0; 
%r15=0; 
%r16=0; 
%r17a=0; r17b=0; 
%r18=0; 
%r19=0; 
%r20a=0; r20b=0;
%r21=0; 
%r22=0; 
%r23=0; 
%r24=0; 
%r25=0;
%r26=0; 
%r27=0; 
%r28=0;
%r29=0; 
%r30=0; 
%r31=0; 
%r32=0; 
%r33=0; 
%r34=0; 
%r35=0;
%r36=0; 
%r37=0; 
%r38=0; 
%r39=0;  
%r40=0; 
%r41=0;
%r42=0; 
%r43=0; 
%r44=0; 
%r45=0;
%r46=0; 
%r47=0; 
%r48=0; 
r49=0;    % ClO+HO2 channel for HCl production is not a listed JPL rate (mentioned in notes), so omit
%r50=0; 
%r51=0;
%r52=0;
%r53=0; 
%r54=0; 
%r55=0;
%r56=0;
%r57a=0; %r57b=0;
%r58=0;
%r59=0;
%r60a=0; r60b=0;
%r61=0;
r62=0;    % produces 5% O3 per day in unperturbed conditions - leave off
%r63=0;
%r64=0;
r65=0;
%r66=0;
%r67=0;

count=count+1;


% REACTIONS:
dCdt = [r1-r6-r7+r12a+2*r13a+2*r14+r15+r44+r45+r47+r53+r61-r65; ...                                             % Cl
        -r1-2*r2+2*r3-r4+r5+r6+r12b+2*r13b-r44-r45-r46-r48-r49-r53-r54-r55+r58; ...                             % ClO
        r2-r3-r13a-r13b; ...                                                                                    % ClOOCl
        r4-r5-r8-r9-r12a-r12b; ...                                                                              % ClONO2
        r7-r8-r10+r46-r47+r49+r65; ...                                                                          % HCl
        r8+r10-r14; ...                                                                                         % Cl2
        -r1+r17b+r18-r24+r26-r27-r29-r31-r33-r52+r63; ...                                                       % NO
        r1-r4+r5+r12b+r16+r17a-r18+r19-r21-r23+r24-r25-r26+r27-r28+r30+2*r31+r32+r33-r34+r52-r56+r57a+r60b; ... % NO2
        r17a+r18+r20a-r24-r25-r26-r35-r36+r40+r41-r44-r50+r58+r59+2*r62; ...                                    % O
        r6+r13a+r17b+r20a+r20b+r26+r27+r28+r32-r35+2*r36+r37+r38+2*r39+r44+r46+r48+r50+r51+r53+r54-r62; ...     % O2
        -r6-r20a-r20b-r27-r28+r35-r36-r37-r39+r49-r51; ...                                                      % O3
        r7+r43; ...                                                                                             % CH3
        -r7-r43-r69; ...                                                                                        % CH4 FK added r69
        -r16-r22+r23+r64; ...                                                                                   % HNO3
        -r9+r10-r11+r22+r30+r32+r38-r42+r47-r66-r73; ...                                                        % H2O  FK -r73 (SO3+H2O->H2SO4)
        r9-r10-r15+r48; ...                                                                                     % HOCl
        -r11-r19+r21; ...                                                                                       % N2O5
        r12a-r17a-r17b+r19-r21+r22+r25+r28-r31+r57b+r60a; ...                                                   % NO3
        r15+r16-r22-r23-r29-r30-r32+r33-r37-r38+r39+2*r42+r43-r45-r46-r47+r57b+r63+r67-r68-r69+2*r71+r72; ...   % OH  FK added r68 r69 r72(SO2+HO2->SO3+OH)
        -r33-r34+r37-r38-r39+r45-r48-r49+r57a+r68+r69-2*r70-r72; ...                                            % HO2  FK added r68 and r69
        r20b-r40-r41-r42-r43; ...                                                                               % O1D 
        r29-r30-r63; ...                                                                                        % HONO
        -r32+r34-r57a-r57b; ...                                                                                 % HNO4
        r55-r58; ...                                                                                            % OClO
        r50-r51+r52+r53+r55+r59+r60a+r61+r67; ...                                                               % Br
        -r50+r51-r52-r53-r54-r55-r56-r59+r60b; ...                                                              % BrO
        r56-r60a-r60b-r66; ...                                                                                  % BrONO2
        r54-r61; ...                                                                                            % BrCl
        0; ...                                                                                                  % HBr
        r66-r67; ...                                                                                            % HOBr
        r8+r9+2*r11-r64+r66; ...                                                                                % HNO3aq
        -r65; ...                                                                                               % C2H6
        -r68-r72;...                                                                                                  % SO2    FK added r68 (SO2+OH->SO3+HO2), r72 (SO2+HO2->SO3+OH)
        +r70-r71; ...                                                                                                 % H2O2
        +r68+r72-r73; ...                                                                                             % SO3    FK: +SO2+OH->SO3+HO2, +SO2+HO2->SO3+OH, -SO3+H2O->H2SO4
        +r73;];                                                                                                       % H2SO4  FK: SO3+H2O->H2SO4
    