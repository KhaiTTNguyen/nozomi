% this script performs (perhaps incomplete) testing of the dwmriWAVE function

clearvars
clc
close all

now = datetime('now');
filename = sprintf('TEST_dwmriWAVE-%s.md',...
    datetime(now,'Format','yyyy-MM-dd-HH-mm-ss'));
fid = fopen(filename,'w');

if ~fid
    warning('Could not open file for writing. Writing to std out')
    fid = 1;
end

fprintf(fid,'# Testing `dwmriWAVE.m`\n');
fprintf(fid,'\n');
fprintf(fid,'Run at %s\n',now);
fprintf(fid,'\n');


%% look at intra-axonal signal 

fprintf(fid,'## Intra-axonal diffusion\n');
fprintf(fid,'\n');
fprintf(fid,['In this test, intra-axonal signal is compared to an ' ...
    'analytic expression given by van Gelderen & DesPres J Magn 1994.\n']);

parms.T2 = [1e10 1e10]; % t2 in myelin and non-myelin compartments
parms.T1 = [1e10 1e10]; % t1 in myelin and non-myelin compartments
parms.D = [1e-10 3 3]; % "free" diffusion coefficients in myelin (radial, circumferential) and non-myelin compartments
parms.M0r = 0.0; % ratio of spin density
parms.N = 100000; % # of spins
parms.dt = 0.01; % timestep

% create a white matter geometry, but it is totally unrestrictive 
Nax = 200;
axf = 0.5;
axd = 12.0;
g = 0.98;
[parms.xr,parms.yr,parms.ro,parms.Lx] = axonDeltaGen(Nax,axf,axd);
parms.ri = parms.ro*g;

bigD = 20; % ms
litD = 5; % ms
te = 30; % ms
Gdiff = makePGwave(parms.dt,te,bigD,litD);
bvalN = linspace(0,2,11);

[sig,intsig] = dwmriWAVE(parms,Gdiff,bvalN);

b = GxtoB(Gdiff,parms.dt);
gmax = sqrt(bvalN/b);
gmax(b==0) = 0;
ana = intsig(1)*vanGelderen(bigD,litD,gmax,axd*g/2,parms.D(3));

figure
plot(bvalN,intsig,'o',bvalN,ana,'-',bvalN,intsig-ana,'^')
grid on
xlabel('b-value (ms/\mu m^2)')
ylabel('intra-axonal signal')

print('-dpng',fullfile('images','intrasig.png'));

fprintf(fid,'\n');
fprintf(fid,['![](' fullfile('images','intrasig.png') ')\n']);
fprintf(fid,'\n');
fprintf(fid,'RMSE: %e\n\n',norm(intsig-ana));
fprintf(fid,'Note: the van Gelderen approximation is only valid for low b*Dapp\n');
