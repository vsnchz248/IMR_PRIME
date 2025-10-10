close all;

rho = 1050; 
pinf = 101325; % (Pa) Atmospheric Pressure 
 
deltat = 1/2e6; deltax = 1.5119e-6; % um/px

load('1E6_Ag_RofTdata_0.mat')

% colorRGB = [ [0, 0.4470, 0.7410]	          	 
%           	[0.8500, 0.3250, 0.0980]	          
%           	[0.9290, 0.6940, 0.1250]	          
%           	[0.4940, 0.1840, 0.5560]	           
%           	[0.4660, 0.6740, 0.1880]	          	 
%           	[0.3010, 0.7450, 0.9330]	          	 
%           	[0.6350, 0.0780, 0.1840] ];  
        
% clRGB = [243,104,86]/255;
% clRGB = [240,176,61]/255;
% clRGB = [81,186,163]/255;
% clRGB = [17,134,134]/255;
 clRGB = [20,47,64]/255;

colorRGB = repmat(clRGB,7,1);


%%


figure,plot(1e6*t,1e6*Rnew,'.-')
  
set(gca,'fontsize',18)
set(gcf,'color','w')
xlabel('Time t (\mus)');
ylabel('Bubble R (\mum)');

%%%%% Rmax %%%%%
Rmax = Rnew(1) 

%%%%% Mean equilibrium radius %%%%%
Req_mean = mean(Rnew(end-30:end))
Req_std = std(Rnew(end-30:end))
lambda_max_mean = Rmax/Req_mean
lambda_max_std = 0.5*(Rmax/(Req_mean-Req_std) - Rmax/(Req_mean+Req_std))

%% Load all other datasets
close all;
files = dir('2E6*.mat'); t_nondim_All = []; R_nondim_All = [];
 
for tempi = [ 1:5,7:10 ]
    load(files(tempi).name );

    figure(1); hold on;
    plot(1e6*t,1e6*Rnew,'.','color',colorRGB(1+mod(tempi,7),:));
    tq = t(2):1e-7:t(end);
    Rq = interp1(t,Rnew,tq,'pchip');  
    hold on; plot(1e6*tq,1e6*Rq,'-','color',colorRGB(1+mod(tempi,7),:));

    %%%%% Normalization %%%%%
    Rmax = Rnew(2); RmaxList(tempi) = Rmax; 
    R_nondim = Rnew(2:end)/Rmax; lambdaList(tempi) = Rnew(2)/Rmax;
    t_nondim = t(2:end)*sqrt(pinf/rho)/Rmax;

    t_nondim_All{tempi} = [t_nondim];
    R_nondim_All{tempi} = [R_nondim];
    figure(2); hold on;
    plot(t_nondim,R_nondim*lambda_max_mean,'.','color',colorRGB(1+mod(tempi,7),:));

end

fig1=figure(1); 
adjust_fig(fig1,[],[],'','');
set(gca,'fontsize',12); set(gcf,'color','w');
xlabel('Time $t$($\mu$s)','interpreter','latex');
ylabel('$R$($\mu$m)','interpreter','latex');
box on; axis([0,80,0,300]); grid on; grid minor;

fig2=figure(2); 
adjust_fig(fig2,[],[],'','');
set(gca,'fontsize',12); set(gcf,'color','w');
xlabel('Normalized time $t^{*}$','interpreter','latex');
ylabel('$\lambda$','interpreter','latex');
box on; axis([0,3,0,8.5]);  

RmaxList_mean = mean(RmaxList)
RmaxList_std = std(RmaxList)


save('Rt_nondim_exp.mat','t_nondim_All','R_nondim_All', ...
    'RmaxList','RmaxList_mean','RmaxList_std' ,...
    'Rmax','Req_mean','Req_std','lambda_max_mean','lambda_max_std');


