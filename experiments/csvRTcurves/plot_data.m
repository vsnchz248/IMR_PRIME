% 1) add your CSV folder to the path
addpath('./csvRTcurves/rtCurves/');

% 2) load raw data
A = readmatrix('histotripsy3.csv');   % 1462×2
x = A(:,1);
y = A(:,2);

% 3) detect new‐curve boundaries (where x decreases)
wrapIdx    = find( diff(x) < 0 );
boundaries = [1; wrapIdx+1; numel(x)+1];

% 4) physical constants
rho = 101325;    % Pa
p8  = 1064;      % same units as in your scale

% 5) crop threshold (on raw x)
xMin = -200;     

% 6) plot each segment
figure; hold on;
for k = 1:numel(boundaries)-1
    i1 = boundaries(k);
    i2 = boundaries(k+1)-1;
    
    xi_full = x(i1:i2);
    yi_full = y(i1:i2);
    
    % 6a) drop raw‐x < xMin
    keep = xi_full >= xMin;
    xi = xi_full(keep);
    yi = yi_full(keep);
    
    % 6b) compute Rmax from the full segment
    Rmax_k = max(yi_full);
    
    % 6c) normalize
    scaleX   = Rmax_k * sqrt(rho / p8);
    xi_norm  = xi / scaleX;
    yi_norm  = yi / Rmax_k;
    
    plot(xi_norm, yi_norm, 'LineWidth', 1);
end
hold off;

% 7) decorate
xlabel('$t/t_c$', 'Interpreter','latex');
ylabel('$R/R_{\max}$',            'Interpreter','latex');

% 8) legend
nCurves = numel(boundaries)-1;
leg = arrayfun(@(n) sprintf('Curve %d',n), 1:nCurves, 'UniformOutput',false);

