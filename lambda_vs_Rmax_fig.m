% UM data
addpath('../')
addpath('experiments/')
load('Processed_Data.mat')
markers = {'o', 's', '^', '>', 'v', '<', 'd'};
hold on
k = 1;
for i = 7:15
    a1 = scatter(expts(i).R0.*1e6, 1/(expts(i).Req/expts(i).R0), 1000, 'Marker', markers{1}, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
    Rmax_a1(k) = expts(i).R0;
    lambda_a1(k) = 1/(expts(i).Req/expts(i).R0);
    k = k+1;
end
k = 1;
for i = 78:98%125:135%
    a2 = scatter(expts(i).R0.*1e6, 1/(expts(i).Req/expts(i).R0), 1000, 'Marker', markers{2}, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
    Rmax_a2(k) = expts(i).R0;
    lambda_a2(k) = 1/(expts(i).Req/expts(i).R0);
    k = k+1;
end
k = 1;
for i = 223:234
    a3 = scatter(expts(i).R0.*1e6, 1/(expts(i).Req/expts(i).R0), 1000, 'Marker', markers{3}, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
    Rmax_a3(k) = expts(i).R0;
    lambda_a3(k) = 1/(expts(i).Req/expts(i).R0);
    k = k+1;
end
hold off

% UTA data
hold on
load('./experiments/10percent/Rt_nondim_exp.mat')
k = 1;
for i = 1:length(R_nondim_All)
    b1 = scatter(nanmax(R_nondim_All{i})*RmaxList(i).*1e6,1/mean(R_nondim_All{i}(end-floor(0.1*length(R_nondim_All{i}))+1:end)), 1000, 'Marker', markers{4}, 'MarkerFaceColor', 'c', 'MarkerEdgeColor', 'k');
    Rmax_b1(k) = nanmax(R_nondim_All{i})*RmaxList(i);
    lambda_b1(k) = 1/(mean(R_nondim_All{i}(end-floor(0.1*length(R_nondim_All{i}))+1:end)));
    k = k+1;
end
load('./experiments/Ag5.0/Rt_nondim_exp.mat')
k = 1;
for i = [1:5,7:10]
    b2 = scatter(nanmax(R_nondim_All{i})*RmaxList(i).*1e6,1/(mean(R_nondim_All{i}(end-floor(0.1*length(R_nondim_All{i}))+1:end))), 1000, 'Marker', markers{5}, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k');
    Rmax_b2(k) = nanmax(R_nondim_All{i})*RmaxList(i);
    lambda_b2(k) = 1/(mean(R_nondim_All{i}(end-floor(0.1*length(R_nondim_All{i}))+1:end)));
    k = k+1;
end
% load('./forbenchmarks/Rt_nondim_exp_PEGDA_S3L6.mat')
% k = 1;
% for i = 1:length(R_nondim_All)
%     b3 = scatter(nanmax(R_nondim_All{i})*RmaxList(i).*1e6,1/(mean(R_nondim_All{i}(end-floor(0.1*length(R_nondim_All{i}))+1:end))), 1000, 'Marker', markers{6}, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k');
%     Rmax_b3(k) = nanmax(R_nondim_All{i})*RmaxList(i);
%     lambda_b3(k) = 1/(mean(R_nondim_All{i}(end-floor(0.1*length(R_nondim_All{i}))+1:end)));
%     k = k+1;
% end
hold off

set(gca, 'TickLabelInterpreter', 'latex','FontSize',35);
ax = gca; % Get the current axis handle
ax.LineWidth = 2;
xlabel('$R_{\mathrm{max}}$ ($\mu$m)', 'Interpreter', 'latex')
ylabel('$\Lambda = \frac{R_{\mathrm{max}}}{R_{\mathrm{eq}}}$', 'Interpreter', 'latex')
pbaspect([1.5 1 1]);
box on
xlim([125 375])
ylim([2 12])
Rmax = mean([Rmax_a1,Rmax_a2,Rmax_a3,Rmax_b1,Rmax_b2]);
Rmax_std = std([Rmax_a1,Rmax_a2,Rmax_a3,Rmax_b1,Rmax_b2]);
lambda = mean([lambda_a1,lambda_a2,lambda_a3,lambda_b1,lambda_b2]);
lambda_std = std([lambda_a1,lambda_a2,lambda_a3,lambda_b1,lambda_b2]);
Req = Rmax/lambda;

formatPlot

function formatPlot()
    ax  = gca;
    fig = gcf;

    % ---------- Make window as large as possible with fixed aspect ratio ----------
    ar = 1.5;                                 % desired width/height (match pbaspect)
    scr = get(0,'ScreenSize');                % [left bottom width height] (px)
    pad = 60;                                 % margin for OS chrome

    scrW = scr(3) - 2*pad;
    scrH = scr(4) - 2*pad;

    W = scrW; H = W/ar;                       % try width-limited
    if H > scrH                                % if too tall, limit by height instead
        H = scrH; W = H*ar;
    end

    left   = pad + (scr(3) - 2*pad - W)/2;    % center on screen
    bottom = pad + (scr(4) - 2*pad - H)/2;

    set(fig, 'Units','pixels', 'Position',[left bottom W H]);

    % ---------- Axes styling ----------
    box(ax,'on');
    set(ax,'LineWidth',2,'FontSize',36,'TickLabelInterpreter','latex');
    pbaspect(ax,[ar 1 1]);                    % keep same visual ratio as window

    % ---------- Tighten horizontally only (keep vertical unchanged) ----------
    drawnow;                                  % update layout metrics
    set(ax,'Units','normalized');

    % Avoid extra padding when exporting
    set(ax,'LooseInset', get(ax,'TightInset'));

    ti  = get(ax,'TightInset');               % [left bottom right top] (norm)
    pos = get(ax,'Position');                 % [x y w h] (norm)

    % keep current y, h; adjust x and w using left/right insets
    pos(1) = ti(1);                           % left edge clears y labels
    pos(3) = 1 - ti(1) - ti(3);               % fill remaining width
    set(ax,'Position',pos);

    % ---------- Clean vector export defaults ----------
    set(fig,'Renderer','painters','Color','w');
    set(ax,'Color','w');                   % transparent axes bg
end



