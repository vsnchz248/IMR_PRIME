function formatPlot()
    box on
    set(gca, 'LineWidth', 2);
    set(gca, 'FontSize', 28);
    set(gca, 'TickLabelInterpreter','latex');
    pbaspect([1.5 1 1]);
end