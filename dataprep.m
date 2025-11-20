function [expR0,expReq,noisyR,noisyRdot,R_stats,Rdot_stats,Rdotvalues,t,dimtvector,indicator] = dataprep(tmatrix,Rmatrix,tc,Rdim,Rm)

tc_ave = mean(tc); % average characteristic time
t_nd = mean(tmatrix,2);

collapse_times = tmatrix(end,:);
[~, sort_idx] = sort(collapse_times, 'descend');
tfin_col_idx = sort_idx(2);
t_nd = tmatrix(:,tfin_col_idx);

% compute derivative to obtain Rdot
for i = 1:length(Rmatrix(1,:))
    Rdotvalues(:,i) = pade(tmatrix(:,i),Rmatrix(:,i));
end
Rdotvalues(1,:) = 0;
for i = 1:length(t_nd)
    R(i,1) = nanmean(Rmatrix(i,:)); % mean value of R for all data sets
    Rdot(i,1) = nanmean(Rdotvalues(i,:));
end

Rvec = nanmean(Rdim,2);
lastFewExpPoints = Rvec(end-round(length(R)*0.1)-1:end);
expR0 = mean(Rm);
expReq = mean(lastFewExpPoints);
ratio = expReq/expR0;

% generate noisy data
[noisyR,noisyRdot,R_stats,Rdot_stats,t] = GNI(t_nd,tmatrix,Rdim,Rmatrix,Rdotvalues);

collapse_times = tmatrix(end,:);
[~, sort_idx] = sort(collapse_times, 'descend');
tfin_col_idx = sort_idx(2);
tnoisy = tmatrix(:,tfin_col_idx);

final_tmatrix = [tnoisy, tmatrix];

% Get the number of rows and columns
[nRows, nCols] = size(final_tmatrix);

% Convert to a single column vector while keeping track of original indices
long_time_vector = final_tmatrix(:);  % Flatten matrix
column_indices = repelem(1:nCols, nRows)'; % Column tracking

% Sort time values while keeping track of indices
[sorted_time_vector, sort_idx] = sort(long_time_vector);
sorted_column_indices = column_indices(sort_idx);

% Get unique time values while maintaining order
[unique_time_vector, ~, unique_idx] = unique(sorted_time_vector, 'stable');

% Initialize indicator matrix
indicator = zeros(length(unique_time_vector), nCols);

% Populate indicator matrix
for i = 1:length(sorted_time_vector)
    row = unique_idx(i); % Row index in unique_time_vector
    col = sorted_column_indices(i); % Column index from original matrix
    indicator(row, col) = 1; % Mark presence
end
tvector = unique_time_vector.*mean(tc);
dimtvector = tvector;
end