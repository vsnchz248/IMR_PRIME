A = readmatrix('laser0.csv');

k = 1;
for i = 1:length(A(:,1))
    if A(i,1) == -1000
        row(k) = i+1;
        k = k + 1;
    end
end

k = 1;
for i = 1:length(row)-1
    data{k} = A(row(i):row(i+1)-2,:);
    k = k + 1;
end
hold on
for i = 1:length(data)
    plot(data{i}(:,1),data{i}(:,2))
end
hold off