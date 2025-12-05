function [Rdot] = pade(t,R)
% Padé approximation (Ax=B)
j = 1;
k = 1;
for i = 1:length(t)
    if isnan(t(i)) || isnan(R(i))
        row(j) = i;
        j = j + 1;
    else
        tin(k,1) = t(i);
        Rin(k,1) = R(i);
        k = k + 1;
    end
end

n = length(tin);
h1 = diff(tin);
h = [h1; h1(end)];

A = zeros(n,n);
C = spdiags([ones(n,1) 4*ones(n,1) ones(n,1)],-1:1,n,n);
A = A + C;
A(1,1:2) = [1,2];
A(end,end-1:end) = [2,1];

B = zeros(n,1);
B(1,1) = (-5/2)*Rin(1)+2*Rin(2)+(1/2)*Rin(3);
B(end,1) = (5/2)*Rin(end)-2*Rin(end-1)-(1/2)*Rin(end-2);
for i = 2:n-1
    if isnan(B(i,1))
        continue
    else
    B(i,1) = 3*(Rin(i+1)-Rin(i-1));
    end
end
B = (1./h).*B;
Rdotinit = A \ B;
Rdot = zeros(length(t),1);
j = 1;
for i = 1:length(t)
    if isnan(R(i))
        Rdot(i) = NaN;
    else
        Rdot(i) = Rdotinit(j);
        j = j + 1;
    end
end

end