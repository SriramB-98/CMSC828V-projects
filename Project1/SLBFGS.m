function [w,f_arr,normgrad] = SLBFGS(fun,gfun,Xtrain,~,w,bsz,kmax,tol)
n = size(Xtrain,1);
f_arr = zeros(kmax,1);
normgrad = zeros(kmax,1);
iter = 1;
nor = tol+1;
m = 5;
eta = 0.1;
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
dim = size(w, 1);
s = zeros(dim,m);
y = zeros(dim,m);
rho = zeros(1,m);
Ig = randperm(n,bsz);
g = gfun(Ig, w);
while nor > tol && iter <= kmax
    if iter < m
        p = finddirection(g,s(:,1 : iter),y(:,1 : iter),rho(1 : iter));
    else
        p = finddirection(g,s,y,rho);
    end
    [a,j, f] = linesearch(Ig,w,p,g,fun,eta,gam,jmax);
    if j == jmax
        p = -g;
        [a,~, f] = linesearch(Ig,w,p,g,fun,eta,gam,jmax);
    end
    step = a*p;
    wnew = w + step;
    Ig = randperm(n,bsz);
    gnew = gfun(Ig, wnew);
    s = circshift(s,[0,1]); 
    y = circshift(y,[0,1]);
    rho = circshift(rho,[0,1]);
    s(:,1) = step;
    y(:,1) = gnew - g;
    rho(1) = 1/(step'*y(:,1));
    w = wnew;
    g = gnew;
    nor = norm(g);
    f_arr(iter) = f;
    normgrad(iter) = nor;
    fprintf('k = %d, f = %d, ||g|| = %d\n',iter,f_arr(iter),normgrad(iter));
    iter = iter + 1;
end
iter = iter - 1;
fprintf('k = %d, f = %d, ||g|| = %d\n',iter,f_arr(iter),normgrad(iter));
end

function p = finddirection(g,s,y,rho)
% input: g = gradient dim-by-1
% s = matrix dim-by-m, s(:,i) = x_{k-i+1}-x_{k-i}
% y = matrix dim-by-m, y(:,i) = g_{k-i+1}-g_{k-i}
% rho is 1-by-m, rho(i) = 1/(s(:,i)'*y(:,i))
m = size(s,2);
a = zeros(m,1);  
for i = 1 : m
    a(i) = rho(i)*s(:,i)'*g;
    g = g - a(i)*y(:,i);
end
gam = s(:,1)'*y(:,1)/(y(:,1)'*y(:,1)); % H0 = gam*eye(dim)
g = g*gam;
for i = m :-1 : 1
    aux = rho(i)*y(:,i)'*g;
    g = g + (a(i) - aux)*s(:,i);
end
p = -g;
end
        
function [a,j, f1] = linesearch(I, x,p,g,fun,eta,gam,jmax)
    a = 1;
    f0 = fun(I,x);
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        f1 = fun(I,xtry);
        if f1 < f0 + a*aux
            break;
        else
            a = a*gam;
        end
    end
end
    
