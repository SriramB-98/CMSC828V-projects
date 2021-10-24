function [w,f,normgrad] = SNAG(fun,gfun,Xtrain,~,w,bsz,kmax,tol)
alpha = 0.05;
n = size(Xtrain,1);
I = 1:n;
wx = w; 
wy = w;
f = zeros(kmax,1);
normgrad = zeros(kmax,1);
for k = 1 : kmax
    Ig = randperm(n,bsz);
    g = gfun(Ig,wx);
    normgrad(k) = norm(g);
    f(k) = fun(I,wx);
    mu = 1 - 3/(5 + k); 
    wx_new = wy - alpha*g;
    wy_new = (1 + mu)*wx_new - mu*wx; 
    wx = wx_new;
    wy = wy_new;
    if mod(k,100)==0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k),normgrad(k));
    end
    if normgrad(k) < tol
        break;
    end
end
w = wx_new;
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k),normgrad(k));
end
        
        
    
    
