function [w,f,normgrad] = SGD(fun,gfun,Xtrain,~,w,bsz,kmax,tol)
eta = 0.3;
n = size(Xtrain,1);
I = 1:n;
f = zeros(kmax,1);
normgrad = zeros(kmax,1);
for k = 1 : kmax
    Ig = randperm(n,bsz);
    g = gfun(Ig,w);
    normgrad(k) = norm(g);
    f(k) = fun(I,w);
    w = w - (eta)*g;
    if mod(k,100)==0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k),normgrad(k));
    end
    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k),normgrad(k));
end
        
        
    
    
