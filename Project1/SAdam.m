function [w,f,normgrad] = SAdam(fun,gfun,Xtrain,~,w,bsz,kmax,tol)
n = size(Xtrain,1);
I = 1:n;
m_1 = zeros(size(w));
m_2 = zeros(size(w));
f = zeros(kmax,1);
alpha = 0.002;
beta_1 = 0.9;
beta_2 = 0.999;
epsilon = 1e-12;
normgrad = zeros(kmax,1);
for k = 1 : kmax
    Ig = randperm(n,bsz);
    g = gfun(Ig,w);
    normgrad(k) = norm(g);
    f(k) = fun(I,w);
    m_1 = beta_1*m_1 + (1 - beta_1)*g;
    m_2 = beta_2*m_2 + (1 - beta_2)*g.^2;
    
    m_1_hat = m_1/(1 - beta_1^k);
    m_2_hat = m_2/(1 - beta_2^k);
    
    w = w - alpha*m_1_hat./(sqrt(m_2_hat) + epsilon);
    if mod(k,100)==0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k),normgrad(k));
    end
    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k),normgrad(k));
end
        
        
    
    
