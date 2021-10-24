function [w,f, normgrad] = GaussNewton(fun, X,y,w,kmax,tol)
tic;
gam = 0.9;
jmax = 10; % ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5;
n = size(w,1);
f = zeros(kmax + 1,1);
normgrad = zeros(kmax,1);
nfail = 0;
nfailmax = 5;
% size(X)
% size(y)
% size(w)
for k = 1 : kmax
    [r,J] = Res_and_Jac(X,y,w);
    B = J'*J + (1e-6)*eye(n);
    g = (J'*r);
    s = -B\g;
    a = 1;
    f0 = 0.5*sum(r.^2);
    f(k) = f0;
    normgrad(k) = norm(g);
    aux = eta*g'*s;
    for j = 0 : jmax
        wtry = w + a*s;
        f1 = fun(X, y, wtry);
        if f1 < f0 + a*aux
%             fprintf('Linesearch: j = %d, f1 = %d, f0 = %d, |as| = %d\n',j,f1,f0,norm(a*s));
            break;
        else
            a = a*gam;
        end
    end
    if j < jmax
        w = wtry;
    else
        nfail = nfail + 1;
    end
    if mod(k,100)==0
        fprintf('k = %d, a = %d, f = %d, ||g|| = %d\n',k,a,f(k+1),normgrad(k));
    end
    if nfail > nfailmax
        f(k+2:end) = [];
        normgrad(k+1:end) = [];
        fprintf('stop iteration as linesearch failed more than %d times\n',nfailmax);
        break;
    end
    if normgrad(k) < tol || (k > 5 && f(k - 5) - f(k) <= 0)
        break
    end
    %fprintf('%d, %d\n', k, f(k));
end
fprintf('k = %d, a = %d, f = %d, ||g|| = %d\n',k,a,f(k),normgrad(k));
toc;
end


