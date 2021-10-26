function [w,f, normgrad] = LevenbergMaquardt(fun,X, y,w,kmax, tol)
tic;
delta_max = 100;
delta = 0.2*delta_max;
eta = 0.1;
f = zeros(kmax + 1, 1);
normgrad = zeros(kmax,1);
for k = 1 : kmax
    [r,J] = Res_and_Jac(X,y,w);
    g = (J'*r);
    [p, m] = LMOptimize(J, g, delta);
    new_w = w + p;
    f(k) = fun(X, y, w);
    normgrad(k) = norm(g);
    rho = (f(k) - fun(X, y, new_w))/(- m);
    if rho < 1/4
        delta = 0.25*delta;
    elseif rho > 3/4 && norm(p) == delta
        delta = min(2*delta, delta_max);
    end
    if rho > eta
        w = w + p;
    end 
    if normgrad(k) < tol || (k > 5 && f(k - 5) - f(k) <= 0)
        break
    end
    %fprintf('%d, %d\n', k, f(k));
end
fprintf('k = %d, delta = %d, f = %d, ||g|| = %d\n',k, delta, f(k),normgrad(k));
toc;
end


function [p, m] = LMOptimize(J, g, delta)
% do Tikhonov regularization for the case J is rank-deficient
I = eye(size(J,2));
B = J'*J + (1e-6)*I;
pstar = -B\g; % unconstrained minimizer
if norm(pstar) <= delta
    p = pstar;
else % solve constrained minimization problem
    lam = 1; % initial guess for lambda
    while 1
        B1 = B + lam*I;
        C = chol(B1); % do Cholesky factorization of B
        p = -C\(C'\g); % solve B1*p = -g
        np = norm(p);
        dd = abs(np - delta); % R is the trust region radius
        if dd < 1e-6
            break
        end
        q = C'\p; % solve C^\top q = p
        nq = norm(q);
        lamnew = lam + (np/nq)^2*(np - delta)/delta;
        if lamnew < 0
            lam = 0.5*lam;
        else
            lam = lamnew;
        end
    end
end
m =  p'*(g) + 0.5*(p')*B*(p);
end