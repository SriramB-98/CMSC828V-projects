function [w,f_arr,normgrad] = SLBFGS(fun,gfun,Xtrain,w,bsz,bszH,M,eta,kmax,tol)
tic;
n = size(Xtrain,1);
iter = 1;
nor = tol+1;
m = 5;
% gam = 0.9; % line search step factor
% jmax = 10; %ceil(log(1e-14)/log(gam)); % max # of iterations in line search
dim = size(w, 1);
s = zeros(dim,m);
y = zeros(dim,m);
rho = zeros(1,m);
%IH = randperm(n, bszH);      % random index selection
Ig = randperm(n,bsz);
g = gfun(Ig, w);
wnew = w - eta*g;
gnew = gfun(Ig, wnew);
s(:,1) = wnew - w;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
Ig = randperm(n,bsz);
g = gfun(Ig, w);
num_batches = ceil(n/bsz);
update_freq = ceil(num_batches/5);
f_arr = zeros(kmax*6,1);
normgrad = zeros(kmax*6,1);
plot_it = 1;
while nor > tol && iter <= kmax
    for it = 1: num_batches
        if num_batches*(iter - 1) + it < m
            In = 1 : num_batches*(iter - 1) + it;  
%             s(:,In)
%             y(:,In)
%             rho(In)
            p = finddirection(g,s(:,In),y(:,In),rho(In));
        else
            p = finddirection(g,s,y,rho);
        end
        
%         [a,j, ~] = linesearch(Ig,w,p,g,fun,eta,gam,jmax);
%         if j == jmax
%             p = -g;
%             [a,~, ~] = linesearch(Ig,w,p,g,fun,eta,gam,jmax);
%         end
        step = eta*p;
        wnew = w + step;
        IH = randperm(n, bszH);
        Ig = IH(:, 1:bsz);
        gnew = gfun(Ig, wnew);
        
        if (mod(num_batches*(iter - 1) + (it-1),M) == 0)
            gnewH = gfun(IH, wnew);
            gH = gfun(IH, w);
            % replace oldest (s,y) vector pair and associated rho step
            s = circshift(s,[0,1]); 
            y = circshift(y,[0,1]);
            rho = circshift(rho,[0,1]);
            s(:,1) = step;
            y(:,1) = gnewH - gH;
            rho(1) = 1/(step'*y(:,1));
        end
        w = wnew;
        g = gnew;
        if mod(it, update_freq) == 0
            nor = norm(g);
            %a = num_batches*(iter - 1) + floor((it)/update_freq)
            f_arr(plot_it) = fun(1:n, w);
            normgrad(plot_it) = nor;
            plot_it = plot_it + 1;
        end
    end
    %fprintf('k = %d, f = %d, ||g|| = %d\n',iter,f_arr(plot_it-1),normgrad(plot_it-1));
    iter = iter + 1;

end
iter = iter - 1;
fprintf('k = %d, f = %d, ||g|| = %d\n',iter,f_arr(iter),normgrad(iter));
toc;

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
    
