function PCAmap(dat_name, k)
fsz = 16;
dat = load(sprintf("%s.mat",dat_name));
X = dat.data3;
[n, d] = size(X);
N = 1000;
coeff = pca(X);
coeff = coeff(:,1:k);
X_pca = X*coeff;
size(X_pca)
if d == 3
    c = zeros(n,3);
    col = parula(N);
    for i = 1:n
        c(i,:) = col(getcolor(mod(i-1, 32),32,N),:);
    end
else
    c = dat.colors;
end

if d == 3
    figure();
    hold on;
    for i = 1:n
        plot3(X(i,1),X(i,2),X(i,3),'.','Markersize',15,'color',c(i, :));
    end
    set(gca,'Fontsize',fsz);
    view(3);
    daspect([1,1,1]);
    saveas(gcf, sprintf("%s_PCA_3d_colored.png", dat_name))
end

figure();
hold on;
for i = 1:n
    if k == 2
        plot(X_pca(i,1),X_pca(i,2),'.','Markersize',15,'color',c(i, :));
    elseif k ==3
        plot3(X_pca(i,1),X_pca(i,2),X_pca(i,3),'.','Markersize',15,'color',c(i, :));
    end 
end
set(gca,'Fontsize',fsz);
view(k);
saveas(gcf, sprintf("%s_PCA_unrolled_%dd.png", dat_name, k))
% for i = 1 : n
%     for j = 1 : k
%         edge = X([i,ineib(i,j)],:);
%         plot3(edge(:,1),edge(:,2),edge(:,3),'k','Linewidth',0.25);
%     end
% end
end

%%
function c = getcolor(u,umax,N)
c = max(1,round(N*(u/umax)));
end
