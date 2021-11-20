function isomap(dat_name)
fsz = 16;
dat = load(sprintf("%s.mat",dat_name));
X = dat.data3;

pcs = 10;
coeff = pca(X);
coeff = coeff(:,1:pcs);
X = X*coeff;

[n,dim] = size(X);
%% compute pairwise distances
d = zeros(n);
e = ones(n,1);
for i = 1 : n
    d(i,:) = sqrt(sum((X - e*X(i,:)).^2,2));
end
%% k-isomap
% STEP 1: find k nearest neighbors and define weighted directed graph
k = 15; % the number of nearest neighbors for computing distances
% for each point, find k nearest neighbors
ineib = zeros(n,k);
dneib = zeros(n,k);
for i = 1 : n
    [dsort,isort] = sort(d(i,:),'ascend');
    dneib(i,:) = dsort(1:k);
    ineib(i,:) = isort(1:k);
end
figure();
hold on;
plot3(X(:,1),X(:,2),X(:,3),'.','Markersize',15,'color','b');
daspect([1,1,1])
for i = 1 : n
    for j = 1 : k
        edge = X([i,ineib(i,j)],:);
        plot3(edge(:,1),edge(:,2),edge(:,3),'k','Linewidth',0.25);
    end
end
set(gca,'Fontsize',fsz);
view(3);
saveas(gcf, sprintf("%s_isomap_graph.png", dat_name))

%
% STEP 2: compute shortest paths in the graph
D = zeros(n);
ee = ones(1,k);
g = ineib';
g = g(:)';
w = dneib';
w = w(:)';
G = sparse(kron((1:n),ee),g,w);
G = G+abs(G-G');
m = 1;%randi(n);
mf = n;%randi(n);
c = zeros(n,3);
if dim ~= 3
    c = dat.colors;
end

for i = 1 : n
    [dist,path,~] = graphshortestpath(G,i);
    D(i,:) = dist;
    if i == m
        figure()
        hold on
        dmax = max(dist);
        N = 1000;
        col = parula(N);
        for ii = 1 : n
            if dim ==3
                c(ii,:) = col(getcolor(dist(ii),dmax,N),:);
            end
            plot3(X(ii,1),X(ii,2),X(ii,3),'.','Markersize',15,'color',c(ii,:));
        end
        p = path{[mf]};
        for j = 2 : length(p)
            I = [p(j-1),p(j)];
            plot3(X(I,1),X(I,2),X(I,3),'Linewidth',2,'color','r');
        end
        view(3)
        daspect([1,1,1])
        set(gca,'Fontsize',fsz);
        saveas(gcf, sprintf("%s_isomap_rand_path.png", dat_name))
    end
end

%
% STEP 3: do MDS
% symmetrize D
D = 0.5*(D + D');
% D(isinf(D)) = 100;
Y = mdscale(D,2, 'criterion', 'metricsstress');
figure();
hold on
for ii = 1 : n
    plot(Y(ii,1),Y(ii,2),'.','Markersize',15,'color',c(ii,:));
end
% plot edges
% for i = 1 : n
%     for j = 1 : k
%         edge = Y([i,ineib(i,j)],:);
%         plot(edge(:,1),edge(:,2),'r','Linewidth',0.25, 'color', 'k');
%     end
% end
% plot path
for j = 2 : length(p)
    I = [p(j-1),p(j)];
    plot(Y(I,1),Y(I,2),'Linewidth',2,'color','r');
end
set(gca,'Fontsize',fsz);
daspect([1,1,1]);
saveas(gcf, sprintf("%s_isomap_rolled_out.png", dat_name))

end

%%
function c = getcolor(u,umax,N)
c = max(1,round(N*(u/umax)));
end


