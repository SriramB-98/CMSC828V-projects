function mnist_2categories_quad_hypersurface(nPCA, opt)
close all
fsz = 20;
fprintf('nPCA: %d, opt %s \n', nPCA, opt);
%% Pick the number of PCAs for the representation of images
% nPCA = 10;
%%
mdata = load('mnist.mat');
imgs_train = mdata.imgs_train;
imgs_test = mdata.imgs_test;
labels_test = mdata.labels_test;
labels_train = mdata.labels_train;
%% find 1 and 7 in training data
ind1 = find(double(labels_train)==2);
ind2 = find(double(labels_train)==8);
n1train = length(ind1);
n2train = length(ind2);
fprintf("There are %d 1's and %d 7's in training data\n",n1train,n2train);
train1 = imgs_train(:,:,ind1);
train2 = imgs_train(:,:,ind2);
%% find 1 and 7 in test data
itest1 = find(double(labels_test)==2);
itest2 = find(double(labels_test)==8);
n1test = length(itest1);
n2test = length(itest2);
fprintf("There are %d 1's and %d 7's in test data\n",n1test,n2test);
test1 = imgs_test(:,:,itest1);
test2 = imgs_test(:,:,itest2);
%% plot some data from category 1
% figure; colormap gray
% for j = 1:20
%     subplot(4,5,j);
%     imagesc(train1(:,:,j));
%     axis off
% end
% %% plot some data from category 2
% figure; colormap gray
% for j = 1:20
%     subplot(4,5,j);
%     imagesc(train2(:,:,j));
%     axis off
% end
%% use PCA to reduce dimensionality of the problem to 20
[d1,d2,~] = size(train1);
X1 = zeros(n1train,d1*d2);
X2 = zeros(n2train,d1*d2);
for j = 1 : n1train
    aux = train1(:,:,j);
    X1(j,:) = aux(:)';
end
for j = 1 :n2train
    aux = train2(:,:,j);
    X2(j,:) = aux(:)';
end
X = [X1;X2];
% D1 = 1:n1train;
% D2 = n1train+1:n1train+n2train;
[U,~,~] = svd(X','econ');
% esort = diag(Sigma);
% figure;
% plot(esort,'.','Markersize',20);
% grid;
Xpca = X*U(:,1:nPCA); % features
% figure; 
% hold on; grid;
% plot3(Xpca(D1,1),Xpca(D1,2),Xpca(D1,3),'.','Markersize',20,'color','k');
% plot3(Xpca(D2,1),Xpca(D2,2),Xpca(D2,3),'.','Markersize',20,'color','r');
% view(3)
%% split the data to training set and test set
Xtrain = Xpca;
Ntrain = n1train + n2train;
Xtest1 = zeros(n1test,d1*d2);
for j = 1 : n1test
    aux = test1(:,:,j);
    Xtest1(j,:) = aux(:)';
end
for j = 1 :n2test
    aux = test2(:,:,j);
    Xtest2(j,:) = aux(:)'; %#ok<AGROW>
end
Xtest = [Xtest1;Xtest2]*U(:,1:nPCA);
%% category 1 (1): label 1; category 2 (7): label -1
label = ones(Ntrain,1);
label(n1train+1:Ntrain) = -1;
%% dividing hyperplane: w'*x + b
%% optimize w and b using a smooth loss function and SINewton
dim = nPCA;
w = ones(dim^2+dim+1,1);
%w(1:dim^2) = reshape(eye(dim),1,[]);
kmax = 30; % the max number of iterations
tol = 1e-6;
% call the optimizer
%opt = 'GaussNewton';
if strcmp(opt, 'GaussNewton')
    [w,f,gnorm] = GaussNewton(@fun,Xtrain,label,w,kmax,tol);
elseif strcmp(opt, 'LevenbergMaquardt')
    [w,f,gnorm] = LevenbergMaquardt(@fun,Xtrain,label,w,kmax, tol);
end
% plot the objective function
figure;
plot(f,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('f','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');
savefig(strcat(opt,'_f_iter_', string(nPCA)));
% plot the norm of the gradient
pause(5);
figure;
plot(gnorm,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('||g||','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');
savefig(strcat(opt,'_gradf_iter_', string(nPCA)));

%% apply the results to the test set
Ntest = n1test+n2test;
testlabel = ones(Ntest,1);
testlabel(n1test+1:Ntest) = -1;
W = reshape(w(1:dim^2),[dim,dim]);
v = w(dim^2+1:dim^2+dim);
b = w(end);
qterm = diag(Xtest*W*Xtest');
test = testlabel.*qterm + ((testlabel*ones(1,dim)).*Xtest)*v + testlabel*b;
hits = find(test > 0);
misses = find(test < 0);
nhits = length(hits);
nmisses = length(misses);
fprintf('n_correct = %d, n_wrong = %d, accuracy %d percent\n',nhits,nmisses,100*nhits/Ntest);

end
%%
%% The objective function
function q = myquadratic(X,y,w)
[N,d] = size(X);
d2 = d^2;
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = zeros(N,1);
for i=1:N
     qterm(i) = y(i)*X(i,:)*W*X(i,:)';
end
q = qterm + ((y*ones(1,d)).*X)*v + y*b;
end

function f = fun(X,y,w)
q = myquadratic(X,y,w);
f = 0.5*sum((log(1 + exp(-q) ) ).^2);
end