function [U, V, M_rec, Rtest_hist,Rtrain_hist,R11_hist,R12_hist,R21_hist,Rtest] = f_tensor_completion_with_psudlabel(M, Mpuso, Mask1, Mask2, r, algConfig)
% Input:
%   M, N: partially observed tensors of size m x n x n3
%   Mask1, Mask2: binary matrices of size m x n x n3, indicating observed entries
%   r: target rank
%   nu: regularization parameter
%   max_iter: maximum number of iterations
%   tol: tolerance for convergence
% Output:
%   U, V: factor matrices of size m x r x n3, n x r x n3
%   M_rec: reconstructed matrices of size m x n x n3
%   err_hist: history of reconstruction error
max_iter=algConfig.max_iter;
tol=algConfig.tol;
Mtrue=algConfig.Mtrue;
m1=algConfig.m1;
n1=algConfig.n1;
nu = algConfig.nu;
L = algConfig.L;
verbose = algConfig.verbose;

[m, n, d3] = size(M);

U = randn(m, r, d3);
V = randn(n, r, d3);

 

Remp_hist = zeros(max_iter, 1);
Rtrain_hist = zeros(max_iter, 1);
R11_hist = zeros(max_iter, 1);
R12_hist = zeros(max_iter, 1);
R21_hist = zeros(max_iter, 1);
Rtest_hist = zeros(max_iter, 1);

for iter = 1:max_iter
    U_old = U;
    V_old = V;

    % Update U
    for i = 1:m
        idx1 = find(Mask1(i,:));
        idx2 = find(Mask2(i,:));
        V_sub1 = V(idx1,:,:);
        V_sub2 = V(idx2,:,:);
        M_sub = M(i,idx1,:);
        N_sub = Mpuso(i,idx2,:);

        MV = tprod(M_sub, V_sub1, L);
        NV = tprod(N_sub, V_sub2, L);
        V1tV1 = tprod(tran(V_sub1,L), V_sub1, L);
        V2tV2 = tprod(tran(V_sub2,L), V_sub2, L);

        MV_NV = MV + nu*NV;
        i_V1V2 = teninv(V1tV1+nu*V2tV2, L);
        U(i,:,:) = tprod(MV_NV, i_V1V2, L);

        % U(i,:) = (M_sub * V_sub1 + nu * N_sub * V_sub2) * pinv(V_sub1' * V_sub1 + nu * V_sub2' * V_sub2);
    end

    % Update V
    for j = 1:n
        idx1 = find(Mask1(:,j));
        idx2 = find(Mask2(:,j));
        U_sub1 = U(idx1,:,:);
        U_sub2 = U(idx2,:,:);
        M_sub = M(idx1,j,:);
        N_sub = Mpuso(idx2,j,:);

        UM = tprod(tran(U_sub1, L), M_sub, L);
        UN = tprod(tran(U_sub2, L), N_sub, L);
        U1tU1 = tprod(tran(U_sub1,L), U_sub1, L);
        U2tU2 = tprod(tran(U_sub2,L), U_sub2, L);

        UM_UN = UM + nu*UN;
        i_U1U2 = teninv(U1tU1+nu*U2tU2, L);
        V(j,:,:) = tprod(tran(UM_UN,L), i_U1U2, L);
    end

    % Recovery and Error
    M_rec = tprod(U, tran(V,L), L);
    ErrAll = M_rec - Mtrue;

    % Empirical Risk
    mask_ten = tensor_from_matrix(Mask1,d3);
    Remp = norm((M_rec(:) - M(:)) .* mask_ten(:), 'fro')^2/sum(Mask1(:));
    Remp_hist(iter) = Remp;

    % Population Riks on Dtrain
    ErrTrain = ErrAll;
    ErrTrain(m1+1:end,n1+1:end)=0;
    Rtrain = norm(ErrTrain(:), 'fro')^2/(m*n-(m-m1)*(n-n1));
    Rtrain_hist(iter) = Rtrain;

    %Population Risk on D11
    Err11 = ErrAll(1:m1,1:n1);
    R11 = norm(Err11(:), 'fro')^2/(m1*n1);
    R11_hist(iter) = R11;

    %Population Risk on D12
    Err12 = ErrAll(1:m1,n1+1:end);
    R12 = norm(Err12(:), 'fro')^2/(m1*(n-n1));
    R12_hist(iter) = R12;

    %Population Risk on D21
    Err21 = ErrAll(m1+1:end,1:n1);
    R21 = norm(Err21(:), 'fro')^2/((m-m1)*n1);
    R21_hist(iter) = R21;

    % Population Riks on Dtest
    Err22 = ErrAll(m1+1:end,n1+1:end);
    Rtest = norm(Err22(:), 'fro')^2/((m-m1)*(n-n1));
    Rtest_hist(iter) = Rtest;

    if rem(iter,10)==1 && verbose
        fprintf('DS-ERM:Iteration %d: Remp = %.2e, Rtrain = %.2e, R11 = %.2e, R12=%.2e, R21=%.2e, Rtest = %.2e\n', iter, Remp, Rtrain, R11,R12,R21, Rtest);
    end

    % Check convergence
    if norm(U(:) - U_old(:), 'fro') < tol && norm(V(:) - V_old(:), 'fro') < tol
        break;
    end
end

M_rec = tprod(U, tran(V,L), L);

if verbose
    % Print final results
    fprintf('+++DS-ERM Finished:\nIteration %d: Remp = %.2e, Rtrain = %.2e, R11 = %.2e, R12=%.2e, R21=%.2e, Rtest = %.2e\n\n\n', iter, Remp, Rtrain, R11,R12,R21, Rtest);
end

% Plot reconstruction error history
if algConfig.plotFig
    figure;
    plot(1:iter, log10(Rtest_hist(1:iter)), '-r', 'LineWidth', 2,'DisplayName', 'Rtest');
    hold on;
    plot(1:iter, log10(Rtrain_hist(1:iter)), '-g', 'LineWidth', 1,'DisplayName', 'Rtrain');
    hold on;
    plot(1:iter, log10(R11_hist(1:iter)), '-b', 'LineWidth', 2,'DisplayName', 'R11');
    hold on;
    plot(1:iter, log10(R12_hist(1:iter)), '-m', 'LineWidth', 1,'DisplayName', 'R12');
    hold on;
    plot(1:iter, log10(R21_hist(1:iter)), '-k', 'LineWidth', 1,'DisplayName', 'R21');
    hold off;
    xlabel('Number of Iterations');
    ylabel('Reconstruction Error in Logarithmic Scale (Base 10)');
    title('Reconstruction Error History of DS-ERM');
    legend;
    grid on;
    pause(0.1)
end
end