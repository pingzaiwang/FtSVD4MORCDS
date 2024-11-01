function [U, V, M_rec, Rtest_hist, Rtrain_hist,R11_hist,R12_hist,R21_hist,Rtest] = f_tensor_completion(M, Mask, r, algConfig)
% Input:
%   M: partially observed matrix of size m x n x n3
%   Mask: binary matrix of size m x n x n3, indicating observed entries
%   Mtrue: true matrix of size m x n x n3 (used for error computation)
%   r: target rank
%   max_iter: maximum number of iterations
%   tol: tolerance for convergence
% Output:
%   U, V: factor matrices of size m x r x n3, n x r x n3
%   M_rec: reconstructed matrix of size m x n x n3
%   err_hist: history of reconstruction error (observed entries)
%   true_err_hist: history of reconstruction error (all entries)

max_iter=algConfig.max_iter;
tol=algConfig.tol;
Mtrue=algConfig.Mtrue;
m1=algConfig.m1;
n1=algConfig.n1;
% transform
L=algConfig.L;
verbose=algConfig.verbose;

[m, n, d3] = size(M);
% initialize U V using the subspace assumpion

U = randn(m, r, d3);
V = randn(n, r, d3);
 
Mask_ten = tensor_from_matrix(Mask, d3);
Remp_hist = zeros(max_iter, 1);
Rtrain_hist = zeros(max_iter, 1);
R11_hist = zeros(max_iter, 1);
R12_hist = zeros(max_iter, 1);
R21_hist = zeros(max_iter, 1);
Rtest_hist = zeros(max_iter, 1);

% transform the data into Transformer


for iter = 1:max_iter
    U_old = U;
    V_old = V;

    % Update U
    for i = 1:m
        idx = find(Mask(i,:));
        V_sub = V(idx,:,:);
        M_sub = M(i,idx,:);
        % t-product
        MV = tprod(M_sub, V_sub, L);
        VtV = tprod(tran(V_sub, L), V_sub, L);
        iVtV = teninv(VtV, L);
        U(i,:,:) = tprod(MV, iVtV, L);
 
    end

    % Update V
    for j = 1:n
        idx = find(Mask(:,j));
        U_sub = U(idx,:,:);
        M_sub = M(idx,j,:);

        U_sub_t = tran(U_sub, L);
        UM = tprod(U_sub_t, M_sub, L);
        UMt = tran(UM, L);
        UtU = tprod(U_sub_t, U_sub, L);
        iUtU = teninv(UtU, L);
        V(j,:,:) = tprod(UMt, iUtU, L);
 
    end

    % Recovery and Error
    Vt = tran(V, L);
    M_rec = tprod(U, Vt, L); ErrAll = M_rec - Mtrue;

    % Empirical Risk
    Remp = norm((M_rec(:) - M(:) ) .* Mask_ten(:), 'fro')^2/sum(Mask(:));
    Remp_hist(iter) = Remp;

    % Population Riks on Dtrain
    ErrTrain = ErrAll;
    ErrTrain(m1+1:end,n1+1:end, :)=0;
    Rtrain = norm(ErrTrain(:), 'fro')^2/(m*n-(m-m1)*(n-n1));
    Rtrain_hist(iter) = Rtrain;

    %Population Risk on D11
    Err11 = ErrAll(1:m1,1:n1,:);
    R11 = norm(Err11(:), 'fro')^2/(m1*n1);
    R11_hist(iter) = R11;

    %Population Risk on D12
    Err12 = ErrAll(1:m1,n1+1:end,:);
    R12 = norm(Err12(:), 'fro')^2/(m1*(n-n1));
    R12_hist(iter) = R12;

    %Population Risk on D21
    Err21 = ErrAll(m1+1:end,1:n1,:);
    R21 = norm(Err21(:), 'fro')^2/((m-m1)*n1);
    R21_hist(iter) = R21;

    % Population Riks on Dtest
    Err22 = ErrAll(m1+1:end,n1+1:end,:);
    Rtest = norm(Err22(:), 'fro')^2/((m-m1)*(n-n1));
    Rtest_hist(iter) = Rtest;

    if rem(iter,10)==1 && verbose
        fprintf('ERM:Iteration %d: Remp = %.2e, Rtrain = %.2e, R11 = %.2e, R12=%.2e, R21=%.2e, Rtest = %.2e\n', iter, Remp, Rtrain, R11,R12,R21, Rtest);
    end

    % Check convergence
    if norm(U(:) - U_old(:), 'fro')/norm(U(:),'fro') < tol && norm(V(:) - V_old(:), 'fro')/norm(V(:),'fro') < tol
        break;
    end
end

Vt = tran(V, L);
M_rec = tprod(U, Vt, L);

% Print final results
if verbose
    fprintf('+++ERM Finished!\nIteration %d: Remp = %.2e, Rtrain = %.2e, R11 = %.2e, R12=%.2e, R21=%.2e, Rtest = %.2e\n\n\n', iter, Remp, Rtrain, R11,R12,R21, Rtest);
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
    title('Reconstruction Error History of ERM');
    legend;
    grid on;
    pause(0.1)
end
end