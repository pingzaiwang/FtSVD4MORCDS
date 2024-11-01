clear; close all;
addpath('src')
m1=80; m2=120;
n1=80; n2=120;
 
% the third dimension
n3=5;
% parameter for single training
single_nu = 2;
% parameter for overparameter training
over_nu = 4;

% transform
L = dct(eye(n3));

% embedding dimensions to test
r = 15;
 
max_iter = 20;

% singular value decay rate
gamma=1;

% covariance shift parameter
kapcov = 1;
sr_train_arr = 0.5:0.01:0.9;

seeds = [2021,2022,2023,2024];
fid = fopen('results_sr_train.txt', 'w');

for seed = seeds

    rng(seed,'twister')
    fprintf(fid, 'Random seed: %d\n', seed);
    fprintf(fid, 'sr_train \tRtest1\tRtest22\n');


    for sr_train = sr_train_arr
        %%+++++Date generation+++++

        [U1,~,V1]=tsvds(randn(m1,m1,n3),r,L);
        [Q,~,~]=tsvds(randn(r,r,n3),r,L);

        sig1 = (1:r)*(-(1+gamma)/2);
        sig2 = kapcov*sig1.*rand(1,r);
        Sig1 = tensor_from_matrix(diag(sig1),n3);
        Sig2 = tensor_from_matrix(diag(sig2),n3);

        Sig1=inverselineartransform(Sig1,L);
        Sig2=inverselineartransform(Sig2,L);

        A1 = tprod_mul(L, U1, Sig1,tran(Q,L));
        B1 = tprod_mul(L, V1, Sig1, tran(Q,L));

        [U2,~,V2]=tsvds(randn(max(m2,r),max(m2,r),n3), r, L);
        A2 = tprod_mul(L,U2,Sig2,tran(Q,L));
        B2 = tprod_mul(L,V2,Sig2,tran(Q,L));

        U = zeros(m1+m2,r,n3);
        V = zeros(n1+n2,r,n3);
        U(1:m1,:,:) = A1;
        U(m1+1:end,:,:) = A2;
        V(1:n1,:,:) = B1;
        V(n1+1:end,:,:) = B2;
        X = tprod(U,tran(V,L),L);

        %%-----Date generation-----

        % Uniformly sample from X11,X12,X21
        % sampling rate
        srTrain1st=sr_train;
        sr11Train1st=0.1;
        % sampling step1: sample on all entries on X randomly
         
        MaskTrain1st = rand(m1+m2,n1+n2)<srTrain1st;
         
        Mask11Train1st = rand(m1,n1)<sr11Train1st;
         
        MaskTrain1st(1:m1,1:n1)=Mask11Train1st;
        % sampling step2: make the position of X22 unseen
        MaskTrain1st(m1+1:end,n1+1:end)=0;
        MaskTrain1st_ten = tensor_from_matrix(MaskTrain1st,n3);
        Xtrain1st= X.*MaskTrain1st_ten;


        %%+++++Algorithm 1++++++
        % Single ERM
        % preset rank
        k = round(single_nu*r);
        algConfig.Mtrue = X;
        algConfig.m1=m1;
        algConfig.n1=n1;
        algConfig.tol = 1e-10;
        algConfig.max_iter = max_iter;
        algConfig.plotFig = 0;
        algConfig.L = L;
        algConfig.verbose = 0;
        [~, ~, ~, ~, ~, ~, ~, ~, Rtest1] = f_tensor_completion(Xtrain1st, MaskTrain1st,k, algConfig);

        %%-----Algorithm 1------

        %%+++++Algorithms 2++++++
        % Double ERM with Semisupervised Learning
        % Step 1: Overparameterized training
        p = round(over_nu*r);
        algConfig.Mtrue = X;
        algConfig.m1=m1;
        algConfig.n1=n1;
        algConfig.tol = 1e-10;
        algConfig.max_iter = max_iter;
        algConfig.plotFig = 0;
        [Ahat21, Bhat21, ~, ~, ~, ~, ~, ~, ~] = f_tensor_completion(Xtrain1st, MaskTrain1st, p, algConfig);

        % Step 2: Sample on D11
        sr_cov = 1;
        mu = max(abs(X(:)))/sum(MaskTrain1st(:));
        Mask_SR_Cov_Row = rand(m1,1)<sr_cov;
        Mask_SR_Cov_Col = rand(n1,1)<sr_cov;
        n_sample_row = length(find(Mask_SR_Cov_Row));
        n_sample_col = length(find(Mask_SR_Cov_Col));

        Ahat21_sub = Ahat21(Mask_SR_Cov_Row,:,:);
        Bhat21_sub = Bhat21(Mask_SR_Cov_Col,:,:);

        Asub_Asub = tprod_mul(L, tran(Ahat21_sub, L), Ahat21_sub);
        Bsub_Bsub = tprod_mul(L, tran(Bhat21_sub, L), Bhat21_sub);
        eye_ten = tensor_from_matrix(eye(p),n3);

        CovA = (Asub_Asub+mu*eye_ten) ./ n_sample_row;
        CovB = (Bsub_Bsub+mu*eye_ten) ./ n_sample_col;

        % Step 3: Dimension Reduction
        rhat = r;

        CovA_half = sqrtm_ten(CovA,L);
        i_CovA_CovB_CovA = teninv(tprod_mul(L, CovA_half, CovB, CovA_half), L);
        sqrtm_i_CovA = sqrtm_ten(i_CovA_CovB_CovA, L);
        W = tprod_mul(L, CovA_half, sqrtm_i_CovA, CovA_half);
        W_sqrt = sqrtm_ten(W, L);
        SigBal = tprod_mul(L, W_sqrt, CovB, W_sqrt);
        [V, ~] = teig(SigBal, L);
        Vrhat = V(:, 1:rhat, :);
        Prhat = tprod_mul(L, Vrhat, tran(Vrhat,L));
        Qrhat = sqrtm_ten(tprod_mul(L, teninv(W,L), Prhat, sqrtm_ten(teninv(W,L),L)), L);
        % creat the psude-label: AB with dimensional reduction
        ABred=tprod_mul(L,Ahat21,Qrhat,tran(Bhat21,L));

        % Step 4: Supervised + Psudo-label
        SampleRatioTrain2ndTo1st = 1;
        MaskTrain2nd = rand(m1+m2,n1+n2)<srTrain1st*SampleRatioTrain2ndTo1st;
        Mask11Train2nd = rand(m1,n1)<sr11Train1st*SampleRatioTrain2ndTo1st;
        MaskTrain2nd(1:m1,1:n1)=Mask11Train2nd;
        MaskTrain2nd(m1+1:end,n1+1:end)=0;
        Xtrain2nd= X.*tensor_from_matrix(MaskTrain2nd,n3);

        SR11=1;
        Mask11 = zeros(m1+m2,n1+n2);
        Mask11(1:m1,1:n1) = rand(m1,n1)<SR11;

        algConfig.Mtrue = X;
        algConfig.m1=m1;
        algConfig.n1=n1;
        algConfig.nu = rhat^4;
        algConfig.tol = 1e-10;
        algConfig.max_iter = max_iter;
        algConfig.plotFig = 0;
        algConfig.U0=tprod(Ahat21,Vrhat,L);
        algConfig.V0=tprod(Bhat21,Vrhat,L);

        [~, ~, ~, ~, ~, ~, ~, ~, Rtest22] = f_tensor_completion_with_psudlabel(Xtrain2nd, ABred, MaskTrain2nd, Mask11, rhat, algConfig);

        fprintf(fid, '%.2f\t%.2e\t%.2e\n', sr_train, Rtest1, Rtest22);
         
    end
    fprintf(fid, '\n');
end
fclose(fid);   