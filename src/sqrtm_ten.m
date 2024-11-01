function Y = sqrtm_ten(X, transform)
    % Y = sqrtm_ten(X, transform) computes the tensor square root of X under the t-SVD transform framework.
    %
    % Input:
    %       X - n*n*n3 tensor
    %       transform - a linear transform
    %
    % Output:
    %       Y - n*n*n3 tensor, the tensor square root of X
    %
    % version 1.0 - 05/16/2024
    %
    % Written by xxx (xxx@gmail.com)

    [n1,n2,n3] = size(X);
    Xt = lineartransform(X, transform);
    Y = zeros(n1,n2,n3);
    for i=1:n3
        Y(:,:,i) = sqrtm(Xt(:,:,i));
    end
    % Apply the inverse transform to obtain the final result
    Y = inverselineartransform(Y, transform);
end