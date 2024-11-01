function [U,S] = teig(A,transform)

% [U,S,V] = eig(A,transform) computes the tensor eigenvalue decomposition under linear transform, 
% i.e., A=U*S*U^*, where S is an r*r*n3 f-diagonal tensor, U is orthogonal tensors of size n*n*n3.
%
% Input:
%       A       -   n*n*n3 tensor
%       transform - a structure which defines the linear transform
%                   transform.L: the linear transform of two types:
%                             - type I: function handle, i.e., @fft, @dct
%                             - type II: invertible matrix of size n3*n3
%                   transform.inverseL: the inverse linear transform of transform.L
%       r       -   the specified tubal rank
%
% Output: 
%       U - n1*r*n3 tensor
%       S - r*r*n3 tensor 
%       U - n2*r*n3 tensor
%
% See also lineartransform, inverselineartransform
%
% version 1.0 - 05/16/2023
%
% Written by xxx (xxx@gmail.com)

[n1,~, n3] = size(A);

if nargin < 2
    % fft is the default transform
    transform = 'fft';  
end

A = lineartransform(A,transform);

U = zeros(n1,n1,n3);
S = zeros(n1,n1,3);

for i = 1 : n3
    [U(:,:,i),S(:,:,i)] = eig(A(:,:,i));
end

U = inverselineartransform(U,transform);
S = inverselineartransform(S,transform);

end