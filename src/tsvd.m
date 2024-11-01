function [U,S,V] = tsvd(A,transform,r)

% [U,S,V] = tsvd(A,transform,r) computes the tensor SVD with a specified tubal rank under linear transform, 
% i.e., A=U*S*V^*, where S is an r*r*n3 f-diagonal tensor, U and V are orthogonal tensors of size n1*r*n3 and n2*r*n3.
%
% Input:
%       A       -   n1*n2*n3 tensor
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
%       V - n2*r*n3 tensor
%
% See also lineartransform, inverselineartransform
%
% version 1.0 - 05/16/2023
%
% Written by xxx

[n1,n2,n3] = size(A);

if nargin < 2
    % fft is the default transform
    transform.L = @fft; transform.inverseL = @ifft;
end

A = lineartransform(A,transform);

U = zeros(n1,r,n3);
S = zeros(r,r,n3);
V = zeros(n2,r,n3);

for i = 1 : n3
    [U(:,:,i),S(:,:,i),V(:,:,i)] = svds(A(:,:,i),r);
end

U = inverselineartransform(U,transform);
S = inverselineartransform(S,transform);
V = inverselineartransform(V,transform);

end