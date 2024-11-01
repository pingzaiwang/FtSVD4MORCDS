function X = tprod_mul(L, varargin)
% TPROD_MUL Compute the t-product of multiple tensors
%   X = TPROD_MUL(L, A, B, C, ...) computes the t-product of tensors A, B, C, ...
%   using the transform matrix L.
%
%   Inputs:
%   - L: Transform matrix
%   - varargin: Variable number of input tensors (A, B, C, ...)
%
%   Output:
%   - X: Result of the t-product of input tensors

% Check if at least two input tensors are provided
if length(varargin) < 2
    error('At least two input tensors are required.');
end

% Initialize the result with the first two tensors
X = tprod(varargin{1}, varargin{2}, L);

% Iterate over the remaining tensors and compute the t-product
for i = 3:length(varargin)
    X = tprod(X, varargin{i}, L);
end
end