function A = tensor_from_matrix(M, n3)
    % A = tensor_from_matrix(M, n3) generates an n1*n2*n3 tensor A from an n1*n2 matrix M,
    % where each frontal slice of A is identical to M.
    %
    % Input:
    %       M - n1*n2 matrix
    %       n3 - the desired number of frontal slices in the output tensor
    %
    % Output:
    %       A - n1*n2*n3 tensor
    %
    % version 1.0 - 05/16/2024
    %
    % Written by xxx

    A = repmat(M, [1, 1, n3]);
end