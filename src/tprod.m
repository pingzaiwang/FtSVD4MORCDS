function C = tprod(A,B,L)

% Tensor-tensor product of two 3 way tensors: C = A*B
% A - n1*n2*n3 tensor
% B - n2*l*n3  tensor
% C - n1*l*n3  tensor
 

[n1,n2,n3] = size(A);
[m1,m2,m3] = size(B);

if n2 ~= m1 || n3 ~= m3
    error('Inner tensor dimensions must agree.');
end

A = lineartransform(A,L);
B = lineartransform(B,L);
C = zeros(n1,m2,n3);

% if strcmp(L,'fft')
%     % first frontal slice
%     C(:,:,1) = A(:,:,1)*B(:,:,1);
%     % i=2,...,halfn3
%     halfn3 = round(n3/2);
%     for i = 2 : halfn3
%         C(:,:,i) = A(:,:,i)*B(:,:,i);
%         C(:,:,n3+2-i) = conj(C(:,:,i));
%     end    
%     % if n3 is even
%     if mod(n3,2) == 0
%         i = halfn3+1;
%         C(:,:,i) = A(:,:,i)*B(:,:,i);
%     end
% else    
    for i = 1 : n3
        C(:,:,i) = A(:,:,i)*B(:,:,i);
    end
% end

C = inverselineartransform(C,L);
