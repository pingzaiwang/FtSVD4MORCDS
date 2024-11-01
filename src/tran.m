function At = tran(A,transform)
 

[n1,n2,n3] = size(A);
if nargin < 2
    % fft is the default transform
    transform  = 'fft';
end
At = zeros(n2,n1,n3);
if isequal(transform, 'fft')
    % fft transform
    At(:,:,1) = A(:,:,1)';
    for i = 2 : n3
        At(:,:,i) = A(:,:,n3-i+2)';
    end       
elseif isa(transform,'function_handle')
    A = lineartransform(A,transform);
    for i = 1 : n3
        At(:,:,i) = A(:,:,i)';
    end
    At = inverselineartransform(At,transform);
elseif ismatrix(transform)
    if isreal(transform)
        % L is a real matrix
        for i = 1 : n3
            At(:,:,i) = A(:,:,i)';
        end
    else
        % L is a complex matrix
        A = lineartransform(A,transform);
        for i = 1 : n3
            At(:,:,i) = A(:,:,i)';
        end
        At = inverselineartransform(At,transform);
    end
end

