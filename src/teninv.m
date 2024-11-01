function A = teninv(A,transform)


[n1,n2,n3] = size(A);
if n1 ~= n2
    error('Error using tinv. Tensor must be square.');
end
if nargin < 2
    % fft is the default transform
    transform = 'fft'; 
end

I = eye(n1);
if isequal(transform, 'fft')
    % efficient computing for fft transform
    A = fft(A,[],3);
    halfn3 = ceil((n3+1)/2);
    for i = 1 : halfn3
        A(:,:,i) = A(:,:,i)\I;
    end
    for i = halfn3+1 : n3
        A(:,:,i) = conj(A(:,:,n3+2-i));
    end
    A = ifft(A,[],3);
else
    % other transform
    A = lineartransform(A,transform);
    for i = 1 : n3
        A(:,:,i) = pinv(A(:,:,i));%\I;
    end
    A = inverselineartransform(A,transform);
end
