function A = inverselineartransform(A,L)

% inverse linear transform along 3rd dim

if strcmp(L,'fwt')
    A = nmodetransform(A,'ifwt',3);
elseif strcmp(L,'fft')
    A = ifft(A,[],3);
elseif ismatrix(L)
    l = norm(L(:,1))^2;
    A = tmprod(A,L'/l,3);
end