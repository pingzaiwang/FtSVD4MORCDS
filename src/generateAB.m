function [U, V] = generateAB(m1,n1,m2,n2,r)


[Q,~,~]=svd(randn(r,r),'econ');

A1=randn(m1,r)*Q';
B1=randn(m1,r)*Q';

A2=randn(m2,r)*Q';
B2=randn(m2,r)*Q';

U=[A1;A2];
V=[B1;B2];
end