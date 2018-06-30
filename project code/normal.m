function [zx,zy]=normal(Ir,Il,l)
double(Ir),double(Il),double(l);
s=[Ir-Il];
p=l(1,1);
q=l(3,1);
w=l(2,1);
nx=s/p;
h(1,1)=l(2,1)^2+l(3,1)^2;
h(1,2)=-s*q;
h(1,3)=0.25*s^2+w^2*[nx^2-1];
d=roots(h);

nz=d(1,1);
if imag(nz)~=0
    zy=0;
    zx=0;
 
    return
end
ny=sqrt(1-nx^2-nz^2);

zx=nx/nz;
zy=ny/nz;
end