function dis= ctour(g,Ir,Il,l,xr,yr,xl,yl,m)
t=0.22;
v=0.44;
[zx,zy]=normal(Ir,Il,l);
if imag(zx)~=0
    dis=1000;
    return
end

cx=[-cos(t)+zx*sin(t)]/[cos(t)+zx*sin(t)];

ye=yl-cx;
if ye>m|ye<1
    ye=1;
end
Ir=g(xr,yr-1);
xe=xl;
Il=value(g,xl,ye);
[zx,zy]=normal(Ir,Il,l);
if imag(zx)~=0
    dis=1000;
    return
end

cy=zy*sin(v)/[cos(t)-zx*sin(t)];
ye=ye+cy;
if ye>m|ye<1
    ye=1;
end
Ir=g(xr+1,yr-1);
xe=xe+1;
Il=value(g,xe,ye);
[zx,zy]=normal(Ir,Il,l);
if imag(zx)~=0
    dis=1000;
    return
end

cx=[-cos(t)+zx*sin(t)]/[cos(t)+zx*sin(t)];

ye=ye+cx;
if ye>m|ye<1
    ye=1;
end
Ir=g(xr+1,yr);
Il=value(g,xe,ye);
[zx,zy]=normal(Ir,Il,l);
if imag(zx)~=0
    dis=1000;
    return
end

cy=zy*sin(v)/[cos(t)-zx*sin(t)];
ye=ye-cy;
if ye>m|ye<1;
    ye=1;
end
dis1=abs(ye);
dis2=ctour2(g,Ir,Il,l,xr,yr,xl,yl,m);
dis3=ctour3(g,Ir,Il,l,xr,yr,xl,yl,m);
dis=dis1+dis2+dis3;
end
