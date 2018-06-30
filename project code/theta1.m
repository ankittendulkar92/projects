close all
clear all
% load image
a=imread('face.jpg');
a=rgb2gray(a);


a=double(a);
ma=max(max(a));
g=a;
g(g<50)=0;


 [m,n]=size(a);
 g=zeros(m,n);
 for i=1:m-1
    for j=1:n-1
     
        if g(i,j+1)==0
        if g(i,j)>50
         g(i,j)=255;
          end
     end
    end
 end
 imshow(g);
[BW,thresh,gv,gh] = edge(a,'sobel');
 edge = atan2(gv, gh);
 
nor(1,1)=cos(edge(644,545));nor(1,2)=sin(edge(644,545));inte(1,1)=a(644,545)/ma;
nor(2,1)=cos(edge(645,545));nor(2,2)=sin(edge(645,545));inte(2,1)=a(645,545)/ma;
nor(3,1)=cos(edge(646,545));nor(3,2)=sin(edge(646,545));inte(3,1)=a(646,545)/ma;
nor(4,1)=cos(edge(647,544));nor(4,2)=sin(edge(647,544));inte(4,1)=a(647,544)/ma;
li=mldivide(nor,inte);
li(3,1)=-sqrt(1-li(1,1)^2-li(2,1)^2);

a=double(a);
for i=1:m
    for j=1:n/2
Ir=a(i,j);
Il=a(i,623-j);
[zx,zy]=normal(Ir,Il,li)
zx1(i,j)=zx;
zy1(i,j)=zy;
zx1(i,623-j)=-zx;
zy1(i,623-j)=zy;
   end
end
fx=zx1;
fy=zy1;
z = frankotchellappa(fx,fy);

%plot the z value
[ X, Y ] = meshgrid( 1:n, 1:m );
figure;
surf( X, Y, z, 'EdgeColor', 'none' );
camlight left;
lighting phong
title('surface plot');
