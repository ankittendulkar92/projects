close all
clear all

a=imread('face3.jpg');
a=rgb2gray(a);


figure
imshow(a)
a=double(a);
[~, threshold] = edge(a, 'sobel');
fudgeFactor = .5;
BWs = edge(a,'sobel', threshold * fudgeFactor);
%figure, imshow(BWs), title('binary gradient mask');
se90 = strel('line', 3, 90);
se0 = strel('line', 3, 0);
BWsdil = imdilate(BWs, [se90 se0]);
BWdfill = imfill(BWsdil, 'holes');
%figure, imshow(BWdfill);
BWnobord = imclearborder(BWdfill, 4);
seD = strel('diamond',1);
BWfinal = imerode(BWnobord,seD);
BWfinal = imerode(BWfinal,seD);
figure
imshow(BWfinal)
[m,n]=size(a);
[BW,thresh,gv,gh] = edge(a,'sobel');
 edge = atan2(gv, gh);
 a=double(a);
  ma=max(max(a));
nor(1,1)=cos(edge(609,1209));nor(1,2)=sin(edge(609,1209));inte(1,1)=a(609,1209)/ma;
nor(2,1)=cos(edge(610,1209));nor(2,2)=sin(edge(610,1209));inte(2,1)=a(610,1209)/ma;
nor(3,1)=cos(edge(611,1209));nor(3,2)=sin(edge(611,1209));inte(3,1)=a(611,1209)/ma;
nor(4,1)=cos(edge(612,1209));nor(4,2)=sin(edge(612,1209));inte(4,1)=a(612,1209)/ma;
li=mldivide(nor,inte);
li(3,1)=-sqrt(1-li(1,1)^2-li(2,1)^2);
% 
% 
for i=1:m
    for j=1:629
Ir=a(i,j);
Il=a(i,1260-j);
[zx,zy]=normal(Ir,Il,li)
zx1(i,j)=zx;
zy1(i,j)=zy;
zx1(i,1260-j)=-zx;
zy1(i,1260-j)=zy;
   end
end
fx=zx1;
fy=zy1;
z = frankotchellappa(fx,fy);

% plot the z value
[m,n]=size(z); 
[ X, Y ] = meshgrid( 1:n, 1:m );
%mesh(X,Y,z);
figure;
surf( X, Y, z, 'EdgeColor', 'none' );
camlight left;
lighting phong
title('surface plot');
