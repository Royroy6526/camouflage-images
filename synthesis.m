
function [ output_args ] = synthesis(x,y, imA , imAp , imB ,Texture)
%imA = im2double(imread('field-mask.jpg'));
%imAp = im2double(imread('field.jpg'));
%imB = im2double(imread('field-newmask.jpg'));
%imA = rgb2gray(imread('luminance_back.png'));
%imAp = im2double(imread('color_back.png'));
%imB = rgb2gray(imread('opti_luminance.png'));

%imA = rgb2gray(im2double(imread('test-mask.png')));
%imAp = rgb2gray(im2double(imread('test.png')));
%imB = rgb2gray(im2double(imread('tiger-mask.png')));



%norm matrix
imA(imA==239)=255;
imA = im2double(imA);
imB = im2double(imB);
imAp = im2double(imAp);

%range = max(max(imA)) - min(min(imA));
%imA = (imA - min(min(imA))) / range;
%range = max(max(imB)) - min(min(imB));
%imB = (imB - min(min(imB))) / range;

options.scaleFeatures = true;

% shrink the images for speed of debugging
[sizex , sizey] = size(imB);

imA = imcrop(imA ,[x-40 y-40 sizey+80 sizex+80]);
imAp = imcrop(imAp ,[x-40 y-40 sizey+80 sizex+80]);
%imA = imresize(imA, .5);
%imB = imresize(imB, .5);
%imAp = imresize(imAp, .5);

imBp = ia_process(cat(3, imA, imA, imA), imAp, cat(3, imB, imB, imB),options);

Texture_gray = rgb2gray(uint8((Texture)));
Texture = im2double(Texture)/255;
for i = 1:sizex
    for j = 1: sizey
        if(Texture_gray(i,j) <50)
           imBp(i,j,:) =  Texture(i,j,:);
        end
    end
end


imwrite(imBp , 'B.png');
imwrite(imA , 'A-mask.png');
imwrite(imB , 'B-mask.png');
imwrite(imAp , 'A.png');
figure(1); clf;
subplot(221); imshow(imA); title('A')
subplot(222); imshow(imAp); title('A''')
subplot(223); imshow(imB); title('B')
subplot(224); imshow(imBp); title('B''');