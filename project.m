close all;
I = rgb2gray(imread('lion3.bmp'));
I = imresize(I, 0.5);
mask = rgb2gray(imread('lion3(M).bmp'));
mask = imresize(mask, 0.5);

top = -1; bot = 1000; left = 1000; right = -1;
[x,y] = size(I);
for i = 1:x
    for j = 1:y
        if mask(i,j) == 0
            I(i,j) = 255;
        else
            if top < i
                top = i;
            end
            if bot > i
                bot = i;
            end 
            if right < j
                right = j;
            end
            if left > j
                left = j;
            end
        end
    end
end

thresh_f = multithresh(I,7);
valuesMin_f = [min(I(:)) thresh_f];

quant8_f = imquantize(I,thresh_f,valuesMin_f);
quant8_floodfill_f = imcomplement(imfill(imcomplement(quant8_f),'holes'));

figure;
I_crop = imcrop(quant8_floodfill_f, [left bot (right-left) (top-bot)]);
%I_crop = imresize(I_crop, 0.5);
imshow(I_crop);

I_back = rgb2gray(imread('scn_1.bmp'));
mask_back = rgb2gray(imread('scn_1(M).bmp'));


top = -1; bot = 1000; left = 1000; right = -1;
bc_x = 0; bc_y =0; cnt = 0;
[x,y] = size(I_back);
for i = 1:x
    for j = 1:y
        if mask_back(i,j) ~= 255
            I_back(i,j) = 255;
        else
            if top < i
                top = i;
            end
            if bot > i
                bot = i;
            end 
            if right < j
                right = j;
            end
            if left > j
                left = j;
            end
        end
    end
end

thresh = multithresh(I_back,7);
valuesMin = [min(I_back(:)) thresh];

quant8 = imquantize(I_back,thresh,valuesMin);
quant8_floodfill = imcomplement(imfill(imcomplement(quant8),'holes'));

back_crop = imcrop(quant8_floodfill, [left bot (right-left) (top-bot)]);

[x,y] = size(back_crop);
for i = 1:x
    for j = 1:y
        if back_crop(i,j) ~= 255
            bc_x = bc_x + j;
            bc_y = bc_y + i;
            cnt = cnt + 1;
        end
    end
end

bc_x = round(bc_x / cnt);
bc_y = round(bc_y / cnt);

%figure;
%imshow(back_crop);
%rectangle('Position', [bc_x bc_y 1 1], 'LineWidth',2, 'EdgeColor','b');

[x,y] = size(I_crop);
for i = 1:x
    for j = 1:y
        if I_crop(i,j) < 150
            back_crop((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = I_crop(i,j);
        end
    end
end

%figure;
%imshow(back_crop);

global regionMap;
regionMap = zeros(size(I_crop));
label = 1;
step = 8;
[x,y] = size(I_crop);
i = step; j = step;

for i = step:x
    for j = step:y
		if(I_crop(i,j) < 223 && regionMap(i,j) == 0)
			regiongrowing(I_crop,i,j,label);
			label = label + 1;
        end
        j = j + step;
    end
    i = i + step;
    j = step;
end

M = max(regionMap);
maxValue = max(M);
regionMap = round(regionMap / maxValue * 255);
regionMap = uint8(regionMap);

centroids = [];
for i = 1:255
    k = find(regionMap==i);
    [x,y] = size(k);
    if(x < 100)
        regionMap(regionMap == i) = 255;
    end
end
   
regionMap = imcomplement(imfill(imcomplement(regionMap),'holes'));
figure;
imshow(regionMap);

for i = 1:255
    k = find(regionMap==i);
    [x,y] = size(k);
    if(x ~= 0)
        binaryMap = regionMap;
        binaryMap(binaryMap ~= i) = 0;
        binaryMap(binaryMap == i) = 255;
        binaryMap = im2bw(binaryMap,graythresh(binaryMap));
        s = regionprops(binaryMap,'centroid');
        centroids = [centroids ; cat(1, s.Centroid)];
    end
end

imshow(regionMap)
hold on
plot(centroids(:,1),centroids(:,2), 'b*')
hold off

%{
subplot(2,3,1);imshow(F);title('original');
subplot(2,3,2);imshow(quant8_f);title('quant8');
subplot(2,3,3);imshow(quant8_floodfill_f);title('floodfill');

subplot(2,3,4);imshow(I);title('original');
subplot(2,3,5);imshow(quant8);title('quant8');
subplot(2,3,6);imshow(quant8_floodfill);title('floodfill');
%}
