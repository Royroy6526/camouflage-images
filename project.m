close all;
I = rgb2gray(imread('lion3.bmp'));
I = imresize(I, 0.5);
mask = rgb2gray(imread('lion3(M).bmp'));
mask = imresize(mask, 0.5);
%==========================================================================
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

thresh = multithresh(I,7);
valuesMin = [min(I(:)) thresh];

I = imquantize(I,thresh,valuesMin);
I = imcomplement(imfill(imcomplement(I),'holes'));


I_crop = imcrop(I, [left bot (right-left) (top-bot)]);
luminance = I_crop;
%figure;
%imshow(I_crop);
%==========================================================================
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

I_back = imquantize(I_back,thresh,valuesMin);
I_back = imcomplement(imfill(imcomplement(I_back),'holes'));

back_crop = imcrop(I_back, [left bot (right-left) (top-bot)]);

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
%==========================================================================
[x,y] = size(I_crop);           %Combine Foreground and Background
for i = 1:x
    for j = 1:y
        if I_crop(i,j) < 220
            back_crop((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = 255;  %Substitute with I_crop(i,j)
        end
    end
end

luminance_back = back_crop;
%figure;
%imshow(back_crop);
%==========================================================================
%PROCESS FOR FORGROUND
global regionMap;
regionMap = zeros(size(I_crop));
label = 1;
step = 8;
[x,y] = size(I_crop);
i = step; j = step;

%對整張影像灑種子
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
%==========================================================================
centroids = [];             %對RegionGrowing完的結果做處理
                            %100 pixel以下的Segment用隔壁的pixel Color取代
for i = 1:maxValue     
    [row,col] = find(regionMap == i);
    [index_x,index_y] = size(row);
    if(index_x > 0 && index_x < 100)
        if(col(1)-1 < 1)
            intensity = regionMap(row(index_x),col(index_x)+1);
        else
            intensity = regionMap(row(1),col(1)-1);
        end
        regionMap(regionMap == i) = intensity;
    end
end
%==========================================================================
reLabel_cnt = 1;
for i = 1:maxValue              %Relabeling
    k = find(regionMap == i);
    [x,y] = size(k);
    if(x ~= 0)
        regionMap(regionMap == i) = reLabel_cnt;
        reLabel_cnt = reLabel_cnt + 1;
    end
end
maxValue = reLabel_cnt - 1;     %New label of regionMap is from 1 to maxValue

%{
M = max(regionMap);
maxValue = max(M);
regionMap = round(regionMap / maxValue * 255);
%}
regionMap = uint8(regionMap);  %double 轉回 uint8
%==========================================================================
[re_row,re_col] = size(regionMap);   %size of regionMap
neighbor = [-1 0; 1 0; 0 -1; 0 1];
neighbor_label = [];

for i = 1:maxValue              %對每個Segment做二值化，在對二值化影像找中心點座標
    binaryMap = regionMap;
    binaryMap(binaryMap ~= i) = 0;
    binaryMap(binaryMap == i) = 255;
    binaryMap = im2bw(binaryMap,graythresh(binaryMap));
    %index = [i,0];
    s = regionprops(binaryMap,'centroid');
    centroids = [centroids ; cat(1, s.Centroid)];

    [row,col] = find(binaryMap == 1);
    boundary = bwboundaries(binaryMap);
    label = zeros(maxValue,1,'uint8');

    [seg_x,seg_y] = size(boundary);    %seg_x = the number of contour (some contour might have holes in it)
    for seg_index = 1:seg_x
        bound_point = boundary{seg_index};        %bound_point is the coordinate of all contour points
        [bound_x,bound_y] = size(bound_point);    %bound_x = the number of contour points in the BOUNDARY
        for j = 1:bound_x
            for k = 1:4                           %check neighbor piexl in the region boundary
                m = bound_point(j,1) + neighbor(k,1);
                n = bound_point(j,2) + neighbor(k,2);
                if(m == 0 || n == 0 || m > re_row || n > re_col)
                    continue;
                end
                if(regionMap(m,n) ~= i && regionMap(m,n) ~= 0)         %record neighbor color
                    label(regionMap(m,n)) = 1;
                    break;
                end
            end
            if(seg_index > 1)           %Holes in one segment only consider the first iteration (Assume only one segment in the hole)
                break;
            end
        end
    end
    if(i==1)
        neighbor_label = label;
    else
        neighbor_label = horzcat(neighbor_label,label);     %concate the array of neighbor color of each segment
    end
end

fore_regionMap = regionMap;

for i = 0:maxValue                     %Assign original luminance value to segmented foreground image
    [row,col] = find(fore_regionMap == i);          %"luminance" means original luminance foreground image
    luminance(fore_regionMap == i) = I_crop(row(1),col(1));
end

%figure;
%imshow(luminance);
%==========================================================================

%==========================================================================
%PROCESS FOR BACKGROUND
regionMap = zeros(size(back_crop));
label = 1;
step = 8;
[x,y] = size(back_crop);
i = step; j = step;

%對整張影像灑種子
for i = step:x
    for j = step:y
		if(back_crop(i,j) < 223 && regionMap(i,j) == 0)
			regiongrowing(back_crop,i,j,label);
			label = label + 1;
        end
        j = j + step;
    end
    i = i + step;
    j = step;
end

M = max(regionMap);
maxValue = max(M);
%==========================================================================
back_centroids = [];             %對RegionGrowing完的結果做處理
                            %100 pixel以下的Segment用隔壁的pixel Color取代
for i = 1:maxValue     
    [row,col] = find(regionMap == i);
    [index_x,index_y] = size(row);
    if(index_x > 0 && index_x < 100)
        if(col(1)-1 < 1)
            intensity = regionMap(row(index_x),col(index_x)+1);
        else
            intensity = regionMap(row(1),col(1)-1);
        end
        regionMap(regionMap == i) = intensity;
    end
end
%==========================================================================
reLabel_cnt = 1;
for i = 1:maxValue              %Relabeling
    k = find(regionMap == i);
    [x,y] = size(k);
    if(x ~= 0)
        regionMap(regionMap == i) = reLabel_cnt;
        reLabel_cnt = reLabel_cnt + 1;
    end
end
maxValue = reLabel_cnt - 1;     %New label of regionMap is from 1 to maxValue

%{
M = max(regionMap);
maxValue = max(M);
regionMap = round(regionMap / maxValue * 255);
%}
regionMap = uint8(regionMap);  %double 轉回 uint8
%==========================================================================
[re_row,re_col] = size(regionMap);   %size of regionMap
back_neighbor_label = [];

for i = 1:maxValue              %對每個Segment做二值化，在對二值化影像找中心點座標
    binaryMap = regionMap;
    binaryMap(binaryMap ~= i) = 0;
    binaryMap(binaryMap == i) = 255;
    binaryMap = im2bw(binaryMap,graythresh(binaryMap));
    %index = [i,0];
    s = regionprops(binaryMap,'centroid');
    back_centroids = [back_centroids ; cat(1, s.Centroid)];

    [row,col] = find(binaryMap == 1);
    boundary = bwboundaries(binaryMap);
    label = zeros(maxValue,1,'uint8');

    [seg_x,seg_y] = size(boundary);    %seg_x = the number of contour (some contour might have holes in it)
    for seg_index = 1:seg_x
        bound_point = boundary{seg_index};        %bound_point is the coordinate of all contour points
        [bound_x,bound_y] = size(bound_point);    %bound_x = the number of contour points in the BOUNDARY
        for j = 1:bound_x
            for k = 1:4                           %check neighbor piexl in the region boundary
                m = bound_point(j,1) + neighbor(k,1);
                n = bound_point(j,2) + neighbor(k,2);
                if(m == 0 || n == 0 || m > re_row || n > re_col)
                    continue;
                end
                if(regionMap(m,n) ~= i && regionMap(m,n) ~= 0)         %record neighbor color
                    label(regionMap(m,n)) = 1;
                    break;
                end
            end
            if(seg_index > 1)
                break;
            end
        end
    end
    if(i==1)
        back_neighbor_label = label;
    else
        back_neighbor_label = horzcat(back_neighbor_label,label);     %concate the array of neighbor color of each segment
    end
end

for i = 0:maxValue                     %Assign original luminance value to segmented background image
    [row,col] = find(regionMap == i);          %"luminance_back" means original luminance background image
    luminance_back(regionMap == i) = back_crop(row(1),col(1));
end

figure;
imshow(luminance_back);
%==========================================================================
%===============Graph Construction=====================
%=================Standout Edges=======================
graph = java.util.ArrayList;        %Graph is a ArrayList(2D LinkedList)
                                    %Index start from 1 (Calling Java ArrayList)
                                    %Ex. graph.get(0);
[x,y] = size(centroids);

for i = 1:x     %(x is the number of Foregroun Segment)
    index = find(neighbor_label(:,i)==1);
    [index_x,index_y] = size(index);
    if(index_x == 0)    %Discard isolated Segment
        continue;
    else
        node = java.util.ArrayList;
        for j = 1:index_x
            node.add(uint8(index(j)));
        end
    end
    graph.add(node);
end
%=================Immersion Edges=====================
%idx is the index of Three Nearest Neighbor of centroids
%dist is the distance
[x,y] = size(fore_regionMap);           %Combine Foreground and Background
for i = 1:x
    for j = 1:y
        if fore_regionMap(i,j) > 0
            regionMap((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = fore_regionMap(i,j);  %Substitute with I_crop(i,j)
        end
    end
end

centroids(:,1) = (bc_y-round(x/2)) + centroids(:,1);
centroids(:,2) = (bc_x-round(y/2)) + centroids(:,2);

knn = 3;
[idx,dist] = knnsearch(back_centroids,centroids,'k',knn,'distance','euclidean');     %KNN(K=3)
dim = size(idx);
for i = 1:dim(1)
    for j = 1:knn
        graph.get(i-1).add(uint8(idx(i,j)));
    end
end
%==========================================================================
%draw graph
%{
imshow(regionMap);
hold on;
for k=1:1
   b = boundary{k};
   plot(b(:,2),b(:,1),'g','LineWidth',1);
end

figure;
imshow(fore_regionMap);
hold on
plot(centroids(:,1),centroids(:,2), 'g*')     %Draw Segments Centroids 
hold off
%}

figure;
imshow(regionMap);
hold on
plot(centroids(:,1),centroids(:,2), 'g*')     %Draw Segments Centroids 
plot(back_centroids(:,1),back_centroids(:,2), 'ro')     %Draw Segments Centroids 
dim = size(centroids);


for i = 1:dim(1)
    length = graph.get(i-1).size();
    for j = 1:length-knn          %foreground
        plot([centroids(i,1),centroids(graph.get(i-1).get(j-1),1)], [centroids(i,2),centroids(graph.get(i-1).get(j-1),2)], 'g');
    end
    for k = 1:knn                 %background
        plot([centroids(i,1),back_centroids(graph.get(i-1).get(j+k-1),1)], [centroids(i,2),back_centroids(graph.get(i-1).get(j+k-1),2)], 'r');
    end
end

hold off
%==========================================================================




