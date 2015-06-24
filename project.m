close all;

quant_num = 7;
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

thresh = multithresh(I,quant_num);
valuesMin = [min(I(:)) thresh];

I = imquantize(I,thresh,valuesMin);
I = imcomplement(imfill(imcomplement(I),'holes'));

%{
for i = 1:x
    for j = 1:y
        if mask(i,j) == 0
            I(i,j) = 255;
        end
    end
end
%}
I(I == valuesMin(quant_num + 1)) = 255;
I_crop = imcrop(I, [left bot (right-left) (top-bot)]);
luminance = I_crop;
%==========================================================================
I_back = rgb2gray(imread('scn_1.bmp'));
I_back_color = imread('scn_1.bmp');
mask_back = rgb2gray(imread('scn_1(M).bmp'));

top = -1; bot = 1000; left = 1000; right = -1;
bc_x = 0; bc_y =0; cnt = 0;
[x,y] = size(I_back);
for i = 1:x
    for j = 1:y
        if mask_back(i,j) ~= 255
            I_back(i,j) = 255;
            I_back_color(i,j,:) = 255;
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

thresh = multithresh(I_back,quant_num);
valuesMin = [min(I_back(:)) thresh];

I_back = imquantize(I_back,thresh,valuesMin);
I_back = imcomplement(imfill(imcomplement(I_back),'holes'));

%{
for i = 1:x
    for j = 1:y
        if mask_back(i,j) == 0
            I_back(i,j) = 255;
        end
    end
end
%}
I_back(I_back == valuesMin(quant_num + 1)) = 255;
back_crop = imcrop(I_back, [left bot (right-left) (top-bot)]);
I_back_color = imcrop(I_back_color, [left bot (right-left) (top-bot)]);
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
Texture = zeros(x,y,3);
Texture(Texture == 0) = 255;
for i = 1:x
    for j = 1:y
        if I_crop(i,j) < 255
            Texture(i,j,:) = I_back_color((bc_x-round(y/2))+i,(bc_y-round(x/2))+j,:);
            back_crop((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = 255;  %Substitute with 255
        end
    end
end

luminance_back = back_crop;
figure;
imshow(uint8(Texture));
%==========================================================================

%==========================================================================
%PROCESS FOR BACKGROUND
global regionMap;
regionMap = zeros(size(back_crop));
label = 1;
step = 8;
[x,y] = size(back_crop);
i = step; j = step;

%對整張影像灑種子
for i = step:x
    for j = step:y
		if(back_crop(i,j) < 255 && regionMap(i,j) == 0)
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
    if(index_x > 0 && index_x < 80)
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
regionMap = uint16(regionMap);  %double 轉回 uint8
%==========================================================================
[re_row,re_col] = size(regionMap);   %size of regionMap
neighbor = [-1 0; 1 0; 0 -1; 0 1];
back_seg_area = [];
back_image_area = 0;

for i = 1:maxValue              %對每個Segment做二值化，在對二值化影像找中心點座標
    binaryMap = regionMap;
    binaryMap(binaryMap ~= i) = 0;
    binaryMap(binaryMap == i) = 255;
    binaryMap = im2bw(binaryMap,graythresh(binaryMap));

    s = regionprops(binaryMap,'centroid','area');
    back_centroids = [back_centroids ; cat(1, s.Centroid)];
	back_seg_area = [back_seg_area;s.Area];
	back_image_area = back_image_area + s.Area;
end

back_regionMap = regionMap;
back_seg_num = maxValue;

back_seg_lumminance = [];
for i = 1:maxValue                     %Assign original luminance value to segmented background image
    [row,col] = find(back_regionMap == i);          %"luminance_back" means original luminance background image
    back_seg_lumminance = [back_seg_lumminance;back_crop(row(1),col(1))];
end



for i = 1:maxValue                             
    luminance_back(back_regionMap == i) = back_seg_lumminance(i);
end

%figure;
%imshow(luminance_back);
back_maxValue = maxValue;
luminance_back_org = luminance_back;
%==========================================================================
%PROCESS FOR FORGROUND
regionMap = zeros(size(I_crop));
label = 1;
step = 8;
[x,y] = size(I_crop);
i = step; j = step;

%對整張影像灑種子
for i = step:x
    for j = step:y
		if(I_crop(i,j) < 255 && regionMap(i,j) == 0)
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
    if(index_x > 0 && index_x < 70)
        if(col(1)-1 < 1 || I_crop(row(1),col(1)-1) == 255)
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
regionMap = uint16(regionMap);  %double 轉回 uint8
%==========================================================================
[re_row,re_col] = size(regionMap);   %size of regionMap
[back_x back_y] = size(back_regionMap);
neighbor_label = [];

%input for optimization
fore_seg_length = [];
fore_seg_area = [];
stdout_shared_length = [];
immers_shared_length = [];
fore_image_area = 0;

for i = 1:maxValue              %對每個Segment做二值化，在對二值化影像找中心點座標
    binaryMap = regionMap;
    binaryMap(binaryMap ~= i) = 0;
    binaryMap(binaryMap == i) = 255;
    binaryMap = im2bw(binaryMap,graythresh(binaryMap));

    s = regionprops(binaryMap,'centroid','area');
    centroids = [centroids ; cat(1, s.Centroid)];
	fore_seg_area = [fore_seg_area;s.Area];
	fore_image_area = fore_image_area + s.Area;

    [row,col] = find(binaryMap == 1);
    boundary = bwboundaries(binaryMap);
    
    label = zeros(maxValue,1,'uint8');
    std_shared_length = zeros(maxValue,1,'uint16');
	imm_shared_length = zeros(back_maxValue,1,'uint16');

    [seg_x,seg_y] = size(boundary);    %seg_x = the number of contour (some contour might have holes in it)
    for seg_index = 1:seg_x
        bound_point = boundary{seg_index};        %bound_point is the coordinate of all contour points
        [bound_x,bound_y] = size(bound_point);    %bound_x = the number of contour points in the BOUNDARY
        if (seg_index == 1)
			fore_seg_length = [fore_seg_length;bound_x];
		end
        for j = 1:bound_x
            for k = 1:4                           %check neighbor piexl in the region boundary
                m = bound_point(j,1) + neighbor(k,1);   
                n = bound_point(j,2) + neighbor(k,2);
				s = m + (bc_x-round(re_col/2));   
				t = n + (bc_y-round(re_row/2));
                if( m == 0 || n == 0 || m > re_row || n > re_col) 		%outside foreground image
					if ( s == 0 || t == 0 || s > back_x || t > back_y)	%outside background image
						continue;
					else												
						if (back_regionMap(s,t) ~= 0)	%if inside the background contour,add 1 to the shared length
							imm_shared_length(back_regionMap(s,t)) = imm_shared_length(back_regionMap(s,t)) + 1;
							%temp1 = [temp1;s t];
							break;
						end
					end
					continue;
                end
				if (regionMap(m,n) == 0)
					if (back_regionMap(s,t) ~= 0)
						imm_shared_length(back_regionMap(s,t)) = imm_shared_length(back_regionMap(s,t)) + 1;
						%temp1 = [temp1;s t];
						break;
					end
				end
                if((regionMap(m,n) ~= i) && (regionMap(m,n) ~= 0))         %record neighbor color
                    label(regionMap(m,n)) = 1;
					if(seg_index > 1)								%holes shared length = the number of contour points
						std_shared_length(regionMap(m,n)) = bound_x;
					else
						std_shared_length(regionMap(m,n)) = std_shared_length(regionMap(m,n)) + 1;
					end
                    break;
                end
            end
            if(seg_index > 1)
                break;
            end
        end
    end
    if(i==1)
        neighbor_label = label;
		stdout_shared_length = std_shared_length;
		immers_shared_length = imm_shared_length;
    else
        neighbor_label = horzcat(neighbor_label,label);     %concate the array of neighbor color of each segment
		stdout_shared_length = horzcat(stdout_shared_length,std_shared_length);
		immers_shared_length = horzcat(immers_shared_length,imm_shared_length);
    end
end

fore_regionMap = uint8(regionMap);
%imshow(fore_regionMap);
fore_seg_num = maxValue;

fore_seg_lumminance = [];
for i = 1:maxValue                     %Assign original luminance value to segmented foreground image
    [row,col] = find(fore_regionMap == i);          %"luminance" means original luminance foreground image
    fore_seg_lumminance = [fore_seg_lumminance;I_crop(row(1),col(1))];
end

for i = 1:maxValue                             
    luminance(fore_regionMap == i) = fore_seg_lumminance(i);
end

%figure;
%imshow(luminance);
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


[x,y] = size(I_crop);           %Combine Foreground and Background
for i = 1:x
    for j = 1:y
        if I_crop(i,j) < 255
            luminance_back((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = luminance(i,j);  %Substitute with 255
        end
    end
end

figure;
imshow(luminance_back);
hold on
plot(centroids(:,1),centroids(:,2), 'b*')     %Draw Segments Centroids 
plot(back_centroids(:,1),back_centroids(:,2), 'ro')     %Draw Segments Centroids 
dim = size(centroids);


for i = 1:dim(1)
    length = graph.get(i-1).size();
    for j = 1:length-knn          %foreground
        plot([centroids(i,1),centroids(graph.get(i-1).get(j-1),1)], [centroids(i,2),centroids(graph.get(i-1).get(j-1),2)], 'b');
    end
    for k = 1:knn                 %background
        plot([centroids(i,1),back_centroids(graph.get(i-1).get(j+k-1),1)], [centroids(i,2),back_centroids(graph.get(i-1).get(j+k-1),2)], 'y');
    end
end

hold off

fore_regionMap(fore_regionMap == 28) = 255;
fore_regionMap(fore_regionMap == 34) = 255;
figure;
imshow(uint8(fore_regionMap));
%}
%==========================================================================
%===================Immersion energy(Data cost)============================
datacost = zeros(back_seg_num , fore_seg_num);
immersion_weight = zeros(fore_seg_num , 3);

for i = 1:fore_seg_num
	for j = 1:back_seg_num
		cost = 0;
		for k = 1:knn
			length = graph.get(i-1).size();
			k_neighbor = graph.get(i-1).get(length-knn+k-1);	%k_neighbor is the kth neighbor of ith foreground segment
			if (immers_shared_length(k_neighbor,i) == 0)
				a = [centroids(i,1) centroids(i,2)];
				b = [back_centroids(k_neighbor,1) back_centroids(k_neighbor,2)];
				imm_weight = (back_seg_area(k_neighbor)/back_image_area) * exp(-norm(a-b)/0.1);
			else
				imm_weight = (double(immers_shared_length(k_neighbor,i))/double(fore_seg_length(i)))*(back_seg_area(k_neighbor)/back_image_area);
			end
			power_result = double(back_seg_lumminance(j) - back_seg_lumminance(k_neighbor));
			power_result = power_result.^2;
			cost = cost + power_result * imm_weight;
            immersion_weight(i , k) = imm_weight;
		end
		datacost(j,i) = cost;
	end
end


smooth_weight = zeros(fore_seg_num,fore_seg_num);

for i = 1:fore_seg_num
	for j = 1:fore_seg_num
		smooth_weight(i,j) = ((double(stdout_shared_length(i,j)))/double(fore_seg_length(i)) * (fore_seg_area(i)/fore_image_area) + (double(stdout_shared_length(j,i)))/double(fore_seg_length(j)) * (fore_seg_area(j)/fore_image_area))/2;
	end
end
%==========================================================================
%===================Standout energy(Smooth cost)===========================
smooth_cost = zeros(back_seg_num,back_seg_num);
%smooth_weight = zeros(fore_seg_num,fore_seg_num);

fore_luminance_diff = zeros(fore_seg_num,fore_seg_num);
%{
for i = 1:back_seg_num
	for j = 1:back_seg_num
		diff = double(back_seg_lumminance(i)) - double(back_seg_lumminance(j));
		smooth_cost(i,j) = diff;
	end
end
%}
for i = 1:fore_seg_num
	for j = 1:fore_seg_num
		if (smooth_weight(i,j) ~= 0)
				fore_luminance_diff(i,j) = double(fore_seg_lumminance(i))- double(fore_seg_lumminance(j));
		end
	end
end

%smooth_cost = smooth_cost.^2;
%==========================================================================
%======================Luminance Optimaization=============================
%Solve AX = B, linear least square
immersion_weight = normr(immersion_weight);
%smooth_weight = normr(smooth_weight);

immersion_weight = immersion_weight * 0.5;
smooth_weight = smooth_weight * 2;

A_row_size = knn*fore_seg_num + fore_seg_num.^2 - fore_seg_num;
A = zeros(A_row_size, fore_seg_num);
B = zeros(A_row_size, 1);
for i = 1:fore_seg_num
    A((i-1)*knn+1,i) = immersion_weight(i,1);
    A((i-1)*knn+2,i) = immersion_weight(i,2);
    A((i-1)*knn+3,i) = immersion_weight(i,3);
    
    length = graph.get(i-1).size();
    B((i-1)*knn+1,1) = immersion_weight(i,1) * back_seg_lumminance(graph.get(i-1).get(length-knn+0));
    B((i-1)*knn+2,1) = immersion_weight(i,2) * back_seg_lumminance(graph.get(i-1).get(length-knn+1));
    B((i-1)*knn+3,1) = immersion_weight(i,3) * back_seg_lumminance(graph.get(i-1).get(length-knn+2));
end

for i = (knn*fore_seg_num+1):A_row_size
    for j = 1:fore_seg_num
        for k = 1:fore_seg_num
            if(j ~= k)
                A(i,j) = smooth_weight(j,k);
                A(i,k) = -smooth_weight(j,k);
                
                B(i,1) = fore_luminance_diff(j,k) * smooth_weight(j,k);
            end
        end
    end
end

fore_seg_characteristic = zeros(fore_seg_num , 1);
for i = 1:fore_seg_num
   for j = 1: fore_seg_num
      if (j > i)
          fore_seg_characteristic(i) =  fore_seg_characteristic(i) + abs(fore_luminance_diff(i,j));
          fore_seg_characteristic(j) =  fore_seg_characteristic(j) + abs(fore_luminance_diff(i,j));
      end
   end
end

for i = 1:fore_seg_num
    fore_seg_characteristic(i) = double(fore_seg_characteristic(i))/fore_seg_area(i);
end
characteristic_index = [];
for i = 1:3
    [value,index] = max(fore_seg_characteristic);
    fore_seg_characteristic(index) = 0;
    characteristic_index = [characteristic_index;index];
end

X = lsqnonneg(A,B);
X = uint8(X);

for i = 1:3
   X(characteristic_index(i)) = fore_seg_lumminance(characteristic_index(i));
end

opti_luminance = zeros(size(luminance));
opti_luminance(fore_regionMap == 0) = 255;
for i = 1:fore_seg_num
    opti_luminance(fore_regionMap == i) = X(i);
end
opti_luminance = uint8(opti_luminance);
%figure;
%imwrite(opti_luminance,'opti_luminance.png');



[x,y] = size(opti_luminance);           %Combine Foreground and Background
for i = 1:x
    for j = 1:y
        if opti_luminance(i,j) < 255
            luminance_back((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = opti_luminance(i,j);  %Substitute with I_crop(i,j)
       
        else
            if(luminance_back((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) == 255 && (bc_y-round(x/2))+j < bc_y)
                luminance_back((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = luminance_back((bc_x-round(y/2))+i,(bc_y-round(x/2))+j-10);
            end
           
        end
    end
end

figure;
imshow(luminance_back);
%imwrite(luminance_back,'opti_result.png');

synthesis((bc_y-round(x/2)),(bc_x-round(y/2)) , luminance_back_org, I_back_color,opti_luminance ,Texture);











