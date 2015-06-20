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

back_quant8 = valuesMin;

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
%{
[x,y] = size(I_crop);           %Combine Foreground and Background
for i = 1:x
    for j = 1:y
        if I_crop(i,j) < 220
            back_crop((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = 255;  %Substitute with I_crop(i,j)
        end
    end
end

figure;
imshow(back_crop);
%}
%==========================================================================
%figure;
%imshow(back_crop);
[x,y] = size(I_crop);           %Combine Foreground and Background to capture centroid of combined image
for i = 1:x
    for j = 1:y
        if I_crop(i,j) < 220
            back_crop((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = valuesMin(8);  %Substitute with I_crop(i,j)
        end
    end
end

luminance_back = back_crop;
%figure;
%imshow(back_crop);
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
%figure;
%imshow(regionMap);
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
%figure;
%imshow(regionMap);
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




%figure;
%imshow(backregionMap_for_centroid);







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
	
	%figure;
	%imshow(binaryMap);
    %index = [i,0];
    s = regionprops(binaryMap,'centroid','area');
    back_centroids = [back_centroids ; cat(1, s.Centroid)];
	back_seg_area = [back_seg_area;s.Area];
	back_image_area = back_image_area + s.Area;
end




back_regionMap = regionMap;
back_seg_num = maxValue;

back_seg_lumminance = [];
for i = 1:maxValue                     %Assign original luminance value to segmented foreground image
    [row,col] = find(back_regionMap == i);          %"luminance" means original luminance foreground image
    back_seg_lumminance = [back_seg_lumminance;back_crop(row(1),col(1))];
end



for i = 1:maxValue                             
    luminance_back(back_regionMap == i) = back_seg_lumminance(i);
end

figure;
imshow(luminance_back);

back_maxValue = maxValue;

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
for_image_area = 0;

%figure;
%imshow(back_regionMap);
for i = 1:maxValue              %對每個Segment做二值化，在對二值化影像找中心點座標
    binaryMap = regionMap;
    binaryMap(binaryMap ~= i) = 0;
    binaryMap(binaryMap == i) = 255;
	
	
    binaryMap = im2bw(binaryMap,graythresh(binaryMap));
    %index = [i,0];
    s = regionprops(binaryMap,'centroid','area');
    centroids = [centroids ; cat(1, s.Centroid)];
	fore_seg_area = [fore_seg_area;s.Area];
	for_image_area = for_image_area + s.Area;

    [row,col] = find(binaryMap == 1);
    boundary = bwboundaries(binaryMap);

	
    label = zeros(maxValue,1,'uint8');
	std_shared_length = zeros(maxValue,1,'uint16');
	imm_shared_length = zeros(back_maxValue,1,'uint16');

    [seg_x,seg_y] = size(boundary);    %seg_x = the number of contour (some contour might have holes in it)
    for seg_index = 1:seg_x
        bound_point = boundary{seg_index};        %bound_point is the coordinate of all contour points(same as matrix index)
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

fore_regionMap = regionMap;
for_seg_num = maxValue;

for_seg_lumminance = [];
for i = 1:maxValue                     %Assign original luminance value to segmented foreground image
    [row,col] = find(fore_regionMap == i);          %"luminance" means original luminance foreground image
    for_seg_lumminance = [for_seg_lumminance;I_crop(row(1),col(1))];
end

for i = 1:maxValue                             
    luminance(fore_regionMap == i) = for_seg_lumminance(i);
end

figure;
imshow(luminance);
%==========================================================================


%==========================================================================
%{
imshow(regionMap);
hold on;
for k=1:numel (boundary)
   b = boundary{k};
   plot(b(:,2),b(:,1),'g','LineWidth',1);
end
figure;
imshow(fore_regionMap);
hold on
plot(centroids(:,1),centroids(:,2), 'g*')     %Draw Segments Centroids 
hold off
%}

[x,y] = size(fore_regionMap);           %Combine Foreground and Background
for i = 1:x
    for j = 1:y
        if fore_regionMap(i,j) > 0
            back_regionMap((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = fore_regionMap(i,j);  %Substitute with I_crop(i,j)
        end
    end
end

centroids(:,1) = (bc_y-round(x/2)) + centroids(:,1);
centroids(:,2) = (bc_x-round(y/2)) + centroids(:,2);

[x,y] = size(fore_regionMap);

figure;
imshow(back_regionMap);
hold on

plot(centroids(:,1),centroids(:,2), 'g*')     %Draw Segments Centroids 
plot(back_centroids(:,1),back_centroids(:,2), 'r*')     %Draw Segments Centroids 
%plot(temp1(:,2),temp1(:,1), 'ro')
%}
hold off
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
knn = 3;
[idx,dist] = knnsearch(back_centroids,centroids,'k',knn,'distance','euclidean');     %KNN(K=3)
dim = size(idx);
for i = 1:dim(1)
    for j = 1:knn
        graph.get(i-1).add(uint8(idx(i,j)));
    end
end



%===================Immersion energy(Data cost)====================================
datacost = zeros(back_seg_num , for_seg_num);


for i = 1:for_seg_num
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
		end
		datacost(j,i) = cost;
	end
end
datacost = datacost;

for i = 1:for_seg_num
	for j = 1:for_seg_num
		smooth_weight(i,j) = ((double(stdout_shared_length(i,j)))/double(fore_seg_length(i)) * (fore_seg_area(i)/for_image_area) + (double(stdout_shared_length(j,i)))/double(fore_seg_length(j)) * (fore_seg_area(j)/for_image_area))/2;
	end
end

smooth_weight = smooth_weight * 0.25;

for_seg_lumminance_diff = 0;
std_edge_cnt = 0;
 %{
for i = 1:for_seg_num
	for j = 1:for_seg_num
		if (j > i)
			if (smooth_weight(i,j) ~= 0)
				%(li-lj)^2 * weight 
				std_edge_cnt = std_edge_cnt + 1;
				diff = double(for_seg_lumminance(i)) - double(for_seg_lumminance(j));
				for_seg_lumminance_diff = for_seg_lumminance_diff + diff;
				diff = diff.^2;
				diff = diff * smooth_weight(i,j);
				datacost(:,i) = datacost(:,i) + diff;
				datacost(:,j) = datacost(:,j) + diff;
				%-2(li'li-li'lj-lj'li+lj'lj) * weight
				d_back_seg_lumminance = double(back_seg_lumminance);
				for k = 1:back_seg_num
					result = double(back_seg_lumminance(k));
					diff = double(for_seg_lumminance(i)) - double(for_seg_lumminance(j));
					result = result * diff;
					possible_lj = sum(d_back_seg_lumminance);
					possible_lj = possible_lj * diff;
					result = result - possible_lj;
					result = result * (-2) * smooth_weight(i,j);
					datacost(k,i) = datacost(k,i) + result;
					datacost(k,j) = datacost(k,j) + result;
				end
				
			end
		end
	end
end
%}
datacost = datacost * 0.75;
for_seg_lumminance_diff = for_seg_lumminance_diff / std_edge_cnt;


%===================Standout energy(Smooth cost)====================================

smooth_cost = zeros(back_seg_num,back_seg_num);
smooth_weight = zeros(for_seg_num,for_seg_num);

for_luminance_diff = zeros(for_seg_num,for_seg_num);

for i = 1:back_seg_num
	for j = 1:back_seg_num
		diff = double(back_seg_lumminance(i)) - double(back_seg_lumminance(j));
		diff = diff.^2;
		smooth_cost(i,j) = diff;
	end
end




%===================optimization====================================

h = GCO_Create(for_seg_num,back_seg_num);
GCO_SetDataCost(h,datacost);
GCO_SetSmoothCost(h,smooth_cost);
GCO_SetNeighbors(h,smooth_weight);
GCO_Swap(h);
opt_label = GCO_GetLabeling(h); 


%===================final result====================================
opt_luminance = [];

for i = 1:for_seg_num
	opt_luminance = [opt_luminance;back_seg_lumminance(opt_label(i))];
end


for i = 1:for_seg_num                             
    luminance(fore_regionMap == i) = opt_luminance(i);
end


[x,y] = size(luminance);           %Combine Foreground and Background to capture centroid of combined image
for i = 1:x
    for j = 1:y
        if luminance(i,j) < 220
            luminance_back((bc_x-round(y/2))+i,(bc_y-round(x/2))+j) = luminance(i,j);  %Substitute with I_crop(i,j)
        end
    end
end

figure;
imshow(luminance_back);
