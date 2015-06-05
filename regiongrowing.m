function regiongrowing(I,x,y,regionLabel)
global regionMap;
[Ix,Iy] = size(I);
checkList = [x y];
listSize = 1;
first = 1;
last = 1;
neighbor = [-1 0; 1 0; 0 -1; 0 1];
while (listSize > 0)
	candidate = checkList(first,:);
	regionMap(candidate(1,1),candidate(1,2)) = regionLabel;
	  
	for i = 1:4
		m = candidate(1,1) + neighbor(i,1);
		n = candidate(1,2) + neighbor(i,2);
		if (m < 1 || m >= Ix) continue; end
		if (n < 1 || n >= Iy) continue; end
		
		if ( abs( double(I(x,y)) - double(I(m,n))) < 1 && regionMap(m,n) == 0)
			newPoint = [m n];
			isNew = 1;
			for j = first:last
				if (newPoint(1,1) == checkList(j,1) && newPoint(1,2) == checkList(j,2))
					isNew = 0;
					break;
				end
			end
			if (isNew == 1)
				checkList = [checkList ; newPoint];
				last = last + 1;
				listSize = listSize + 1;
			end
		end
	end
	
	checkList(first,:) = 0;
	first = first + 1;
	listSize = listSize - 1;
end