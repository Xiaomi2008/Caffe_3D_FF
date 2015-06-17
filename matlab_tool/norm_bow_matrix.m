function  matrix=norm_bow_matrix(matrix)
   % [r,c] =size(matrix);
   % minx=min(matrix);
   % maxx=max(matrix);
   % for i=1:c
       % matrix(:,i)=(matrix(:,i)-minx(i))./(maxx(i)-minx(i));
   % end
   % for i = 1: length(index_zero)
    % tr_matrix(:, index_zero(i)) = zeros( size(tr_matrix,1), 1);
   % end
   
   
max_value = 1;
min_value = 0;

max_feat = max(matrix);
min_feat = min(matrix);

diff = max_feat - min_feat;
index_zero = find(diff == 0);

for i = 1:size(matrix,2)
    matrix(:,i) = ((matrix(:,i)-min_feat(i))/(max_feat(i)-min_feat(i)))*(max_value-min_value)+min_value;
%     te_matrix(:,i) = ((te_matrix(:,i)-min_feat(i))/(max_feat(i)-min_feat(i)))*(max_value-min_value)+min_value;
    
end

for i = 1: length(index_zero)
    matrix(:, index_zero(i)) = zeros( size(matrix,1), 1);
end
   
   
end