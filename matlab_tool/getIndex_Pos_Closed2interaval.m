function index_closest=getIndex_Pos_Closed2interaval(section_info_struct,pos_interval_all)
pos=section_info_struct.pos;
dist_p=zeros(1,length(pos));
index_closest=zeros(1,length(pos_interval_all));
	for i=1:length(pos_interval_all)
		cur_interv=pos_interval_all(i);
		for j=1:length(pos)
			dist_p(j)=abs(cur_interv-pos(j));
		end
		[~,idx]=min(dist_p);
		index_closest(i)=idx;
	end
end