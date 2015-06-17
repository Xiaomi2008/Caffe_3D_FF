function idx=findFirstCannotAnnotatedLabel(serial_section_info,ontology_level)
   l=size(serial_section_info(1).lable{ontology_level}(:,1),1);
   len=length(serial_section_info);
   level_labs=zeros(len,l);
   for i=1:len
	   level_labs(i,:)=serial_section_info(i).lable{ontology_level}(:,1); % 1 =pattern.
   end
	
	idx=~(level_labs(:,1)==0);
    %labels=level_labs(idx,:);
    %matrix=matrix(idx,:);
end
