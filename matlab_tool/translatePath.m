function path= translatePath( pstring )

% p = strrep(p,'$',[pstring '\']);
%import java.lang.*;
%username =char(System.getProperty('user.name'));
username ='tzeng';

if ispc
    path = strrep(pstring,'/','\');
    path = strrep(path,['\home\' username],'Z:');
    %p = strrep(p,':',';');  
else
    path = strrep(pstring,'\','/');
    path = strrep(path,'z:',['/home/' username]);
    path = strrep(path,'Z:',['/home/' username]);
    path = strrep(path,';',':');
    

end
% function path=tranlateRootDir(path)
%     path= translatePath( path );    
%     if ispc
%         
%     else
%         if strmp(':',path(2))
%             path =['/home/tzeng' path(3:end)];
%         end
%     end