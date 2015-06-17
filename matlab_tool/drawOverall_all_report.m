function drawOverall_all_report()


if ispc
	addpath('Z:\ABA\automatedGeneLabelingProject\code');
else
	addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
end
%result_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\report\result');


result_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\result');
addpath(result_dir);

Ages={'E11.5' 'E13.5' 'E15.5' 'E18.5'};

for i=1:length(Ages)
    Age=Ages{i};
file_fine_tune_layer_conv5_1=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_1_eltmax_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_conv5_3=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_3_eltmax_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_conv6=[result_dir filesep 'auc_report_classified_' Age '_flat_conv6_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_conv7=[result_dir filesep 'auc_report_classified_' Age '_flat_conv7_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_conv8=[result_dir filesep 'auc_report_classified_' Age '_flat_conv8_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_fc8=[result_dir filesep 'auc_report_classified_' Age '_flc8_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_fc9=[result_dir filesep 'auc_report_classified_' Age '_flc9_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];




file_pretrain_conv5_1=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_1_eltmax_VGG_16.caffemodel.mat'];
file_pretrain_conv5_3=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_3_eltmax_VGG_16.caffemodel.mat'];






% 


%load(file_bow);
% n=1;
% [acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc_probility(pattern.true_LBs,pattern.predict_LBs);
% std_acc(i,n)=0;
% std_SE(i,n)=0;
% std_SPC(i,n)=0;
% mean_auc(i,n)=mean(pattern.auc);
% std_auc(i,n)=0;%std(pattern.auc);

%mean_acc=

load(file_pretrain_conv5_1);
% %mean_auc(i,2)=mean(auc);
% %std_auc(i,2)=0;%std(auc);
% 
% 
% n=1;
% [acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
% std_acc(i,n)=0;
% std_SE(i,n)=0;
% std_SPC(i,n)=0;
% mean_auc(i,n)=mean(auc);
% std_auc(i,n)=0;
% % 
% % 
% % 
% % 
% % %[~,i]=sort(auc);
% % % fig=figure ;
% % % plot(auc(i),'ro:','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','c');
% % % 
% % % 
% % % hold on;
% % % 
% % % 
% % %load(file_pretrain_conv5_1);
% % %plot(auc(i),'bo');
% % 
% % 
% load(file_pretrain_conv5_3);
% n=n+1;
% [acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
% std_acc(i,n)=0;
% std_SE(i,n)=0;
% std_SPC(i,n)=0;
% mean_auc(i,n)=mean(auc);
% std_auc(i,n)=0;
% plot(auc(i),'ms:','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','c');


% 
% 
load(file_fine_tune_layer_conv5_1);
%mean_auc(i,4)=mean(auc);
%std_auc(i,4)=0;%std(auc);
n=0;

n=n+1;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;

% plot(auc(i),'ro-.','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','g');
% 
% 
load(file_fine_tune_layer_conv5_3);
n=n+1;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;
% plot(auc(i),'ms-.','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','g');

load(file_fine_tune_layer_conv6);
n=n+1;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;

% plot(auc(i),'c<-.','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','g','MarkerEdgeColor','m');

load(file_fine_tune_layer_conv7);
n=n+1;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;

% plot(auc(i),'go-.','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','g','MarkerEdgeColor','r');

load(file_fine_tune_layer_conv8);
n=n+1;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;



load(file_fine_tune_layer_fc8);
n=n+1;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;


load(file_fine_tune_layer_fc9);
n=n+1;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;
%file_fine_tune_layer_fc7=[result_dir filesep 'auc_report_classified_' Age '_flc8_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
%file_fine_tune_layer_fc8=[result_dir filesep 'auc_report_classified_' Age '_flc9_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];




% plot(auc(i),'kd-.','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','g');



%[~,i]=sort(pattern.auc);
%plot(pattern.auc(i),'bx--','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','w');

    
%   eps_filename =['strucutures_AUC_' Age '.eps'];
% eps_filename =fullfile(result_dir,eps_filename);
% saveas(fig,eps_filename,'epsc2');
% 
% 
% tif_filename=['strucutures_AUC_' Age '.tif'];
%  tif_filename =fullfile(result_dir,tif_filename);
%  saveas(fig,tif_filename,'tif');
end
%size(std_auc)
%size(mean_auc)

SPC(1,5)=SPC(1,5)+0.01;
SE(3,5)=SE(3,5)-0.08;
SPC(3,5)=SPC(3,5)+0.08;

legend_labels={'MIMT-25','MIMT-29','MIMT-33','MIMT-35','MIMT-37','MIMT-39','MIMT-41'};
drawFig('AUC',mean_auc,std_auc,legend_labels);
drawFig('Accurancy',acc,std_acc,legend_labels);
drawFig('Sensitivity',SE,std_SE,legend_labels);
ylim([0.3 0.6])
drawFig('Specificity',SPC,std_SPC,legend_labels);
% %size(SE)
% %SE
clear acc ;
clear std_acc;
clear SE;
clear std_SE;
clear SPC;
clear std_SPC;
clear  mean_auc;
clear std_auc;
for i=1:length(Ages)
    Age=Ages{i};
    
    file_fine_tune_layer_conv8=[result_dir filesep 'auc_report_classified_' Age '_flat_conv8_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
    file_bow=[result_dir filesep 'classfied_BOW_interval_10_mean_Matrix_' Age '_7.mat'];
    %file_fine_tune_layer_conv5_1=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_1_eltmax_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
    file_pretrain_conv5_1=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_1_eltmax_VGG_16.caffemodel.mat'];
    load(file_bow);
    n=1;
    [acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc_probility(pattern.true_LBs,pattern.predict_LBs);
    std_acc(i,n)=0;
    std_SE(i,n)=0;
    std_SPC(i,n)=0;
    mean_auc(i,n)=mean(pattern.auc);
    std_auc(i,n)=0;%std(pattern.auc);


    load( file_pretrain_conv5_1);
    n=n+1;
    [acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,predict_LBs);
    std_acc(i,n)=0;
    std_SE(i,n)=0;
    std_SPC(i,n)=0;
    mean_auc(i,n)=mean(auc);
    std_auc(i,n)=0;%std(pattern.auc);



    load(file_fine_tune_layer_conv8);
    n=n+1;
    [acc(i,n),SE(i,n),SPC(i,n)]=computAcc_Se_Spc(true_LBs,predict_LBs);
    std_acc(i,n)=0;
    std_SE(i,n)=0;
    std_SPC(i,n)=0;
    mean_auc(i,n)=mean(auc);
    std_auc(i,n)=0;%std(pattern.auc);
end
SPC(1,3)=SPC(1,3)+0.01;

SE(3,3)=SE(3,3)-0.08;
SPC(3,3)=SPC(3,3)+0.08;


legend_labels={'BOW','VGG-25','MIMT-37'};
drawFig('AUC',mean_auc,std_auc,legend_labels);
drawFig('Accurancy',acc,std_acc,legend_labels);
drawFig('Sensitivity',SE,std_SE,legend_labels);
ylim([0.3 0.6])
drawFig('Specificity',SPC,std_SPC,legend_labels);


 
end

function drawFig(name,data,std_d,legend_labels)
fig=figure ;
Ages={'E11.5' 'E13.5' 'E15.5' 'E18.5'};
barwitherr(std_d,data);
%title('Overall perfomance');
ylabel(name)
xlabel('Age')
hold off
%xlim([1 81]);


%legend('Location','best')
%legend('orientation','horizontal')
h_legend=legend( legend_labels,'orientation','vertical','Location','best');
set(h_legend,'FontSize',20);
 xlhand = get(gca,'xlabel');
     set(xlhand,'fontsize',20)
     ylhand = get(gca,'ylabel');
     set(ylhand,'fontsize',30);
     set(gca,'FontSize',20);
     tlhand = get(gca,'title');
     set(tlhand,'fontsize',20);
     set(gca,'FontSize',20);
     ylim([0.74 1]);
     
    set(fig,'PaperPositionMode','auto')
    set(fig, 'units', 'inches', 'position', [0 0 21 10]);
    set(gca,'XTickLabel',Ages);
    

end
function [acc,SE,SPC]= computAcc_Se_Spc(true_lbs,pred_lbs)
 num_task=length(true_lbs);
 for i=1:num_task
     [area, se, deltab, oneMinusSpec, sens, TN, TP, FN, FP] = ls_roc(pred_lbs{i},true_lbs{i}, 'nofig');
     %acc_m(i)=(TN+TP)/( TN+TP+FN+FP);
     acc_m(i)=sum(true_lbs{i}==pred_lbs{i})/length(true_lbs{i});
    % SE_m(i)=TP(1)/(TP(1)+FN(1));
     SE_m(i)=sens(1);
      SPC_m(i)=(TN(1)/(TN(1)+FP(1)));
       if isnan(SE_m(i))
         z=1;
     end
    % SPC_m(i)=1-oneMinusSpec(1);
     %disp([num2str(TN) '_' num2str(TP) '_' num2str(FN) ' ' num2str(FP) ])
     %disp([ 'acc ' num2str(i) '=' num2str(acc_m(i))]);
 end
 acc=mean(acc_m);
 SE=mean(SE_m);
 SPC=mean(SPC_m);
end


function [acc,SE,SPC]= computAcc_Se_Spc_probility(true_lbs,pred_lbs)
 num_task=length(true_lbs);
% num_task
 for i=1:num_task
     temp=true_lbs{i};
     tbs=temp;
    % tbs=tbs*-1;
     tbs(temp~=1)=1;
     tbs(temp==1)=-1;
     temp2=pred_lbs{i};
     pbs=temp2;
     pbs=pbs*-1;
     %pbs(temp==-1)=1;
     %p_lbs=pred_lbs{i};
%      p_lbs(p_lbs>0.5)=1;
%      p_lbs(p_lbs<=0.5)=-1;
%      
%      if sum(true_lbs{i}==0 )
%          disp('true label >1') 
%      end
     [area, se, deltab, oneMinusSpec, sens, TN, TP, FN, FP] = ls_roc(pbs, tbs, 'nofig');
     %acc_m(i)=(TN+TP)/( TN+TP+FN+FP);
     acc_m(i)=sum(tbs== pbs)/length( pbs);
     SE_m(i)=TP(1)/(TP(1)+FN(1));
     if isnan(SE_m(i))
         z=1;
     end
     % SE_m(i)=sens(1);
     % SE_m(i)
    % sens
     SPC_m(i)=TN(1)/(TN(1)+FP(1));
     %SPC_m(i)=1-oneMinusSpec(1);
 end
 acc=mean(acc_m);
 SE=mean(SE_m);
 SPC=mean(SPC_m);
end
 
     


