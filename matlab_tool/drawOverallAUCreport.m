function drawOverallAUCreport()


if ispc
	addpath('Z:\tzeng\ABA\automatedGeneLabelingProject\code');
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


file_bow=[result_dir filesep 'classfied_BOW_interval_10_mean_Matrix_' Age '_7.mat'];

file_pretrain_conv5_1=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_1_eltmax_VGG_16.caffemodel.mat'];
file_pretrain_conv5_3=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_3_eltmax_VGG_16.caffemodel.mat'];





% 


load(file_bow);
n=1;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(pattern.true_LBs,pattern.pred_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(pattern.auc);
std_auc(i,n)=0;%std(pattern.auc);

%mean_acc=

load(file_pretrain_conv5_1);
%mean_auc(i,2)=mean(auc);
%std_auc(i,2)=0;%std(auc);


n=2;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,pred_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;




%[~,i]=sort(auc);
% fig=figure ;
% plot(auc(i),'ro:','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','c');
% 
% 
% hold on;
% 
% 
%load(file_pretrain_conv5_1);
%plot(auc(i),'bo');


load(file_pretrain_conv5_3);
n=3;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,pred_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;
% plot(auc(i),'ms:','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','c');


% 
% 
load(file_fine_tune_layer_conv5_1);
%mean_auc(i,4)=mean(auc);
%std_auc(i,4)=0;%std(auc);


n=4;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,pred_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;

% plot(auc(i),'ro-.','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','g');
% 
% 
load(file_fine_tune_layer_conv5_3);
n=5;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,pred_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;
% plot(auc(i),'ms-.','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','g');

load(file_fine_tune_layer_conv6);
n=6;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,pred_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;

% plot(auc(i),'c<-.','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','g','MarkerEdgeColor','m');

load(file_fine_tune_layer_conv7);
n=7;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,pred_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;

% plot(auc(i),'go-.','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','g','MarkerEdgeColor','r');

load(file_fine_tune_layer_conv8);
n=8;
[acc(i,n),SE(i,n),SPC(i,n)]= computAcc_Se_Spc(true_LBs,pred_LBs);
std_acc(i,n)=0;
std_SE(i,n)=0;
std_SPC(i,n)=0;
mean_auc(i,n)=mean(auc);
std_auc(i,n)=0;

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

fig=figure ;
barwitherr(std_auc,mean_auc);
title('Overall perfomance');
ylabel('AUC')
xlabel('Age')
hold off
%xlim([1 81]);
legend_labels={'BOW','VGG-25','VGG-29','MIMT-25','MIMT-29','MIMT-33','MIMT-35','MIMT-37'};

%legend('Location','best')
%legend('orientation','horizontal')
h_legend=legend( legend_labels,'orientation','vertical','Location','best');
set(h_legend,'FontSize',20);
 xlhand = get(gca,'xlabel');
     set(xlhand,'fontsize',20)
     ylhand = get(gca,'ylabel');
     set(ylhand,'fontsize',20);
     set(gca,'FontSize',20);
     tlhand = get(gca,'title');
     set(tlhand,'fontsize',20);
     set(gca,'FontSize',20);
     ylim([0.74 0.92]);
     
     
    set(fig,'PaperPositionMode','auto')
    set(fig, 'units', 'inches', 'position', [0 0 21 10]);
    set(gca,'XTickLabel',Ages);
 
 
 
end
function [acc,SE,SPC]= computAcc_Se_Spc(true_lbs,pred_lbs)
 num_task=length(true_lbs);
 for i=1:num_task
     [area, se, deltab, oneMinusSpec, sens, TN, TP, FN, FP] = ls_roc(true_lbs{i}, pred_lbs{i},'nofig');
     acc_m(i)=(TN+TP)/( TN+TP+FN+FP);
     SE_m(i)=sens(1);
     SPC_m(i)=1-oneMinusSpec(1);
 end
 acc=mean(acc_m);
 SE=mean(SE_m);
 SPC=mean(SPC_m);
end
 
     


