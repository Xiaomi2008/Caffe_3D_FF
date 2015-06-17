function drawAUCreportForStructure(Age)


if ispc
	addpath('Z:\tzeng\ABA\automatedGeneLabelingProject\code');
else
	addpath('/home/tzeng/ABA/automatedGeneLabelingProject/code');
end
%result_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\report\result');


result_dir=translatePath('Z:\ABA\autoGenelable_multi_lables_proj\result');
addpath(result_dir);

file_fine_tune_layer_conv5_1=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_1_eltmax_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_conv5_3=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_3_eltmax_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_conv6=[result_dir filesep 'auc_report_classified_' Age '_flat_conv6_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_conv7=[result_dir filesep 'auc_report_classified_' Age '_flat_conv7_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];
file_fine_tune_layer_conv8=[result_dir filesep 'auc_report_classified_' Age '_flat_conv8_finetune_vgg_16_concat_convol_multi_sharedwieght_extra_fcs_net_slice10_ish_10_SLICE_ISH_' Age '_iter_10000.mat'];


file_bow=[result_dir filesep 'classfied_BOW_interval_10_mean_Matrix_' Age '_7.mat'];

file_pretrain_conv5_1=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_1_eltmax_VGG_16.caffemodel.mat'];
file_pretrain_conv5_3=[result_dir filesep 'auc_report_classified_' Age '_flat_conv5_3_eltmax_VGG_16.caffemodel.mat'];




%load(file_bow);
%[~,i]=sort(pattern.auc);
%plot(pattern.auc(i),'b^-.','LineWidth',2,'MarkerSize',12,'MarkerFaceColor','y');
% 


load(file_pretrain_conv5_1);
[~,i]=sort(auc);
fig=figure ;
plot(auc(i),'co-.','LineWidth',2,'MarkerSize',11,'MarkerFaceColor','c','MarkerEdgeColor','k');


hold on;
% 
% 
%load(file_pretrain_conv5_1);
%plot(auc(i),'bo');


%load(file_pretrain_conv5_3);
%plot(auc(i),'ms-.','LineWidth',2,'MarkerSize',11,'MarkerFaceColor','c','MarkerEdgeColor','b');


% 
% 
load(file_fine_tune_layer_conv5_1);
plot(auc(i),'ro-.','LineWidth',2,'MarkerSize',11,'MarkerFaceColor','r','MarkerEdgeColor','k');
% 
% 
%load(file_fine_tune_layer_conv5_3);
%plot(auc(i),'ms-.','LineWidth',2,'MarkerSize',11,'MarkerFaceColor','r','MarkerEdgeColor','b');

%load(file_fine_tune_layer_conv6);
%plot(auc(i),'c<-.','LineWidth',2,'MarkerSize',11,'MarkerFaceColor','g','MarkerEdgeColor','m');

%load(file_fine_tune_layer_conv7);
%plot(auc(i),'ro-.','LineWidth',2,'MarkerSize',11,'MarkerFaceColor','g','MarkerEdgeColor','b');

load(file_fine_tune_layer_conv8);
plot(auc(i),'kd-.','LineWidth',2,'MarkerSize',11,'MarkerFaceColor','g');




load(file_bow);
%[~,i]=sort(pattern.auc);
plot(pattern.auc(i),'b^-.','LineWidth',2,'MarkerSize',12,'MarkerFaceColor','y');

%load(file_bow);
%[~,i]=sort(pattern.auc);
%plot(pattern.auc(i),'b^-.','LineWidth',2,'MarkerSize',12,'MarkerFaceColor','y');
title(Age);
ylabel('AUC')
xlabel('Brain Structures')
hold off
xlim([1 81]);
ylim([0.7 1]);
%legend('VGG-25','VGG-29','MIMT-25','MIMT-29','MIMT-33','MIMT-35','MIMT-37','BOW');
legend('VGG-25','MIMT-25','MIMT-37','BOW');

legend('Location','best')
legend('orientation','horizontal')
 xlhand = get(gca,'xlabel');
     set(xlhand,'fontsize',20)
     ylhand = get(gca,'ylabel');
     set(ylhand,'fontsize',20);
     set(gca,'FontSize',20);
     tlhand = get(gca,'title');
     set(tlhand,'fontsize',20);
     set(gca,'FontSize',20);
     
     
     set(fig,'PaperPositionMode','auto')
    set(fig, 'units', 'inches', 'position', [2 2 21 10]);
    
  eps_filename =['strucutures_AUC_' Age '.eps'];
eps_filename =fullfile(result_dir,eps_filename);
saveas(fig,eps_filename,'epsc2');


tif_filename=['strucutures_AUC_' Age '.tif'];
 tif_filename =fullfile(result_dir,tif_filename);
 saveas(fig,tif_filename,'tif');
 
 
 
end

