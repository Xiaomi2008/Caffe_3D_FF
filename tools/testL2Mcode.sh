db_path_label=/home/tzeng/ABA/autoGenelable_multi_lables_proj/temp/train_label_flat_conv5_3_eltmax
db_path_feats=/home/tzeng/ABA/autoGenelable_multi_lables_proj/temp/train_feature_flat_conv5_3_eltmax
out_mat_file=/home/tzeng/ABA/autoGenelable_multi_lables_proj/data/feats.mat
python feature_LDB_to_mat.py --label_db $db_path_label --feature_db $db_path_feats --mat_file $out_mat_file