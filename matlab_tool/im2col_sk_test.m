function im2col_sk_test(give_p)
h=100;
w=1024;
parts=128;
len_part =h*w/parts;

%give_p=0;

h_s= give_p*len_part/w;
h_e= (give_p+1)*len_part/w;

w_s=mod(give_p*len_part,w);
w_e=mod((give_p+1)*len_part,w);

h_s
h_e
w_s
w_e
end