



in_img = imread('D:\Works\Dataset\Interactive_Image_Segmentation\GrabCut\data_GT\scissors.jpg');
box_list = [40, 44, 567, 376];    % Left, Top, Width, Height

handles.Processbar = 0;
Beta = 0.3;
k = 6;
G = 50;
maxIter = 10;
diffThreshold = 0.001;

fixed_bg = ones(size(in_img,1),size(in_img,2));
fixed_bg(box_list(2):box_list(2)+box_list(4)-1,box_list(1):box_list(1)+box_list(3)-1) = 0;
fixed_bg = fixed_bg > 0;

in_img = double(in_img);

L = GCAlgo(in_img, fixed_bg,k,G,maxIter, Beta, diffThreshold, handles.Processbar);
L = double(1 - L);

[row_list, col_list] = find(L>0);
if isempty(row_list) || numel(row_list) < 100
    [row_list, col_list] = find(fixed_bg==0);
end
seg_mask = [row_list, col_list];

figure; imshow(L)