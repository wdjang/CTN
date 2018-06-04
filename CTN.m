function seg_track_one = CTN(param, db_name, net)

% Load frame list
frame_list = dir(fullfile(param.db_path,db_name,'*.png'));
if isempty(frame_list)
    frame_list = dir(fullfile(param.db_path,db_name,'*.jpg'));
end
if isempty(frame_list)
    frame_list = dir(fullfile(param.db_path,db_name,'*.bmp'));
end

seg_track_one = cell(length(frame_list),1);

% First frame
fprintf('(frame %03d)',1);
prev_frame = imread(fullfile(param.db_path,db_name,frame_list(1).name));
frame_name = frame_list(1).name(1:end-4);
prev_segmap = im2double(imread(fullfile(param.gt_path,db_name,sprintf('%s.png',frame_name))));
seg_track_one{1} = prev_segmap;

% Build foreground / background gaussian mixture model
gmm_prev = GCAlgo_first( double(prev_frame), logical(prev_segmap), param );
gmm_init = gmm_prev;

[h_size, w_size] = size(prev_segmap);
x_coord = repmat(1:w_size,h_size,1);
y_coord = repmat([1:h_size]',1,w_size);

% Later frames
for frame_id = 2:length(frame_list)
    
    fprintf(repmat('\b',1,4));
    fprintf('%03d)',frame_id);

    curr_frame = imread(fullfile(param.db_path,db_name,frame_list(frame_id).name));
    
    bflow_map = readFlowFile(fullfile(param.flow_path,db_name,sprintf('b_%s.flo',frame_list(frame_id).name(1:end-4))));
    
    bx_wmap = min(max(round(x_coord + bflow_map(:,:,1)),1),w_size);
    by_wmap = min(max(round(y_coord + bflow_map(:,:,2)),1),h_size);
    b_wmap = sub2ind([h_size,w_size], by_wmap, bx_wmap);
    
    prop_map = prev_segmap(b_wmap);
    curr_propmapf = prop_map;
    
    %% Occlusion detection by Backward-Forward error
    fflow_map = readFlowFile(fullfile(param.flow_path,db_name,sprintf('f_%s.flo',frame_list(frame_id-1).name(1:end-4))));
    
    
    xf_flowmap = fflow_map(:,:,1);
    yf_flowmap = fflow_map(:,:,2);
    bbx_wmap = min(max(round(bx_wmap + xf_flowmap(b_wmap)),1),w_size);
    bby_wmap = min(max(round(by_wmap + yf_flowmap(b_wmap)),1),h_size);
    berr_map = sqrt( (bby_wmap-y_coord).^2 + (bbx_wmap-x_coord).^2 ) / ( 0.005 * (h_size + w_size) ) ;
    
    [ bgLogPL, fgLogPL ]  = OcclusionDetection( double(curr_frame), param.bgk, param.fgk, gmm_init, gmm_prev );
    
    init_occmap = berr_map > 0.5;
    
    border_len = 100;
    init_occmap(1:border_len,:) = 0;
    init_occmap(end-border_len+1:end,:) = 0;
    init_occmap(:,1:border_len) = 0;
    init_occmap(:,end-border_len+1:end) = 0;
    
    occ_map = init_occmap.*( bgLogPL > fgLogPL );
    
    
    %%
    [y_list, x_list] = find(curr_propmapf>0);
    min_y = min(y_list(:));
    min_x = min(x_list(:));
    max_y = max(y_list(:));
    max_x = max(x_list(:));
    mrg_len = 50;
    
    pat_t = max(min_y - mrg_len,1);
    pat_b = min(max_y + mrg_len,h_size);
    pat_l = max(min_x - mrg_len,1);
    pat_r = min(max_x + mrg_len,w_size);
    datac_patch = curr_frame(pat_t:pat_b,pat_l:pat_r,:);
    propc_patch = curr_propmapf(pat_t:pat_b,pat_l:pat_r);
    if isempty(datac_patch)
        pat_t = 1;
        pat_b = h_size;
        pat_l = 1;
        pat_r = w_size;
        datac_patch = curr_frame(pat_t:pat_b,pat_l:pat_r,:);
        propc_patch = curr_propmapf(pat_t:pat_b,pat_l:pat_r);
    end
    
    %%
    datac = im2data(datac_patch, param.datac_sz);
    propc = im2prop(propc_patch, param.propc_sz);
    propb = single(1-propc);
    
    cell_data = cell(3,1);
    cell_data{1} = datac;
    cell_data{2} = propc;
    cell_data{3} = propb;
    
    seg_map = net.forward(cell_data);
    
    im = seg_map{1};
    im = permute(im, [2, 1]);
    im = imresize(im, [size(propc_patch,1), size(propc_patch,2)]);
    
    imb = seg_map{2};
    imb = permute(imb, [2, 1]);
    imb = imresize(imb, [size(propc_patch,1), size(propc_patch,2)]);
    
    imf = seg_map{3};
    imf = permute(imf, [2, 1]);
    imf = imresize(imf, [size(propc_patch,1), size(propc_patch,2)]);
    
    deep_map = zeros(h_size,w_size);
    deep_map(pat_t:pat_b,pat_l:pat_r) = im;

    deep_mapf = zeros(h_size,w_size);
    deep_mapf(pat_t:pat_b,pat_l:pat_r) = imf;

    deep_mapb = ones(h_size,w_size);
    deep_mapb(pat_t:pat_b,pat_l:pat_r) = imb;
    
    %%    
    deep_bnrymap = double( deep_map > param.rho );

    init_seg = occ_map | deep_bnrymap;
    
    fixed_bg = (deep_mapb > param.sigma );
    fixed_bg = fixed_bg & (1-occ_map);
    fixed_fg = (deep_mapf > param.sigma );
    
    %%% Get the flow gradient
    bflow_remap = imresize(bflow_map, 0.5);
    gradHf = bflow_remap(:,2:end,:) - bflow_remap(:,1:end-1,:);
    gradVf = bflow_remap(2:end,:,:) - bflow_remap(1:end-1,:,:);

    gradHf = sum(gradHf.^2, 3);
    gradVf = sum(gradVf.^2, 3);
    
    %%% Use the gradient to calculate the graph's inter-pixels weights
    hCf = exp(-param.beta.*gradHf./mean(gradHf(:)));
    vCf = exp(-param.beta.*gradVf./mean(gradVf(:)));

    %%% These matrices will evantually use as inputs to Bagon's code
    hCf = [hCf, zeros(size(hCf,1),1)];
    vCf = [vCf; zeros(1, size(vCf,2))];
    
    [gcout_map, gmm_prev] = a_GCAlgo(double(curr_frame), logical(fixed_fg), logical(fixed_bg), init_seg, param, param.G, param.maxIter, param.beta, param.diffThreshold,...
        hCf, vCf, gmm_prev, gmm_init, gmm_prev);

    gcout_map = double(1 - gcout_map);
    curr_segmap = gcout_map.*init_seg;
    curr_segmap = imfill(logical(curr_segmap),'holes');
    curr_segmap = double(curr_segmap);
    if ( sum(curr_segmap(:)) == 0 && sum(deep_bnrymap(:)) > 0 )
        curr_segmap = deep_bnrymap;
    end
  
    seg_track_one{frame_id} = curr_segmap;
    
    prev_segmap = curr_segmap;
    
end

fprintf(repmat('\b',1,11));
fprintf('complete');
fprintf('\n');

end


% -----------------------------------------------------------------------------
function data = im2data(im, data_sz)
% -----------------------------------------------------------------------------
[~,~,c] = size(im);
if c == 1
  im = cat(3,im,im,im);
end

data = imresize(im, [data_sz(1),data_sz(2)]);
data = single(data(:, :, [3, 2, 1]));    % convert from RGB to BGR              
data = permute(data, [2, 1, 3]);         % permute width and height            
data(:,:,1) = data(:,:,1) - 103.939; % mean-val b
data(:,:,2) = data(:,:,2) - 116.779; % mean-val g
data(:,:,3) = data(:,:,3) - 123.68; % mean-val r

end

% -----------------------------------------------------------------------------
function data = im2prop(im, data_sz)
% -----------------------------------------------------------------------------
[~,~,c] = size(im);
if c == 3
  im = rgb2gray(im);
end
im = im2double(im);

data = imresize(im, [data_sz(1),data_sz(2)]);
if max(data(:)) > 0
    data = data / max(data(:));
end
data = permute(single(data), [2, 1]);         % permute width and height       

end