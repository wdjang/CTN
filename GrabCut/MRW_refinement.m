% MRW refinement

function [finalLabel, gmm_curr] = MRW_refinement( im, deep_mapf, deep_mapb, deep_map, ...
    occ_errmap, init_seg, K, occ_map, roi_box, gmm_prev, gmm_init, trans_mat )

% function [finalLabel, gmm_curr] = MRW_refinement( im, deep_mapf, deep_mapb, deep_map, ...
%     occ_errmap, init_seg, K, occ_map, roi_box, gmm_prev, gmm_init, trans_mat, sp_map )
%%
[h_size, w_size] = size(deep_mapf);

deep_vecf = reshape(deep_mapf,h_size*w_size,1);
deep_vecb = reshape(deep_mapb,h_size*w_size,1);
deep_vecf = deep_vecf/sum(deep_vecf);
deep_vecb = deep_vecb/sum(deep_vecb);


mix_map = min(deep_map + occ_errmap,1);
prob_vecf = reshape(mix_map,h_size*w_size,1);
prob_vecb = reshape(1-mix_map,h_size*w_size,1);
prob_vecf = prob_vecf/sum(prob_vecf);
prob_vecb = prob_vecb/sum(prob_vecb);

% Fixed FG & BG setting
fixed_bg = (deep_mapb > 0.9);
fixed_bg = fixed_bg & (1-occ_map);
fixed_bg(roi_box) = 1;
fixed_fg = (deep_mapf > 0.9);
fixed_fg(roi_box) = 0;



    tic;

    fgMeanInit = [];
    bgMeanInit = [];
for ii = 1:100000
    if mod(ii,100) == 1
        segment_label = reshape(prob_vecf > prob_vecb,[h_size,w_size]);
        fg_labels = segment_label | fixed_fg;
        fg_labels = fg_labels & (1-fixed_bg);
        bg_labels = (1-segment_label) | fixed_bg;
        bg_labels = bg_labels & (1-fixed_fg);
        [bgLogPL, fgLogPL, bgMeanInit, fgMeanInit, ~ ] =  CalcLogPLikelihood(im, K, ...
            find(bg_labels), find(fg_labels), bgMeanInit, fgMeanInit, gmm_prev, gmm_init);
%         fg_gmm = bgLogPL ./ ( fgLogPL + bgLogPL );
%         bg_gmm = fgLogPL ./ ( fgLogPL + bgLogPL );
        fg_list = find(prob_vecf > prob_vecb);
        fg_gmm = zeros(h_size*w_size,1);
        fg_gmm(fg_list) = -log(prob_vecf(fg_list)-prob_vecb(fg_list));
        bg_list = find(prob_vecb > prob_vecf);
        bg_gmm = zeros(h_size*w_size,1);
        bg_gmm(bg_list) = -log(prob_vecb(bg_list)-prob_vecf(bg_list));
        fg_gmm = reshape(fg_gmm,[h_size,w_size]);
        bg_gmm = reshape(bg_gmm,[h_size,w_size]);
%         fg_gmm = fgLogPL < bgLogPL;
%         bg_gmm = fgLogPL > bgLogPL;
%         fg_labels = max(fg_gmm, fixed_fg/sum(fixed_fg(:)));
        fg_gmm = fg_gmm.*(1-fixed_bg);
%         bg_labels = max(bg_gmm, fixed_bg/sum(fixed_bg(:)));
        bg_gmm = bg_gmm.*(1-fixed_fg);
        repul_vecf = reshape(fg_gmm,h_size*w_size,1);
        repul_vecb = reshape(bg_gmm,h_size*w_size,1);
        repul_vecf = repul_vecf/sum(repul_vecf);
        repul_vecb = repul_vecb/sum(repul_vecb);
    end
%     repul_vecf = (prob_vecf.^2) ./ (prob_vecf+prob_vecb);
%     repul_vecb = (prob_vecb.^2) ./ (prob_vecf+prob_vecb);
%     repul_vecf = repul_vecf/sum(repul_vecf);
%     repul_vecb = repul_vecb/sum(repul_vecb);

    restart_vecf = (1*repul_vecf + 1*deep_vecf)/2;
    restart_vecb = (1*repul_vecb + 1*deep_vecb)/2;

    restart_prob = 0.03;
    prob_vecf = (1-restart_prob)*trans_mat*prob_vecf+restart_prob*restart_vecf;
    prob_vecb = (1-restart_prob)*trans_mat*prob_vecb+restart_prob*restart_vecb;
%     prob_vecf = restart_prob*((speye(h_size*w_size)-(1-restart_prob)*trans_mat)\restart_vecf);
%     prob_vecb = restart_prob*((speye(h_size*w_size)-(1-restart_prob)*trans_mat)\restart_vecb);
    if mod(ii,100) == 1
        subplot(1,2,1); imagesc(reshape(prob_vecf,h_size,w_size));
        subplot(1,2,2); imagesc(reshape(prob_vecb,h_size,w_size));
        toc;
        input('waitkey');
        tic;
    end
end

end
% 
% % Fixed FG & BG setting
% fixed_bg = (deep_mapb > 0.9);
% fixed_bg = fixed_bg & (1-occ_map);
% fixed_bg(roi_box) = 1;
% fixed_fg = (deep_mapf > 0.9);
% fixed_fg(roi_box) = 0;
% 
% % im = imresize(im, 0.5);
% % fixedBG = imresize(fixedBG, 0.5);
% % fixedBG = fixedBG > 0.5;
% % fixedFG = imresize(fixedFG, 0.5);
% % fixedFG = fixedFG > 0.5;
% % init_seg = imresize(init_seg, 0.5);
% % init_seg = double(init_seg > 0.5);
% 
% %%
% occ_vec = zeros(num_sp,1);
% deep_vec = zeros(num_sp,1);
% deep_vecf = zeros(num_sp,1);
% deep_vecb = zeros(num_sp,1);
% for sp_id = 1:num_sp
%     find_list = find(sp_map==sp_id);
%     deep_vecf(sp_id) = mean(deep_mapf(find_list));
%     deep_vecb(sp_id) = mean(deep_mapb(find_list));
%     deep_vec(sp_id) = mean(deep_map(find_list));
%     occ_vec(sp_id) = mean(occ_errmap(find_list));
% end
% deep_vecf = deep_vecf / sum(deep_vecf);
% deep_vecb = deep_vecb / sum(deep_vecb);
% deep_vec = deep_vec / sum(deep_vec);
% occ_vec = occ_vec / sum(occ_vec);
% 
% prob_vecf = 0.3*((eye(num_sp)-(1-0.3)*trans_mat)\deep_vecf);
% prob_vecb = 0.3*((eye(num_sp)-(1-0.3)*trans_mat)\deep_vecb);
% 
% prob_mapf = zeros(h_size,w_size);
% prob_mapb = zeros(h_size,w_size);
% for sp_id = 1:num_sp
%     find_list = find(sp_map==sp_id);
%     prob_mapf(find_list) = prob_vecf(sp_id);
%     prob_mapb(find_list) = prob_vecb(sp_id);
% end
% %%
% 
% 
% %%%%%%%%%%%%%%%%%%%%%
% %%% Get definite labels defining absolute Background :
% prevLabel = double(1-init_seg);
% 
% %%%%%%%%%%%%%%%%%%%%%
% %%% Calculate the smoothness term defined by the entire image's RGB values
% bNormGrad = true;
% 
% %%% Get the image gradient
% gradH = im(:,2:end,:) - im(:,1:end-1,:);
% gradV = im(2:end,:,:) - im(1:end-1,:,:);
% 
% gradH = sum(gradH.^2, 3);
% gradV = sum(gradV.^2, 3);
% 
% %%% Use the gradient to calculate the graph's inter-pixels weights
% if ( bNormGrad )
%     hC = exp(-Beta.*gradH./mean(gradH(:)));
%     vC = exp(-Beta.*gradV./mean(gradV(:)));
% else
%     hC = exp(-Beta.*gradH);
%     vC = exp(-Beta.*gradV);
% end
% 
% %%% These matrices will evantually use as inputs to Bagon's code
% hC = [hC zeros(size(hC,1),1)];
% vC = [vC ;zeros(1, size(vC,2))];
% sc = [0 G;G 0];
% 
% % hC = ( hC + [ct_map(:,2:end), zeros(size(hC,1),1)] + hCf ) / 3;
% % vC = ( vC + [ct_map(2:end,:); zeros(1,size(vC,2))] + vCf ) / 3;
% 
% hC = ( hC + hCf ) / 2;
% vC = ( vC + vCf ) / 2;
% 
% %     fprintf('MRF-pairwise: %.3f\n',toc(stt_tic));
%     
% currLabel = prevLabel;
%     
% %%%%%%%%%%%%%%%%%%%%%
% %%% Start the EM iterations :
% bgMean = [];
% fgMean = [];
% for iter=1:maxIterations
% %     stt_tic = tic;
%     bgIds   = find(prevLabel == 1);
%     fgIds   = find(prevLabel == 0);
%     
%     %%% Use NOT FIXED labels to get the Log Probability Likelihood 
%     %%% of the pixels to a GMM color model (inferred from the labels...)
%     bgMeanInit = bgMean;
%     fgMeanInit = fgMean;
%     if numel(fgIds)<K || numel(bgIds)<K
%         gmm_curr = gmm_prev;
%         break;
%     end
%     [bgLogPL, fgLogPL, bgMean, fgMean, gmm_curr ] =  CalcLogPLikelihood(im, K, bgIds,fgIds, bgMeanInit, fgMeanInit, gmm_prev, gmm_init);
%     
% %     fprintf('MRF-GMM: %.3f\n',toc(stt_tic));
% %     stt_tic = tic;
%     %%% Use our A-Priori knowledge of Background labels & set the Forground
%     %%% weights according to it.
%     fgLogPL(fixedBG) = max(max(fgLogPL));
%     bgLogPL(fixedFG) = max(max(bgLogPL));
%         
%     %%% Now that we have all inputs, calculate the min-cut of the graph
%     %%% using Bagon's code. Not much to explain here, for more details read
%     %%% the graph cut documentation in the   GraphCut.m    file.
%     dc = cat(3, bgLogPL, fgLogPL);
%     graphHandle = GraphCut('open', dc , sc, vC, hC);
%     graphHandle = GraphCut('set', graphHandle, int32(prevLabel == 0));
%     [graphHandle, currLabel] = GraphCut('expand', graphHandle);
%     currLabel = 1 - currLabel;
%     GraphCut('close', graphHandle);
%     
% %     fprintf('MRF-GraphCut: %.3f\n',toc(stt_tic));
%     %%% Break if the current result is somewhat similar to the previous result
%     if nnz(prevLabel(:)~=currLabel(:)) < diffThreshold*numel(currLabel)
%         break;
%     end
%     
%     prevLabel = currLabel;
% %     imagesc(currLabel);
% %     input('wait key');
%         
% end
% finalLabel = double(imresize(currLabel,2.0)>0.5);
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
function [ bgLogPL, fgLogPL, bgMean, fgMean, gmm_curr ] = CalcLogPLikelihood(im, K, bgIds,fgIds , bgMeanInit, fgMeanInit, ...
    gmm_prev, gmm_init)

numPixels = size(im,1) * size(im,2);
allBGLogPL = zeros(numPixels,K);
allFGLogPL = zeros(numPixels,K);
allBGLogPLprev = zeros(numPixels,K);
allFGLogPLprev = zeros(numPixels,K);
allBGLogPLinit = zeros(numPixels,K);
allFGLogPLinit = zeros(numPixels,K);
% allBGLogPLhist = zeros(numPixels,K*10);
% allFGLogPLhist = zeros(numPixels,K*10);

%%% Seperate color channels 
R = im(:,:,1);
G = im(:,:,2);
B = im(:,:,3);

%%% Prepare the color datasets according to the input labels 
imageValues = [R(:) G(:) B(:)];
bgValues = [R(bgIds)    G(bgIds)     B(bgIds)];
fgValues = [R(fgIds)    G(fgIds)     B(fgIds)];
numBGValues = size(bgValues,1);
numFGValues = size(fgValues,1);

%%%%%%
% Use a 'manual' way to calculate the GMM parameters, instead of using
% Matlab's gmdistribution.fit() function. This is due to better speed 
% results..
% Start with Kmeans centroids calculation :
opts = statset('kmeans');
opts.MaxIter = 1;  % 40

%     stt_tic = tic;
if ( ~isempty(bgMeanInit) && ~isempty(fgMeanInit) )
    [bgClusterIds, bgMean] = kmeans_norand(bgValues, K, 'start', bgMeanInit,  'emptyaction','singleton' ,'Options',opts);
    [fgClusterIds, fgMean] = kmeans_norand(fgValues, K, 'start', fgMeanInit,  'emptyaction','singleton', 'Options',opts);
%     [bgClusterIds, bgMean] = kmeans(bgValues, K, 'start', bgMeanInit,  'emptyaction','singleton' ,'Options',opts);
%     [fgClusterIds, fgMean] = kmeans(fgValues, K, 'start', fgMeanInit,  'emptyaction','singleton', 'Options',opts);
else
    [bgClusterIds, bgMean] = kmeans_norand(bgValues, K, 'emptyaction','singleton' ,'Options',opts);
    [fgClusterIds, fgMean] = kmeans_norand(fgValues, K, 'emptyaction','singleton', 'Options',opts);
%     [bgClusterIds, bgMean] = kmeans(bgValues, K, 'emptyaction','singleton' ,'Options',opts);
%     [fgClusterIds, fgMean] = kmeans(fgValues, K, 'emptyaction','singleton', 'Options',opts);
end

%     fprintf('MRF-GMM-kmeans: %.3f\n',toc(stt_tic));
%     stt_tic = tic;
checkSumFG = 0;
checkSumBG = 0;


gmm_curr.bgMean = bgMean;
gmm_curr.fgMean = fgMean;

gmm_curr.bgCovarianceMat = cell(K,1);
gmm_curr.fgCovarianceMat = cell(K,1);

gmm_curr.bgCovarianceMatDet = zeros(K,1);
gmm_curr.fgCovarianceMatDet = zeros(K,1);

gmm_curr.bgGaussianWeight = zeros(K,1);
gmm_curr.fgGaussianWeight = zeros(K,1);

for k=1:K
    %%% Get the k Gaussian weights for Background & Forground 
    gmm_curr.bgGaussianWeight(k) = nnz(bgClusterIds==k)/numBGValues;
    gmm_curr.fgGaussianWeight(k) = nnz(fgClusterIds==k)/numFGValues;
    checkSumBG = checkSumBG + gmm_curr.bgGaussianWeight(k);
    checkSumFG = checkSumFG + gmm_curr.fgGaussianWeight(k);

    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    bgDist = imageValues - repmat(bgMean(k,:),size(imageValues,1),1);
    fgDist = imageValues - repmat(fgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    gmm_curr.bgCovarianceMat{k} = cov(bgValues(bgClusterIds==k,:));
    gmm_curr.fgCovarianceMat{k} = cov(fgValues(fgClusterIds==k,:));
    gmm_curr.bgCovarianceMatDet(k) = det(gmm_curr.bgCovarianceMat{k});
    gmm_curr.fgCovarianceMatDet(k) = det(gmm_curr.fgCovarianceMat{k});
    gmm_curr.bgCovarianceMatInv{k} = inv(gmm_curr.bgCovarianceMat{k});
    gmm_curr.fgCovarianceMatInv{k} = inv(gmm_curr.fgCovarianceMat{k});
    allBGLogPL(:,k) = -log(gmm_curr.bgGaussianWeight(k))+0.5*log(gmm_curr.bgCovarianceMatDet(k)) + ...
        0.5*sum( (bgDist*gmm_curr.bgCovarianceMatInv{k}).*bgDist, 2 );
    allFGLogPL(:,k) = -log(gmm_curr.fgGaussianWeight(k))+0.5*log(gmm_curr.fgCovarianceMatDet(k)) + ...
        0.5*sum( (fgDist*gmm_curr.fgCovarianceMatInv{k}).*fgDist, 2 );
end

%     fprintf('MRF-GMM-unary-iter: %.3f\n',toc(stt_tic));
%     stt_tic = tic;

% for gmm_id = 1:10
%     for k=1:K
%         %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
%         bgDist = imageValues - repmat(gmm_hist{gmm_id}.bgMean(k,:),size(imageValues,1),1);
%         fgDist = imageValues - repmat(gmm_hist{gmm_id}.fgMean(k,:),size(imageValues,1),1);
% 
%         %%% Calculate the gaussian covariance matrix & use it to calculate
%         %%% all of the pixels likelihood to it :
%         allBGLogPLhist(:,k+(gmm_id-1)*K) = -log(gmm_hist{gmm_id}.bgGaussianWeight(k))+0.5*log(gmm_hist{gmm_id}.bgCovarianceMatDet(k)) ...
%             + 0.5*sum( (bgDist*gmm_hist{gmm_id}.bgCovarianceMatInv{k}).*bgDist, 2 );
%         allFGLogPLhist(:,k+(gmm_id-1)*K) = -log(gmm_hist{gmm_id}.fgGaussianWeight(k))+0.5*log(gmm_hist{gmm_id}.fgCovarianceMatDet(k)) ...
%             + 0.5*sum( (fgDist*gmm_hist{gmm_id}.fgCovarianceMatInv{k}).*fgDist, 2 );
%     end
% end

for k=1:K
    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    bgDist = imageValues - repmat(gmm_init.bgMean(k,:),size(imageValues,1),1);
    fgDist = imageValues - repmat(gmm_init.fgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    allBGLogPLinit(:,k) = -log(gmm_init.bgGaussianWeight(k))+0.5*log(gmm_init.bgCovarianceMatDet(k)) ...
        + 0.5*sum( (bgDist*gmm_init.bgCovarianceMatInv{k}).*bgDist, 2 );
    allFGLogPLinit(:,k) = -log(gmm_init.fgGaussianWeight(k))+0.5*log(gmm_init.fgCovarianceMatDet(k)) ...
        + 0.5*sum( (fgDist*gmm_init.fgCovarianceMatInv{k}).*fgDist, 2 );
end


% for k=1:K
%     %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
%     bgDist = imageValues - repmat(gmm_prev.bgMean(k,:),size(imageValues,1),1);
%     fgDist = imageValues - repmat(gmm_prev.fgMean(k,:),size(imageValues,1),1);
% 
%     %%% Calculate the gaussian covariance matrix & use it to calculate
%     %%% all of the pixels likelihood to it :
%     allBGLogPLprev(:,k) = -log(gmm_prev.bgGaussianWeight(k))+0.5*log(gmm_prev.bgCovarianceMatDet(k)) ...
%         + 0.5*sum( (bgDist/gmm_prev.bgCovarianceMat{k}).*bgDist, 2 );
%     allFGLogPLprev(:,k) = -log(gmm_prev.fgGaussianWeight(k))+0.5*log(gmm_prev.fgCovarianceMatDet(k)) ...
%         + 0.5*sum( (fgDist/gmm_prev.fgCovarianceMat{k}).*fgDist, 2 );
% end






bgLogPLinit = reshape(min(allBGLogPLinit, [], 2),size(im,1), size(im,2));
fgLogPLinit = reshape(min(allFGLogPLinit, [], 2),size(im,1), size(im,2));
% bgLogPLprev = reshape(min(allBGLogPLprev, [], 2),size(im,1), size(im,2));
% fgLogPLprev = reshape(min(allFGLogPLprev, [], 2),size(im,1), size(im,2));
bgLogPLiter = reshape(min(allBGLogPL, [], 2),size(im,1), size(im,2));
fgLogPLiter = reshape(min(allFGLogPL, [], 2),size(im,1), size(im,2));

% bgLogPLhist = zeros(size(im,1),size(im,2));
% fgLogPLhist = zeros(size(im,1),size(im,2));
% for gmm_id = 5:5%10
%     bgLogPLhist = bgLogPLhist + reshape(min(allBGLogPLhist(:,1+K*(gmm_id-1):K*(gmm_id)), [], 2),size(im,1), size(im,2));
%     fgLogPLhist = fgLogPLhist + reshape(min(allFGLogPLhist(:,1+K*(gmm_id-1):K*(gmm_id)), [], 2),size(im,1), size(im,2));
% end

% bgLogPLhist = bgLogPLhist/10;
% fgLogPLhist = fgLogPLhist/10;

% bgLogPL = ( bgLogPLinit + bgLogPLprev + bgLogPLiter ) / 3;
% fgLogPL = ( fgLogPLinit + fgLogPLprev + fgLogPLiter ) / 3;

% bgLogPL = ( bgLogPLprev + bgLogPLiter ) / 2;
% fgLogPL = ( fgLogPLprev + fgLogPLiter ) / 2;

% bgLogPL = bgLogPLiter;
% fgLogPL = fgLogPLiter;

% bgLogPL = 0.5*bgLogPLiter + 0.5*bgLogPLhist;
% fgLogPL = 0.5*fgLogPLiter + 0.5*fgLogPLhist;

bgLogPL = 0.5*bgLogPLinit + 0.5*bgLogPLiter;
fgLogPL = 0.5*fgLogPLinit + 0.5*fgLogPLiter;

%     fprintf('MRF-GMM-unary-init: %.3f\n',toc(stt_tic));
%     stt_tic = tic;
%%% Last, as seen in the GrabCut paper, take the minimum Log likelihood
%%% (    argmin(Dn)    )
% bgLogPL = reshape(min(allBGLogPL, [], 2),size(im,1), size(im,2));
% fgLogPL = reshape(min(allFGLogPL, [], 2),size(im,1), size(im,2));


end