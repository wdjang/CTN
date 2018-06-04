% GNU licence:
% Copyright (C) 2012  Itay Blumenthal
% 
%     This program is free software; you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation; either version 2 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program; if not, write to the Free Software
%     Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USAfunction finalLabel = GCAlgo( im, fixedBG,  K, G, maxIterations, Beta, diffThreshold, myHandle )

function [finalLabel, gmm_curr] = a_GCAlgo( im, fixedFG, fixedBG, init_seg, param, G, maxIterations, Beta, diffThreshold, ...
    hCf, vCf, gmm_prev, gmm_init, gmm_history )

% fgK = param.k;
% bgK = param.k;
fgK = param.fgk;
bgK = param.bgk;

[o_h, o_w, ~] = size(im);
im = imresize(im, 0.5);
fixedBG = imresize(fixedBG, 0.5);
fixedBG = fixedBG > 0.5;
fixedFG = imresize(fixedFG, 0.5);
fixedFG = fixedFG > 0.5;
init_seg = imresize(init_seg, 0.5);
init_seg = double(init_seg > 0.5);
    

%%%%%%%%%%%%%%%%%%%%%
%%% Get definite labels defining absolute Background :
prevLabel = double(1-init_seg);

%%%%%%%%%%%%%%%%%%%%%
%%% Calculate the smoothness term defined by the entire image's RGB values
bNormGrad = true;

%%% Get the image gradient
gradH = im(:,2:end,:) - im(:,1:end-1,:);
gradV = im(2:end,:,:) - im(1:end-1,:,:);

gradH = sum(gradH.^2, 3);
gradV = sum(gradV.^2, 3);

%%% Use the gradient to calculate the graph's inter-pixels weights
if ( bNormGrad )
    hC = exp(-Beta.*gradH./mean(gradH(:)));
    vC = exp(-Beta.*gradV./mean(gradV(:)));
else
    hC = exp(-Beta.*gradH);
    vC = exp(-Beta.*gradV);
end

%%% These matrices will evantually use as inputs to Bagon's code
hC = [hC zeros(size(hC,1),1)];
vC = [vC ;zeros(1, size(vC,2))];
sc = [0 G;G 0];

% hC = ( hC + [ct_map(:,2:end), zeros(size(hC,1),1)] + hCf ) / 3;
% vC = ( vC + [ct_map(2:end,:); zeros(1,size(vC,2))] + vCf ) / 3;

hC = ( hC + hCf ) / 2;
vC = ( vC + vCf ) / 2;

%     fprintf('MRF-pairwise: %.3f\n',toc(stt_tic));
    
currLabel = prevLabel;
    
%%%%%%%%%%%%%%%%%%%%%
%%% Start the EM iterations :
% bgMean = gmm_prev.bgMean;
% fgMean = gmm_prev.fgMean;
bgMean = [];
fgMean = [];
for iter=1:maxIterations
%     stt_tic = tic;
    bgIds   = find(prevLabel == 1);
    fgIds   = find(prevLabel == 0);
    
    %%% Use NOT FIXED labels to get the Log Probability Likelihood 
    %%% of the pixels to a GMM color model (inferred from the labels...)
    bgMeanInit = bgMean;
    fgMeanInit = fgMean;
    if numel(fgIds)<fgK || numel(bgIds)<bgK
        gmm_curr = gmm_prev;
        break;
    end
    [bgLogPL, fgLogPL, bgMean, fgMean, gmm_curr ] =  CalcLogPLikelihood(im, fgK, bgK, bgIds,fgIds, bgMeanInit, fgMeanInit, gmm_history, gmm_init);
    
%     fprintf('MRF-GMM: %.3f\n',toc(stt_tic));
%     stt_tic = tic;
    %%% Use our A-Priori knowledge of Background labels & set the Forground
    %%% weights according to it.
    fgLogPL(fixedBG) = max(max(fgLogPL));
    bgLogPL(fixedFG) = max(max(bgLogPL));
        
    %%% Now that we have all inputs, calculate the min-cut of the graph
    %%% using Bagon's code. Not much to explain here, for more details read
    %%% the graph cut documentation in the   GraphCut.m    file.
    dc = cat(3, bgLogPL, fgLogPL);
    graphHandle = GraphCut('open', dc , sc, vC, hC);
    graphHandle = GraphCut('set', graphHandle, int32(prevLabel == 0));
    [graphHandle, currLabel] = GraphCut('expand', graphHandle);
    currLabel = 1 - currLabel;
    GraphCut('close', graphHandle);
    
%     fprintf('MRF-GraphCut: %.3f\n',toc(stt_tic));
    %%% Break if the current result is somewhat similar to the previous result
    if nnz(prevLabel(:)~=currLabel(:)) < diffThreshold*numel(currLabel)
        break;
    end
    
    prevLabel = currLabel;
%     imagesc(currLabel);
%     input('wait key');
        
end
finalLabel = double(imresize(currLabel,[o_h, o_w])>0.5);














function [ bgLogPL, fgLogPL, bgMean, fgMean, gmm_curr ] = CalcLogPLikelihood(im, fgK, bgK, bgIds,fgIds , bgMeanInit, fgMeanInit, ...
    gmm_prev, gmm_init)

numPixels = size(im,1) * size(im,2);
allBGLogPL = zeros(numPixels,bgK);
allFGLogPL = zeros(numPixels,fgK);
allBGLogPLprev = zeros(numPixels,bgK);
allFGLogPLprev = zeros(numPixels,fgK);
allBGLogPLinit = zeros(numPixels,bgK);
allFGLogPLinit = zeros(numPixels,fgK);
% allBGLogPLhist = zeros(numPixels,K*10);
% allFGLogPLhist = zeros(numPixels,K*10);

%%% Seperate color channels 
R = im(:,:,1);
G = im(:,:,2);
B = im(:,:,3);
% [X, Y] = meshgrid(1:size(im,2),1:size(im,1));
% X = X / 2;
% Y = Y / 2;

%%% Prepare the color datasets according to the input labels 
% imageValuesIter = [R(:) G(:) B(:), X(:), Y(:)];
% bgValuesIter = [R(bgIds)    G(bgIds)     B(bgIds)   X(bgIds) Y(bgIds)];
% fgValuesIter = [R(fgIds)    G(fgIds)     B(fgIds)   X(fgIds) Y(fgIds)];

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
    [bgClusterIds, bgMean] = kmeans_norand(bgValues, bgK, 'start', bgMeanInit,  'emptyaction','singleton' ,'Options',opts);
    [fgClusterIds, fgMean] = kmeans_norand(fgValues, fgK, 'start', fgMeanInit,  'emptyaction','singleton', 'Options',opts);
else
    [bgClusterIds, bgMean] = kmeans_norand(bgValues, bgK, 'emptyaction','singleton' ,'Options',opts);
    [fgClusterIds, fgMean] = kmeans_norand(fgValues, fgK, 'emptyaction','singleton', 'Options',opts);
end

%     fprintf('MRF-GMM-kmeans: %.3f\n',toc(stt_tic));
%     stt_tic = tic;
checkSumFG = 0;
checkSumBG = 0;


gmm_curr.bgMean = bgMean;
gmm_curr.fgMean = fgMean;

gmm_curr.bgCovarianceMat = cell(bgK,1);
gmm_curr.fgCovarianceMat = cell(fgK,1);

gmm_curr.bgCovarianceMatDet = zeros(bgK,1);
gmm_curr.fgCovarianceMatDet = zeros(fgK,1);

gmm_curr.bgGaussianWeight = zeros(bgK,1);
gmm_curr.fgGaussianWeight = zeros(fgK,1);

for k=1:bgK
    %%% Get the k Gaussian weights for Background & Forground 
    gmm_curr.bgGaussianWeight(k) = nnz(bgClusterIds==k)/numBGValues;
    checkSumBG = checkSumBG + gmm_curr.bgGaussianWeight(k);

    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    bgDist = imageValues - repmat(bgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    gmm_curr.bgCovarianceMat{k} = cov(bgValues(bgClusterIds==k,:));
    gmm_curr.bgCovarianceMatDet(k) = det(gmm_curr.bgCovarianceMat{k});
    gmm_curr.bgCovarianceMatInv{k} = inv(gmm_curr.bgCovarianceMat{k});
    allBGLogPL(:,k) = -log(gmm_curr.bgGaussianWeight(k))+0.5*log(gmm_curr.bgCovarianceMatDet(k)) + ...
        0.5*sum( (bgDist*gmm_curr.bgCovarianceMatInv{k}).*bgDist, 2 );
end

for k=1:fgK
    %%% Get the k Gaussian weights for Background & Forground 
    gmm_curr.fgGaussianWeight(k) = nnz(fgClusterIds==k)/numFGValues;
    checkSumFG = checkSumFG + gmm_curr.fgGaussianWeight(k);

    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    fgDist = imageValues - repmat(fgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    gmm_curr.fgCovarianceMat{k} = cov(fgValues(fgClusterIds==k,:));
    gmm_curr.fgCovarianceMatDet(k) = det(gmm_curr.fgCovarianceMat{k});
    gmm_curr.fgCovarianceMatInv{k} = inv(gmm_curr.fgCovarianceMat{k});
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

for k=1:bgK
    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    bgDist = imageValues - repmat(gmm_init.bgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    allBGLogPLinit(:,k) = -log(gmm_init.bgGaussianWeight(k))+0.5*log(gmm_init.bgCovarianceMatDet(k)) ...
        + 0.5*sum( (bgDist*gmm_init.bgCovarianceMatInv{k}).*bgDist, 2 );
end

for k=1:fgK
    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    fgDist = imageValues - repmat(gmm_init.fgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    allFGLogPLinit(:,k) = -log(gmm_init.fgGaussianWeight(k))+0.5*log(gmm_init.fgCovarianceMatDet(k)) ...
        + 0.5*sum( (fgDist*gmm_init.fgCovarianceMatInv{k}).*fgDist, 2 );
end


for k=1:bgK
    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    bgDist = imageValues - repmat(gmm_prev.bgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    allBGLogPLprev(:,k) = -log(gmm_prev.bgGaussianWeight(k))+0.5*log(gmm_prev.bgCovarianceMatDet(k)) ...
        + 0.5*sum( (bgDist*gmm_prev.bgCovarianceMatInv{k}).*bgDist, 2 );
end

for k=1:fgK
    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    fgDist = imageValues - repmat(gmm_prev.fgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    allFGLogPLprev(:,k) = -log(gmm_prev.fgGaussianWeight(k))+0.5*log(gmm_prev.fgCovarianceMatDet(k)) ...
        + 0.5*sum( (fgDist*gmm_prev.fgCovarianceMatInv{k}).*fgDist, 2 );
end






bgLogPLinit = reshape(min(allBGLogPLinit, [], 2),size(im,1), size(im,2));
fgLogPLinit = reshape(min(allFGLogPLinit, [], 2),size(im,1), size(im,2));
bgLogPLprev = reshape(min(allBGLogPLprev, [], 2),size(im,1), size(im,2));
fgLogPLprev = reshape(min(allFGLogPLprev, [], 2),size(im,1), size(im,2));
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

bgLogPL = ( bgLogPLinit + bgLogPLprev + bgLogPLiter ) / 3;
fgLogPL = ( fgLogPLinit + fgLogPLprev + fgLogPLiter ) / 3;

% bgLogPL = ( bgLogPLinit + bgLogPLiter ) / 2;
% fgLogPL = ( fgLogPLinit + fgLogPLiter ) / 2;

% bgLogPL = bgLogPLiter;
% fgLogPL = fgLogPLiter;

% bgLogPL = 0.5*bgLogPLiter + 0.5*bgLogPLhist;
% fgLogPL = 0.5*fgLogPLiter + 0.5*fgLogPLhist;

% bgLogPL = 0.5*bgLogPLinit + 0.5*bgLogPLiter;
% fgLogPL = 0.5*fgLogPLinit + 0.5*fgLogPLiter;

%     fprintf('MRF-GMM-unary-init: %.3f\n',toc(stt_tic));
%     stt_tic = tic;
%%% Last, as seen in the GrabCut paper, take the minimum Log likelihood
%%% (    argmin(Dn)    )
% bgLogPL = reshape(min(allBGLogPL, [], 2),size(im,1), size(im,2));
% fgLogPL = reshape(min(allFGLogPL, [], 2),size(im,1), size(im,2));


