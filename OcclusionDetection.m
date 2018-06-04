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

function [ bgLogPL, fgLogPL ]  = OcclusionDetection( im, bgK, fgK, gmm_info, gmm_prev )

numPixels = size(im,1) * size(im,2);
allBGLogPLinit = zeros(numPixels,bgK);
allFGLogPLinit = zeros(numPixels,fgK);
allBGLogPLprev = zeros(numPixels,bgK);
allFGLogPLprev = zeros(numPixels,fgK);

%%% Seperate color channels 
R = im(:,:,1);
G = im(:,:,2);
B = im(:,:,3);

%%% Prepare the color datasets according to the input labels 
imageValues = [R(:) G(:) B(:)];

for k=1:bgK
    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    bgDist = imageValues - repmat(gmm_info.bgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    allBGLogPLinit(:,k) = -log(gmm_info.bgGaussianWeight(k))+0.5*log(gmm_info.bgCovarianceMatDet(k)) ...
        + 0.5*sum( (bgDist*gmm_info.bgCovarianceMatInv{k}).*bgDist, 2 );
end
bgLogPLinit = reshape(min(allBGLogPLinit, [], 2),size(im,1), size(im,2));

for k=1:fgK
    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    fgDist = imageValues - repmat(gmm_info.fgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    allFGLogPLinit(:,k) = -log(gmm_info.fgGaussianWeight(k))+0.5*log(gmm_info.fgCovarianceMatDet(k)) ...
        + 0.5*sum( (fgDist*gmm_info.fgCovarianceMatInv{k}).*fgDist, 2 );
end
fgLogPLinit = reshape(min(allFGLogPLinit, [], 2),size(im,1), size(im,2));

for k=1:bgK
    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    bgDist = imageValues - repmat(gmm_prev.bgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    allBGLogPLprev(:,k) = -log(gmm_prev.bgGaussianWeight(k))+0.5*log(gmm_prev.bgCovarianceMatDet(k)) ...
        + 0.5*sum( (bgDist*gmm_prev.bgCovarianceMatInv{k}).*bgDist, 2 );
end
bgLogPLprev = reshape(min(allBGLogPLprev, [], 2),size(im,1), size(im,2));

for k=1:fgK
    %%% FOR ALL PIXELS - calculate the distance from the k gaussian (BG & FG)
    fgDist = imageValues - repmat(gmm_prev.fgMean(k,:),size(imageValues,1),1);

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    allFGLogPLprev(:,k) = -log(gmm_prev.fgGaussianWeight(k))+0.5*log(gmm_prev.fgCovarianceMatDet(k)) ...
        + 0.5*sum( (fgDist*gmm_prev.fgCovarianceMatInv{k}).*fgDist, 2 );
end
fgLogPLprev = reshape(min(allFGLogPLprev, [], 2),size(im,1), size(im,2));

bgLogPL = min( bgLogPLprev, bgLogPLinit );
fgLogPL = min( fgLogPLprev, fgLogPLinit );
% bgLogPL = bgLogPLinit;
% fgLogPL = fgLogPLinit;
% bgLogPL = bgLogPLprev;
% fgLogPL = fgLogPLprev;

%%% Last, as seen in the GrabCut paper, take the minimum Log likelihood
%%% (    argmin(Dn)    )
% bgLogPL = reshape(min(allBGLogPL, [], 2),size(im,1), size(im,2));
% fgLogPL = reshape(min(allFGLogPL, [], 2),size(im,1), size(im,2));


end