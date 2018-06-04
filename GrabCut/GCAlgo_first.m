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

function gmm_info = GCAlgo_first( im, init_seg, param )

%%%%%%%%%%%%%%%%%%%%%
%%% Get definite labels defining absolute Background :
% prevLabel = double(fixedBG);
prevLabel = double(1-init_seg);


%%%%%%%%%%%%%%%%%%%%%
%%% Start the EM iterations :
bgMean = [];
fgMean = [];

bgIds   = find(prevLabel == 1);
fgIds   = find(prevLabel == 0);

%%% Use NOT FIXED labels to get the Log Probability Likelihood 
%%% of the pixels to a GMM color model (inferred from the labels...)
bgMeanInit = bgMean;
fgMeanInit = fgMean;

gmm_info =  CalcLogPLikelihood(im, param, bgIds,fgIds, bgMeanInit, fgMeanInit );








function gmm_info = CalcLogPLikelihood(im, param, bgIds,fgIds , bgMeanInit, fgMeanInit )

% fgK = param.k;
% bgK = param.k;
fgK = param.fgk;
bgK = param.bgk;

%%% Seperate color channels 
R = im(:,:,1);
G = im(:,:,2);
B = im(:,:,3);

%%% Prepare the color datasets according to the input labels 
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
opts.MaxIter = 40;

if ( ~isempty(bgMeanInit) && ~isempty(fgMeanInit) )
    [bgClusterIds, bgMean] = kmeans_norand(bgValues, bgK, 'start', bgMeanInit,  'emptyaction','singleton' ,'Options',opts);
    [fgClusterIds, fgMean] = kmeans_norand(fgValues, fgK, 'start', fgMeanInit,  'emptyaction','singleton', 'Options',opts);
else
    [bgClusterIds, bgMean] = kmeans_norand(bgValues, bgK, 'emptyaction','singleton' ,'Options',opts);
    [fgClusterIds, fgMean] = kmeans_norand(fgValues, fgK, 'emptyaction','singleton', 'Options',opts);
end

gmm_info.bgMean = bgMean;
gmm_info.fgMean = fgMean;

gmm_info.bgCovarianceMat = cell(bgK,1);
gmm_info.fgCovarianceMat = cell(fgK,1);

gmm_info.bgCovarianceMatDet = zeros(bgK,1);
gmm_info.fgCovarianceMatDet = zeros(fgK,1);

gmm_info.bgGaussianWeight = zeros(bgK,1);
gmm_info.fgGaussianWeight = zeros(fgK,1);

for k=1:bgK
    %%% Get the k Gaussian weights for Background & Forground 
    gmm_info.bgGaussianWeight(k) = nnz(bgClusterIds==k)/numBGValues;

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    gmm_info.bgCovarianceMat{k} = cov(bgValues(bgClusterIds==k,:));
    gmm_info.bgCovarianceMatDet(k) = det(gmm_info.bgCovarianceMat{k});
    gmm_info.bgCovarianceMatInv{k} = inv(gmm_info.bgCovarianceMat{k});
end

for k=1:fgK
    %%% Get the k Gaussian weights for Background & Forground 
    gmm_info.fgGaussianWeight(k) = nnz(fgClusterIds==k)/numFGValues;

    %%% Calculate the gaussian covariance matrix & use it to calculate
    %%% all of the pixels likelihood to it :
    gmm_info.fgCovarianceMat{k} = cov(fgValues(fgClusterIds==k,:));
    gmm_info.fgCovarianceMatDet(k) = det(gmm_info.fgCovarianceMat{k});
    gmm_info.fgCovarianceMatInv{k} = inv(gmm_info.fgCovarianceMat{k});
end



