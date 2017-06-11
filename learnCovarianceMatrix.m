function [covarianceMat,realCovMat] = learnCovarianceMatrix(meanOFImages, Images, numOfPatches, eta, PatchSize)

covarianceMat = zeros(length(meanOFImages), length(meanOFImages));
realCovMat = zeros(length(meanOFImages), length(meanOFImages));

% Run for X random images
for i = 1:numOfPatches
    % Get random patch
    currentImage = randi(length(Images));
    xRand = randi(size(Images(currentImage).data,1) - PatchSize(1));
    yRand = randi(size(Images(currentImage).data,2) - PatchSize(2));
    currentPatch = double(Images(currentImage).data(xRand:xRand + PatchSize(1) - 1, yRand:yRand + PatchSize(2) - 1));
    currentPatch = reshape(currentPatch, length(meanOFImages), 1);
    varPatch = currentPatch - meanOFImages;
    
    % Calculate current patch covariance
    currentCov = varPatch * varPatch';
    
    % Error between learned covariance to current patch covariance
    errorForPatch = currentCov - covarianceMat;
    
    % Change covariance using update rule 
    covarianceMat = covarianceMat + eta * errorForPatch;
    
    % Caclulate real covariance for validation
    realCovMat = realCovMat + currentCov;
end

% Validate covariance matrix 
realCovMat = realCovMat / numOfPatches;
corr2(covarianceMat, realCovMat)
end