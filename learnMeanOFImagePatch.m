function [meanImage, meanImageReal] = learnMeanOFImagePatch(PatchSize, Images, numOfPatches,eta)

meanImage = zeros(PatchSize(1), PatchSize(2));
meanImageReal = zeros(PatchSize(1), PatchSize(2));
% Run for X random images
for i = 1:numOfPatches
    currentImage = randi(length(Images));
    xRand = randi(size(Images(currentImage).data,1) - PatchSize(1));
    yRand = randi(size(Images(currentImage).data,2) - PatchSize(2));
    currentPatch = double(Images(currentImage).data(xRand:xRand + PatchSize(1) - 1, yRand:yRand + PatchSize(2) - 1));
    if i == 1
        meanImage = currentPatch;
    end
    meanImageReal = meanImageReal + currentPatch;
    % calculate error between mean image and current image
    errorForPatch = currentPatch - meanImage;
    
    % Use update rule to update mean
    meanImage = meanImage + eta * errorForPatch;
end
meanImageReal = meanImageReal ./ numOfPatches;
end