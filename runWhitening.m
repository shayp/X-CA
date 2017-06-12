function [originalImages, pcaImages, zcaImages] = runWhitening(wZCA, wPCA, Images, numOfImages, PatchSize,eigenVectors, meanImageLearned)

patchLength = PatchSize(1) * PatchSize(2);
zcaImages = zeros(patchLength,numOfImages);
pcaImages = zeros(patchLength,numOfImages);
originalImages = zeros(patchLength,numOfImages);

% Run for X random images
for i = 1:numOfImages
    
    % Get random patch
    currentImage = randi(length(Images));
    xRand = randi(size(Images(currentImage).data,1) - PatchSize(1));
    yRand = randi(size(Images(currentImage).data,2) - PatchSize(2));
    currentPatch = double(Images(currentImage).data(xRand:xRand + PatchSize(1) - 1, yRand:yRand + PatchSize(2) - 1));
    currentPatch = reshape(currentPatch, patchLength, 1) - meanImageLearned;
    
    % multiply by ZCA filters
    zcaImages(:,i) = wZCA * currentPatch;

    % multiply by PCA filters
    pcaImages(:,i) = wPCA * currentPatch;
    originalImages(:,i) = currentPatch + meanImageLearned;
end

end