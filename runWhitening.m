function [originalImages, pcaImages, zcaImages] = runWhitening(wZCA, wPCA, Images, numOfImages, PatchSize,eigenVectors)

patchLength = PatchSize(1) * PatchSize(2);
zcaImages = zeros(numOfImages, patchLength);
pcaImages = zeros(numOfImages, patchLength);
originalImages = zeros(numOfImages, patchLength);

% Run for X random images
for i = 1:numOfImages
    
    % Get random patch
    currentImage = randi(length(Images));
    xRand = randi(size(Images(currentImage).data,1) - PatchSize(1));
    yRand = randi(size(Images(currentImage).data,2) - PatchSize(2));
    currentPatch = double(Images(currentImage).data(xRand:xRand + PatchSize(1) - 1, yRand:yRand + PatchSize(2) - 1));
    currentPatch = reshape(currentPatch, 1, patchLength);
    
    % multiply by ZCA filters
    zcaImages(i,:) = (wZCA * currentPatch')';

    % multiply by PCA filters
    pcaImages(i,:) = (wPCA * currentPatch')';
    originalImages(i,:) = currentPatch;
end

end