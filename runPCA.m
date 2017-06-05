function [originalImages, predictedImages] = runPCA(Images, eigenVectors, meanOfPatch, numOfImages, PatchSize)

predictedImages = zeros(numOfImages, length(meanOfPatch));
originalImages = zeros(numOfImages, length(meanOfPatch));

% Run for X random images
for i = 1:numOfImages
    currentImage = randi(length(Images));
    xRand = randi(size(Images(currentImage).data,1) - PatchSize(1));
    yRand = randi(size(Images(currentImage).data,2) - PatchSize(2));
    currentPatch = double(Images(currentImage).data(xRand:xRand + PatchSize(1) - 1, yRand:yRand + PatchSize(2) - 1));
    currentPatch = reshape(currentPatch, 1, length(meanOfPatch));
    noMeanPatch = currentPatch - meanOfPatch;
    
    % Get PCA reconstruct image and add mean
    predictedImages(i,:) = (noMeanPatch * eigenVectors') * eigenVectors + meanOfPatch;
    originalImages(i,:) = currentPatch;
end