%% Load and init data
imagesPaths = cell(1,5);
imagesPaths{1} = '1.gif';
imagesPaths{2} = '2.gif';
imagesPaths{3} = '3.gif';
imagesPaths{4} = '4.gif';
imagesPaths{5} = '5.gif';

% Define parameters
etaMean = 0.001;
etaCov = 0.0001;
etaSanger = 0.01;
numOfImagesToPredict = 12;
numOfEigenVectors = 36;
PatchSize = [12 12];
numOfPatches = 100000;
numOfPatchesForStat = 100000;

% Get images from disc
Images = getImagesFromPath(imagesPaths);

% Learn the mean using quadric error
meanImageLearned = learnMeanOFImagePatch(PatchSize, Images, numOfPatchesForStat,etaMean);
meanImageLearned  = reshape(meanImageLearned, 1, PatchSize(1) * PatchSize(2));

% Learn covariance matrix using quadric error
covarianceMat = learnCovarianceMatrix(meanImageLearned, Images, numOfPatchesForStat, etaCov, PatchSize);

% Run neural network with sanger update rule & get the eigen vectors
eigenVectors = sangeLearningPCA(Images, numOfPatches, meanImageLearned, numOfEigenVectors, PatchSize, etaSanger);

% Plot eigen vectors that we get(with mean)
figure();
eigenValues = zeros(1, numOfEigenVectors);
for i = 1:numOfEigenVectors
    eighenImage(i).img = reshape(eigenVectors(i,:), PatchSize(1),PatchSize(2)) + reshape(meanImageLearned, PatchSize(1),  PatchSize(2));
    subplot(6,6,i), imshow(eighenImage(i).img,PatchSize);
    title(num2str(i));
end
eigenValues = diag(eigenVectors * covarianceMat * eigenVectors');

% cumulative sum of eigen values
eigenSum  = sum(eigenValues);

% normalize the eigen values
normEigen = cumsum(eigenValues / eigenSum);
normEigen = normEigen * 100.0;

% Plot explained variance
figure();
plot(normEigen);
title('Explained variance');
xlabel('Eigen vector index');
ylabel('Explained variance %');
text(20, 85, '85% variance explained');

% Run pca with the eigen values that we learned on random images
[originalImages, predictedImages] = runPCA(Images, eigenVectors, meanImageLearned, numOfImagesToPredict, PatchSize);

% Plot the original and reconstructed image
figure();
for i = 1:numOfImagesToPredict
    predicted = reshape(predictedImages(i,:), PatchSize(1), PatchSize(2));
    original = reshape(originalImages(i,:), PatchSize(1), PatchSize(2));
    subplot(numOfImagesToPredict / 2,4, 2 * (i - 1) + 1), imshow(predicted, PatchSize);
    title(['Predicted ' num2str(i)]);
    subplot(numOfImagesToPredict / 2,4, 2 * (i - 1) + 2), imshow(original, PatchSize);
    title(['Original ' num2str(i)]);
end

%% Build Whitening for ZCA and PCA

% Run neural network with sanger update rule & get the eigen vectors
fullEigenVectors = sangeLearningPCA(Images, numOfPatches, meanImageLearned, PatchSize(1) * PatchSize(2), PatchSize, etaSanger);
fullEigenValues = diag(fullEigenVectors * covarianceMat * fullEigenVectors');
fullEigenCorr = corr2(fliplr(fullEigenValues')', eig(covarianceMat))
%%
epsilon = 0.001;

% Calculate Whitening matrix for ZCA and PCA
[wZCA, wPCA] = ZCAPCAWhitening(fullEigenValues, fullEigenVectors,epsilon);
figure();

% Plot some of the filters that we get from ZCA and PCA
for i = 1:numOfEigenVectors
    pcaImage(i).img = reshape(wPCA(i,:), PatchSize(1),PatchSize(2));
    zcaImage(i).img = reshape(wZCA(i,:), PatchSize(1),PatchSize(2));
    subplot(12,6,(i - 1) * 2 + 1), imshow(pcaImage(i).img, PatchSize);
    title(['PCA '  num2str(i)]);
    subplot(12,6,(i - 1) * 2 + 2), imshow(zcaImage(i).img, PatchSize);
    title(['ZCA '  num2str(i)]);
end
%% Run ZCA/PCA Whitening on random images
[original, pcaImages, zcaImages] = runWhitening(wZCA, wPCA, Images, numOfImagesToPredict, PatchSize,eigenVectors);

% plot the images after Whitening
figure();
for i = 1:numOfImagesToPredict
    pcaPredicted = reshape(pcaImages(i,:), PatchSize(1), PatchSize(2));
    zcaPredicted = reshape(zcaImages(i,:), PatchSize(1), PatchSize(2));
    originalPhoto = reshape(original(i,:), PatchSize(1), PatchSize(2));
    subplot(numOfImagesToPredict,3, 3 * (i - 1) + 1), imshow(pcaPredicted, PatchSize);
    title(['PCA ' num2str(i)]);
    subplot(numOfImagesToPredict,3, 3 * (i - 1) + 2), imshow(zcaPredicted, PatchSize);
    title(['ZCA ' num2str(i)]);
    subplot(numOfImagesToPredict,3, 3 * (i - 1) + 3), imshow(originalPhoto, PatchSize);
    title(['original ' num2str(i)]);
end