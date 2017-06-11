%% Load and init data
clear all;
imagesPaths = cell(1,5);
imagesPaths{1} = '1.gif';
imagesPaths{2} = '2.gif';
imagesPaths{3} = '3.gif';
imagesPaths{4} = '4.gif';
imagesPaths{5} = '5.gif';

% Define parameters
etaMean = 0.0001;
etaCov = 0.0001;
numOfImagesToPredict = 12;
numOfEigenVectors = 144;
choosedEigenVectors = 36;
PatchSize = [12 12];
patchLength = PatchSize(1) * PatchSize(2);
numOfPatches = 2000000;
numOfPatchesForStat = 100000;
icaBatchSize = 4000000;

% Get images from disc
Images = getImagesFromPath(imagesPaths);

% Learn the mean using quadric error
[meanImageLearned, meanImageReal] = learnMeanOFImagePatch(PatchSize, Images, numOfPatchesForStat,etaMean);
meanImageLearned  = reshape(meanImageLearned, PatchSize(1) * PatchSize(2), 1);
meanImageReal = reshape(meanImageReal, PatchSize(1) * PatchSize(2), 1);

% Calculate mean quadric error
errorMean = sum(bsxfun(@power,meanImageLearned - meanImageReal, 2))

% Learn covariance matrix using quadric error
[covarianceMatLearned, covMatrixReal] = learnCovarianceMatrix(meanImageLearned, Images, numOfPatchesForStat, etaCov, PatchSize);

% Plot learned and real covariance matrix
figure();
subplot(2,1,1), imshow(covarianceMatLearned,[]);
title('Learned covariance matrix');
colorbar;
subplot(2,1,2), imshow(covMatrixReal, []);
colorbar;
title('Real covariance matrix');

% Calculate quadric error of covariance matrix
errorCov = sum(sum(bsxfun(@power,covarianceMatLearned - covMatrixReal, 2)))
%% Get  Eigen vectors using sanger rule
     
% Get the eigen vectors and eigen values from matlab built in functions for
% validation
[realEigVec, realEigVal]  = svd(covMatrixReal);
    
% Run neural network with sanger update rule & get the eigen vectors
eigenVectors = sangeLearningPCA(Images, numOfPatches, meanImageLearned, numOfEigenVectors, PatchSize);

% Compute eigen values based in the eigen vectors that we found
eigenValues = diag(eigenVectors' * covarianceMatLearned * eigenVectors);

% Calculate error metrics for rigrn values and eigen vectors
EigenCorr = corr2(eigenValues, diag(realEigVal))
errorEigenVal = sum(bsxfun(@power, eigenValues - diag(realEigVal), 2))
errorEigenVec = sum(sum(bsxfun(@power,realEigVec - eigenVectors, 2)))

% Plot eigen vectors that we get
figure();
for i = 1:choosedEigenVectors
    img1 = reshape(eigenVectors(:,i), PatchSize(1),PatchSize(2));
    img2 = reshape(realEigVec(:,i), PatchSize(1),PatchSize(2));
    subplot(12,6,(i - 1) * 2 + 1), imshow(img1, []);
    title(['Learned  ' num2str(i)]);
    subplot(12,6,(i - 1) * 2 +  2), imshow(img2, []);
    title(['Real ' num2str(i)]);
end

% cumulative sum of eigen values
eigenSum  = sum(eigenValues);

% normalize the eigen values
normEigen = cumsum(eigenValues / eigenSum);
normEigen = normEigen * 100.0;

% Plot  - explained variance
figure();
plot(normEigen);
title('Explained variance');
xlabel('Eigen vector index');
ylabel('Explained variance %');
text(101, 85, '85% variance explained');

% Predict images using the eigenvectors that we learn

% Run pca with the eigen values that we learned on random images
[originalImages, predictedImages] = runPCA(Images, eigenVectors(:,1:choosedEigenVectors), meanImageLearned, numOfImagesToPredict, PatchSize);

% Plot the original and reconstructed image
figure();
for i = 1:numOfImagesToPredict
    predicted = reshape(predictedImages(:,i), PatchSize(1), PatchSize(2));
    original = reshape(originalImages(:,i), PatchSize(1), PatchSize(2));
    subplot(numOfImagesToPredict / 2,4, 2 * (i - 1) + 1), imshow(predicted, []);
    title(['Predicted ' num2str(i)]);
    subplot(numOfImagesToPredict / 2,4, 2 * (i - 1) + 2), imshow(original, []);
    title(['Original ' num2str(i)]);
end

%% ZCA Whitening
epsilon = 0;

% Calculate Whitening matrix for ZCA and PCA
[wZCA, wPCA] = ZCAPCAWhitening(eigenValues, eigenVectors,epsilon);
figure();

%Plot some of the filters that w e get from ZCA and PCA
for i = 1:choosedEigenVectors
    img1 = reshape(wPCA(i,:), PatchSize(1),PatchSize(2));
    img2 = reshape(wZCA(i, :), PatchSize(1),PatchSize(2));
    subplot(12,6,(i - 1) * 2 + 1), imshow(img1, []);
    title(['PCA '  num2str(i)]);
    subplot(12,6,(i - 1) * 2 + 2), imshow(img2, []);
    title(['ZCA '  num2str(i)]);
end

% Run ZCA/PCA Whitening on random images
[original, pcaImages, zcaImages] = runWhitening(wZCA, wPCA', Images, numOfImagesToPredict, PatchSize,eigenVectors, meanImageLearned);

% plot the images after Whitening
figure();
for i = 1:numOfImagesToPredict
    pcaPredicted = reshape(pcaImages(:,i), PatchSize(1), PatchSize(2));
    zcaPredicted = reshape(zcaImages(:,i), PatchSize(1), PatchSize(2));
    originalPhoto = reshape(original(:,i), PatchSize(1), PatchSize(2));
    subplot(numOfImagesToPredict,3, 3 * (i - 1) + 1), imshow(pcaPredicted, []);
    title(['PCA ' num2str(i)]);
    subplot(numOfImagesToPredict,3, 3 * (i - 1) + 2), imshow(zcaPredicted, []);
    title(['ZCA ' num2str(i)]);
    subplot(numOfImagesToPredict,3, 3 * (i - 1) + 3), imshow(originalPhoto, []);
    title(['original ' num2str(i)]);
end

%% ICA 
wICA = learnICA(Images, meanImageLearned, PatchSize, icaBatchSize, wZCA);

% Plot some of the filters that we get from ZCA and PCA
indexes = randperm(numOfEigenVectors);
indexes = indexes(1:choosedEigenVectors);
ncount = 1;

figure();
for i = indexes
    img = reshape(wICA(i,:), PatchSize(1),PatchSize(2));
    subplot(6,6,ncount), imshow(img, []);
    title(num2str(i));
    ncount = ncount + 1;
end

% Calulate zca ica transformation
zcaICATransformation = wZCA * wICA;

% plot the filters
figure();
ncount = 1;
for i = indexes
    img = reshape(zcaICATransformation(i,:), PatchSize(1),PatchSize(2));
    subplot(6,6,ncount), imshow(img, []);
    title(num2str(i));
    ncount = ncount + 1;
end

%%
% Calulate zca ica transformation
invzcaICATransformation = inv(zcaICATransformation);

% plot the filters
figure();
ncount = 1;
for i = indexes
    img = reshape(invzcaICATransformation(:,i), PatchSize(1),PatchSize(2));
    subplot(6,6,ncount), imshow(img, []);
    title(num2str(i));
    ncount = ncount + 1;
end