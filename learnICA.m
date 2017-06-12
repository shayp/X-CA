function [wICA] = learnICA(Images, meanPatch, PatchSize, numOfImages, wZCA)
% Parameters
eta = [ 0.001 0.0005 0.0002 0.0001];
imageLength = PatchSize(1) * PatchSize(2);
imagestoShow = 20;

% Set start weights as diagonal matrix(for inverse)
wICA = eye(imageLength);
figure();

% Train ICA network
for i = 1:numOfImages
    
    % Set current eta using the current iteration
    if i <= 0.4 * numOfImages
        currentEta = eta(1);
    elseif i <= 0.6 * numOfImages
        currentEta = eta(2);
    elseif i <= 0.8 * numOfImages
        currentEta = eta(3);
    else
        currentEta = eta(4);
    end
    
    % Get random patch
    currentImage = randi(length(Images));
    xRand = randi(size(Images(currentImage).data,1) - PatchSize(1));
    yRand = randi(size(Images(currentImage).data,2) - PatchSize(2));
    currentPatch = double(Images(currentImage).data(xRand:xRand + PatchSize(1) - 1, yRand:yRand + PatchSize(2) - 1));
    currentPatch = reshape(currentPatch, imageLength, 1) - meanPatch;
    % Run zca whitening
    zcaPatch = wZCA * currentPatch;
        
    % Run ICA on zca whitened data
    hICA = wICA * zcaPatch;
        
    % Calculate y based on infomax learning rule
    yICA = 1 - 2 * sigmoidFunc(hICA);
        
    % Calculate deltaW based on infomax learning rule
    wDeltaICA = inv(wICA') + yICA * zcaPatch';
        
    % update wICA
    wICA = wICA + currentEta * wDeltaICA;
    
    if mod(i, 20000) == 0
        icaStep = i
        % plot  learned ICA vectors randomly
        randInd  = randperm(imageLength);
        randInd = randInd(1:imagestoShow);
        nCount = 1;
        for k = randInd
            img = reshape(wICA(k,:), PatchSize(1),PatchSize(2));
            subplot(5,4,nCount), imshow(mat2gray(img));
            title(num2str(k));
            drawnow;
            nCount = nCount + 1;
        end
    end
end
end