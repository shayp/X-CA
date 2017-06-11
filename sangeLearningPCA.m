function eigenVectors = sangeLearningPCA(Images, numOfImages, meanOfPatch, numOfEigenVectors, PatchSize)

% choose random weightes for init
eigenVectors =  0.01 * rand(length(meanOfPatch), numOfEigenVectors);
% define deltaW matrix
deltaW = zeros(length(meanOfPatch), numOfEigenVectors);
eta = [0.01 0.005 0.001];

% Run for X images
for i = 1:numOfImages
    
    if i < 0.4 * numOfImages
        currentEta = eta(1);
    elseif i < 0.7 * numOfImages
        currentEta = eta(2);
    else
        currentEta = eta(3);
    end
    
    % Get random patch
    currentImage = randi(length(Images));
    xRand = randi(size(Images(currentImage).data,1) - PatchSize(1));
    yRand = randi(size(Images(currentImage).data,2) - PatchSize(2));
    currentPatch = double(Images(currentImage).data(xRand:xRand + PatchSize(1) - 1, yRand:yRand + PatchSize(2) - 1));
    currentPatch = reshape(currentPatch, PatchSize(1) * PatchSize(2), 1);
    reconstructMatrix = zeros(length(meanOfPatch), numOfEigenVectors);
    
    % x is input image after removing mean
    x = currentPatch  - meanOfPatch;

    % Caclulate the values of y, y = w' * x
    y = eigenVectors' * x ;

    % Run for each weighet in the network 
    for neuronIndex = 1:numOfEigenVectors
        for wIndex = 1:length(meanOfPatch)

            % if the current eigen vector is not 1
            if neuronIndex ~= 1
                reconstructMatrix(wIndex, neuronIndex) = reconstructMatrix(wIndex, neuronIndex - 1) + y(neuronIndex - 1) * eigenVectors(wIndex, neuronIndex - 1);
            end
            
            % deltaW = eta * y * (x - yw(prev) - yw(current))
            deltaW(wIndex,neuronIndex) = currentEta * y(neuronIndex) *((x(wIndex) - reconstructMatrix(wIndex, neuronIndex)) - y(neuronIndex) * eigenVectors(wIndex, neuronIndex));
        end
    end
    % Update network
    eigenVectors = eigenVectors + deltaW;
    if mod(i, 20000) == 0
        sangerTrial = i
    end
end
end
