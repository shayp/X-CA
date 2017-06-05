function eigenVectors = sangeLearningPCA(Images, numOfImages, meanOfPatch, numOfEigenVectors, PatchSize, eta)

% choose random weightes for init
eigenVectors = 0.1 * rand(numOfEigenVectors, length(meanOfPatch));

% define deltaW matrix
deltaW = zeros(numOfEigenVectors, length(meanOfPatch));

% Run for X images
for i = 1:numOfImages

    reconstructMatrix = zeros(numOfEigenVectors, length(meanOfPatch));
    
    % Choose random patch
    currentImage = randi(length(Images));
    xRand = randi(size(Images(currentImage).data,1) - PatchSize(1));
    yRand = randi(size(Images(currentImage).data,2) - PatchSize(2));
    currentPatch = double(Images(currentImage).data(xRand:xRand + PatchSize(1) - 1, yRand:yRand + PatchSize(2) - 1));
    currentPatch = reshape(currentPatch, 1, length(meanOfPatch));
    
    % x is input image after removing mean
    x = currentPatch - meanOfPatch;
    
    % Caclulate the values of y, y = x * w
    y = x * eigenVectors';

    % Run for each weighet in the network 
    for neuronIndex = 1:numOfEigenVectors
        for wIndex = 1:length(meanOfPatch)
            
            % if the current eigen vector is not 1
            % yw(prev) = sum(yw(wIndex)) (until now)
            if neuronIndex ~= 1
                reconstructMatrix(neuronIndex, wIndex) = reconstructMatrix(neuronIndex - 1, wIndex) + y(neuronIndex - 1) * eigenVectors(neuronIndex - 1, wIndex);
            end
            % deltaW = eta * y * (x - yw(prev) - yw(current)) 
            deltaW(neuronIndex, wIndex) = eta * y(neuronIndex) *((x(wIndex) - reconstructMatrix(neuronIndex, wIndex)) - y(neuronIndex) * eigenVectors(neuronIndex, wIndex));
        end
    end
    
    % Update network
    eigenVectors = eigenVectors + deltaW;
end
end