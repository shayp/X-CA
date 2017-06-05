function [wZCA, wPCA] = ZCAPCAWhitening(eigenValues, eigenVectors,epsilon)

% wPCA = (eigVals^(-1/2)) * eigVectors'
wPCA = diag(1./sqrt(eigenValues + epsilon)) * eigenVectors;

% wZCA = eigVectors * (eigVals^(-1/2)) * eigVectors'
wZCA = eigenVectors' * diag(1./sqrt(eigenValues + epsilon)) * eigenVectors;
end