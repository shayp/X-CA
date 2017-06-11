function [wZCA, wPCA] = ZCAPCAWhitening(eigenValues, eigenVectors,epsilon)
wPCA = diag(1./sqrt(eigenValues + epsilon)) * eigenVectors';

wZCA = eigenVectors * diag(1./sqrt(eigenValues + epsilon)) * eigenVectors';
end