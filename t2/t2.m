clear all
clc

nOrderIterations = 1e3;
nConvergenceIterations = 1e5;

tmpData = dlmread('wine.data.txt',',');

for iColumn = 2:size(tmpData,2)
    tmpData(:,iColumn) = tmpData(:,iColumn) - mean(tmpData(:,iColumn));
    tmpData(:,iColumn) = tmpData(:,iColumn) / std(tmpData(:,iColumn));
end

inputData = tmpData(:,2:14);

nInputPoints = length(inputData);

% Initialise kohonen nodes
nKohonenPointsX = 20;
nKohonenPointsY = 20;
nKohonenPoints = nKohonenPointsX*nKohonenPointsY;
inputKohonenCoords = rand(nKohonenPoints,13);  % Positions in 13D input space
outputKohonenCoords = zeros(nKohonenPoints,2);  % Positions in 2D output space

for iX = 1:nKohonenPointsX
    for iY = 1:nKohonenPointsY
        linearIndex = (iX-1)*nKohonenPointsX + iY;
        outputKohonenCoords(linearIndex,:) = [iX, iY];
    end
end


% Start ordering phase
for iOrderIt = 1:nOrderIterations
    iRandom = ceil(rand()*nInputsPoints);
    randomPoint = inputData(iRandom,:);
    
    for iKohonenPoint = 1:nKohonenPoints
        currentNorm = norm(randomPoint - inputKohonenPoints(iKohonenPoint,:));
        if currentNorm < smallestNorm
            smallestNorm = currentNorm;
            iWinningPoint = iKohonenPoint;
        end
    end
    winningPoint = inputKohonenPoints(iWinningPoint,:);
    
    for iKohonenPoint = 1:nKohonenPoints
        
    end
    
end





