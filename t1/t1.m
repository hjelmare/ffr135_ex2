clear all
clc
clf

nRandomPoints = 1000;
nKohonenPoints = 100;

nOrderIts = 1e3;
nConvIts = 5e4;

initNbhWidth = 100;     % Change between 100 and 5 for a) and b)
initLearnRate = 0.1;
tau = 200;

convNbhWidth = 0.9;
convLearnRate = 0.01;

% Generate point cloud in triangle shape

leftLine = @(x) sqrt(3)*x;
rightLine = @(x) sqrt(3)*(1-x);
plotLine = [0, 0 ; 0.5, sqrt(3/4) ; 1, 0 ];

randomPoints = zeros(nRandomPoints,2);

for i = 1:nRandomPoints
    while randomPoints(i,2) > leftLine(randomPoints(i,1)) || randomPoints(i,2) > rightLine(randomPoints(i,1)) || isequal(randomPoints(i,:),[0,0])
        randomPoints(i,:) = rand(1,2);    % Generate new random coords
    end                             % until we get a pt below the two lines
end


% Initialize Kohonen network to vertical line at x = 0.5
%kohonenPoints = [linspace(0.5,0.5,nKohonenPoints)',linspace(0,sqrt(3/4),nKohonenPoints)'];
% Let's try just random positions instead - seems to work well
kohonenPoints = rand(nKohonenPoints,2);
kohonenPoints(:,2) = kohonenPoints(:,2) * sqrt(3/4);
%kohonenPoints = randomPoints(1:nKohonenPoints,:);

kohonenPointWeights  = zeros(nKohonenPoints,1); % Used to color points by neighbourhood function value, during debugging

% % Check initalization
% kohonenColors = zeros(nKohonenPoints,3);      % Linear coloring from start to end
% kohonenColors(:,1) = linspace(0,1,nKohonenPoints)';
% kohonenColors(:,3) = linspace(1,0,nKohonenPoints)';
% scatter(kohonenPoints(:,1),kohonenPoints(:,2),10,kohonenColors)

% Set functions for neighbourhood width and learning rate
nbhWidthFunc = @(t) initNbhWidth * exp(-t/tau);
learnRateFunc = @(t) initLearnRate * exp(-t/tau);

figure(1)
% Start ordering iterations
for iOrderIt = 1:nOrderIts
    % Select a random point from our sample
    iRandomPoint = floor(rand()*nRandomPoints) + 1;
    selectedPoint = randomPoints(iRandomPoint,:);
    
    % Find the closest Kohonen point
    smallestNorm = Inf;
    for iKohonenPoint = 1:nKohonenPoints
        currentNorm = norm(kohonenPoints(iKohonenPoint,:) - randomPoints(iRandomPoint,:));
        if currentNorm < smallestNorm
            smallestNorm = currentNorm;
            iSmallestNorm = iKohonenPoint;
        end
    end
    winningPos = kohonenPoints(iSmallestNorm,:);
        
    % Move the Kohonen points
    for iKohonenPoint = 1:nKohonenPoints
        currentPos = kohonenPoints(iKohonenPoint,:);
        
        nbhWidth = nbhWidthFunc(iOrderIt);
        nbhFunction = exp(-(iSmallestNorm-iKohonenPoint)^2 / (2*nbhWidth^2));
        
        kohonenPointWeights(iKohonenPoint) = nbhFunction;
        
        learnRate = learnRateFunc(iOrderIt);
        
        newPos = currentPos + learnRate*nbhFunction*(selectedPoint-currentPos);
        
        kohonenPoints(iKohonenPoint,:) = newPos;
    end
    

%     % Plotting to see progress, disable after debugging
%     scatter(randomPoints(:,1), randomPoints(:,2),1)
%     axis equal
%     hold on
%     plot(randomPoints(iRandomPoint,1),randomPoints(iRandomPoint,2),'or')
%     kohonenColors = zeros(nKohonenPoints,3);
%     kohonenColors(:,1) = kohonenPointWeights / max(kohonenPointWeights);
%     kohonenColors(:,3) = 1 - kohonenPointWeights / max(kohonenPointWeights);
%     scatter(kohonenPoints(:,1),kohonenPoints(:,2),10,kohonenColors)
%     text(0,0.2,num2str(iOrderIt));
%     hold off
%     %pause(0.0000001)
%     %pause

    
end     % End of ordering phase

% Plotting final result
scatter(randomPoints(:,1), randomPoints(:,2),1)
axis equal
hold on
plot(kohonenPoints(:,1),kohonenPoints(:,2),'-or')
plot(plotLine(:,1),plotLine(:,2),'-k');
axis([0 1 0 1])
hold off

filename =  ['t1' num2str(initNbhWidth) 'o.png'];
saveas(gcf,filename,'png')


figure(2)
% Start convergence iterations
for iConvIt = 1:nConvIts
    % Select a random point from our sample
    iRandomPoint = floor(rand()*nRandomPoints) + 1;
    selectedPoint = randomPoints(iRandomPoint,:);
    
    % Find the closest Kohonen point
    smallestNorm = Inf;
    for iKohonenPoint = 1:nKohonenPoints
        currentNorm = norm(kohonenPoints(iKohonenPoint,:) - randomPoints(iRandomPoint,:));
        if currentNorm < smallestNorm
            smallestNorm = currentNorm;
            iSmallestNorm = iKohonenPoint;
        end
    end
    winningPos = kohonenPoints(iSmallestNorm,:);
        
    % Move the Kohonen points
    for iKohonenPoint = 1:nKohonenPoints
        currentPos = kohonenPoints(iKohonenPoint,:);
        
        nbhFunction = exp(-(iSmallestNorm-iKohonenPoint)^2 / (2*convNbhWidth^2));
        
        newPos = currentPos + convLearnRate*nbhFunction*(selectedPoint-currentPos);
        
        kohonenPoints(iKohonenPoint,:) = newPos;
    end
    
%     % Plotting to see progress, disable after debugging
%     if mod(iConvIt,100)==0
%         scatter(randomPoints(:,1), randomPoints(:,2),1)
%         axis equal
%         hold on
%         plot(randomPoints(iRandomPoint,1),randomPoints(iRandomPoint,2),'or')
%         plot(kohonenPoints(:,1),kohonenPoints(:,2),'-or')
%         text(0,0.2,num2str(iConvIt));
%         hold off
%         pause(0.0000001)
%         %pause
%     end
    
end     % End of convergence phase


% Plot final result
scatter(randomPoints(:,1), randomPoints(:,2),1)
axis equal
hold on
plot(kohonenPoints(:,1),kohonenPoints(:,2),'-or')
plot(plotLine(:,1),plotLine(:,2),'-k');
axis([0 1 0 1])
hold off

filename =  ['t1' num2str(initNbhWidth) 'c.png'];
saveas(gcf,filename,'png')
