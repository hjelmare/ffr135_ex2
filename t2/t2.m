clear all
clc

tmpData = dlmread('wine.data.txt',',');

for iColumn = 2:size(tmpData,2)
    tmpData(:,iColumn) = tmpData(:,iColumn) - mean(tmpData(:,iColumn));
    tmpData(:,iColumn) = tmpData(:,iColumn) / std(tmpData(:,iColumn));
end


