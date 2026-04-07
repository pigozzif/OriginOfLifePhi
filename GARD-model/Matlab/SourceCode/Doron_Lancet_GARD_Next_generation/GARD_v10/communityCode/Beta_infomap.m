function out=Beta_infomap(seed, params)
% out=Beta_infomap(seed, params)
% Generates a beta and runs Infomap community analysis on the beta
% Accompanying data for  paper "Predicting Proto-Species Emergence in Simulated Complex Pre-Biotic Network", by Markovitch & Krasnogor

addpath '/gard'; %Directory of GARD MATLAB code
if ~exist('params', 'var') || isempty(params); params=tgs_parameters_v10; end

params.seed=seed;
b=tgs_newbeta_v10(params); %This generates the BETA
infomap='/Infomap '; %Infomap executeable
infoparams=' ./ --input-format ''link-list''  -d -k -z';
time=tic;
blink=zeros(params.NG^2,3);
ndx=1;
for i=1:params.NG; for j=1:params.NG; blink(ndx,1:2)=[i-1 j-1]; blink(ndx,3)=b(i,j); ndx=ndx+1; end; end;
prefix=['seed', num2str(seed)];
dlmwrite([prefix, '.txt'], blink, ' ');
[~,verb]=unix([infomap, [prefix,'.txt '], infoparams]);
comminfo=readfile([prefix, '.tree']);
inds(comminfo(:,4))=comminfo(:,1); %For each index of molecule (or node), to which community does it belong

out.seed=seed;
out.params=params;
out.infomap=infomap;
out.infoparams=infoparams;
out.infomapoutput=verb;
out.comminfo=comminfo;
out.indices=inds;
out.seconds=toc(time);
return;

function [arr, lines]=readfile(filename)
fileID=fopen(filename);
lines=[];
cont=true;

while cont;
	lines{end+1}=fgetl(fileID);
	cont=ischar(lines{end});
end

fclose(fileID);
arr=zeros(length(lines)-3,5); %[,1]=cluster id, [,2]=inner index, [,3]=flow, [,4]=index 1-100, [,5]=index 0-99

for l=3:length(lines)-1;
	bla=sscanf(lines{l},'%u:%u %f "%u" %u');
	arr(l-2,:)=bla;
end;

return


