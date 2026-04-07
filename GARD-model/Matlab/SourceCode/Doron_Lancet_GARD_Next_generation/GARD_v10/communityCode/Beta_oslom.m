function out=Beta_oslom(seed, params)
% out=Beta_Oslom(seed, params)
% Generates a beta and runs OSLOM community analysis on the beta
% Accompanying data for  paper "Predicting Proto-Species Emergence in Simulated Complex Pre-Biotic Network", by Markovitch & Krasnogor

addpath '/gard'; %Directory of GARD MATLAB code
if ~exist('params', 'var') || isempty(params); params=tgs_parameters_v10; end

params.seed=seed;
b=tgs_newbeta_v10(params); %This generates the BETA
filename=['seed', num2str(seed), '.txt'];
oslomdir=[filename, '_oslo_files/'];
oslom=['oslom_dir -f ', filename]; %OSLOM executeable & parameters
oslomparams=[' -w -seed ', num2str(seed), ' -singlet'];
time=tic;
blink=zeros(params.NG^2,3);
ndx=1;
for i=1:params.NG; for j=1:params.NG; blink(ndx,1:2)=[i j]; blink(ndx,3)=b(i,j); ndx=ndx+1; end; end;
dlmwrite(filename, blink, ' ');
[~,verb]=unix([oslom, oslomparams]);
[inds, comminfo]=readoslomfile([oslomdir, 'tp']); %CAREFUL: can have overlap, meaning a molecule in two different communities

[~,verb]=unix(['\rm -r ', oslomdir]);

out.seed=seed;
out.params=params;
out.oslom=oslom;
out.oslomparams=oslomparams;
out.oslomoutput=verb;
out.modules=comminfo;
out.indices=inds; %Because there can be overlaps, this is given as an array of MATLAB 'cell' structures
out.seconds=toc(time);
return;

function [arr, modules, lines]=readoslomfile(filename)
fileID=fopen(filename);
lines=[];
cont=true;

while cont;
	lines{end+1}=fgetl(fileID);
	cont=ischar(lines{end});
end

fclose(fileID);
manyclust=floor(length(lines)/2);
modules=cell(manyclust,1);
arr=cell(100,1);
inds=[];

for i=1:manyclust;
	bla=sscanf(lines{2*i-1},'#module %u size: %u bs: %f');
	modules{i}.manymembers=bla(2);
	modules{i}.score=bla(3);
	bla=sscanf(lines{2*i},'%u');
	modules{i}.members=bla;
	for m=1:length(bla); arr{bla(m)}(end+1)=i; end;	
end;

return


