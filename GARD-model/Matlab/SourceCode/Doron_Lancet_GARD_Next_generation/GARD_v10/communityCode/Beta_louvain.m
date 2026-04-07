function out=Beta_louvain(seed, params)
% out=Beta_Oslom(seed, params)
% Generates a beta and runs Louvain community analysis on the beta
% Accompanying data for  paper "Predicting Proto-Species Emergence in Simulated Complex Pre-Biotic Network", by Markovitch & Krasnogor

%addpath '\Community_BGLL_Matlab'; %Directory of Louvain MATLAB code
%addpath '/gard'; %Directory of GARD MATLAB code
if ~exist('params', 'var') || isempty(params); params=tgs_parameters_v10; end

params.seed=seed;
b=tgs_newbeta_v10(params); %This generates the BETA
comty=cluster_jl_orient(b);
ind=squeeze(cell2mat(comty.COM(end)));

out.seed=seed;
out.commind=ind;

return;



