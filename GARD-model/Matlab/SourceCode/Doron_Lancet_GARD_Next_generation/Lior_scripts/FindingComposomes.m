clc
%% Initial parameters
% N initial values seeds
% N beta matrices

p = tgs_parameters_v10(); % this is p.Nmax is set
p.NG = 1000;
p.seed = [139 1 39];
p.sigma = 3.25;
p.Beta = tgs_newbeta_v10(p);
p.gen = 1000;
p.splitsize = 0.1 % nmax = 1000 *0.1 = 100
%%
%% Add path for Gard10 source files
addpath('/home/labs/lancet/Collaboration/liors/SourceCode/Doron_Lancet_GARD_Next_generation/GARD_v10');

% the pmc is:
% pmc>1 is mutual-catalysis excess
pmc = pmc_beta_v10(p.Beta);
fprintf('pmc = %2.2f\n', pmc)
%c=tgs_carpet(o.trace); title('GARD'); xlabel('Generation'); ylabel('Generation');


% Simulate
%Find the location where the number of +- compositions that have H>=0.9 to each other is specified. Make both these numbers easy to modify at will
O = [];
tic
for beta_seed=139:139
    for Composition_seed=1:1
        p.seed = [beta_seed Composition_seed 39];
        p.Beta = tgs_newbeta_v10(p);
        o = tgs_agard_v10(p, 1);
        O = [O o];
    end
    beta_seed
end
toc
%save('c:\temp\20180715.mat', 'o_results', '-v7.3');

