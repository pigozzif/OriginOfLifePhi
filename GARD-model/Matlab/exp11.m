% Programmer : Lior Segev
% Date : 17 Feb 2019
% Find the number of compotype for a range of beta matrices (specified
% seeds) and a predifined initial composition vectors

% The output of this script 
% raw_results_table => index => number of run
%                  init_cond => index into init_cond_comp_mat to find
%                  initial composition
%                  beta_seed => beta seed number
%                  beta_seed_index => to index into beta_matrices and
%                  retrieve a specific beta matrix
% compotype_mat => cell array (length = number of table rows) contains
%                  compotype vectors for a specific run
% beta_matrices => cell array (length = number of beta matrices seeds) contains the
%                  beta matrix used in this run(row)
% init_cond_comp_mat => matrix => columns are the initial composition
%                       vector (sum of elements is equal to splitsize*NG/2)
% output => array of result run => length number of rows, include all
%                                  results and parameter of that run
% selected_beta_seeds => beta seeds that produced 1 compotype for all
%                        inital compositions
clearvars
%% INPUTS
beta_matrices_seeds = 1:100;
num_init_composition = 1;

%% Supply time series of compositional vectors

p=tgs_parameters_v10();
p.gen = 2;
p.splitsize = 1;
p.NG = 100;
p.hthresh = 0.9;
%beta_seed = 3;
initial_composition_seed = 1;
%p.seed=[beta_seed initial_composition_seed 1]; % first parameters is the seed for beta
%p.Beta=tgs_newbeta_v10(p);
p.randomRun = false;  % this means that the GARD simulation will be random but with fixed beta matrix and fixed initial conditions
storeRandstate = rand('state');

%% Generate synthetic conditions
% generate Initial condition vectors, should be read from a mat file
% generated in python by Amit
nmax=ceil(p.splitsize*p.NG); %the size at which the assembly splits
nmin=floor(nmax/2);
init_cond_comp_mat = histc(rand(nmin ,num_init_composition)*p.NG, 0:p.NG);
init_cond_comp_mat = init_cond_comp_mat(1:p.NG,:);
% THIS IS WHERE you should load init_cond_comp_mat FILENAME.MAT ->
% load("filepath and name")
init_cond_comp_mat = csvread('Compotypes compositions for experiment 15Jun20.csv');

% Generate Beta matrices
beta_matrices={};
for b_index = 1:size(beta_matrices_seeds, 2)
    p.seed = [beta_matrices_seeds(b_index) initial_composition_seed 1];
    beta_matrices{b_index} = tgs_newbeta_v10(p);
end


%% Simulate - generate generations
output = []; 
for b_index = 1:size(beta_matrices_seeds, 2)
    for c_index = 1:num_init_composition
        p.seed = [beta_matrices_seeds(b_index) initial_composition_seed 1];
        p.Beta = beta_matrices{b_index};
        p.n = init_cond_comp_mat(:,c_index);
        output = [output; tgs_agard_v10(p, 2)]; % second parameter selects the cluster norm => 1 cosine,2 sqEuclidean dist
        fprintf('initial composition number %d/%d\n', c_index, num_init_composition)
    end
    fprintf('beta seed number %d/%d done\n', b_index, size(beta_matrices_seeds, 2))
end

% return seed
if (~p.randomRun); rand('state', storeRandstate); end

%% Analyze results
init_cond =[]; beta_seed=[]; num_compotypes=[]; compotype_mat = {};c1 = 1;index=[]; beta_seed_index =[];
for b_index = 1:size(beta_matrices_seeds, 2)
    for c_index = 1:num_init_composition
         init_cond = [init_cond; c_index];
         beta_seed = [beta_seed; beta_matrices_seeds(b_index)];
         beta_seed_index = [beta_seed_index; b_index]; 
         num_compotypes = [num_compotypes; size(output(c1).comps, 2)];
         compotype_mat{c1} = output(c1).comps;
         index = [index;c1];
         c1 = c1 +1;
    end
end

results_table = table(index, init_cond, beta_seed, beta_seed_index, num_compotypes);
raw_results_table = results_table

% find beta seeds that have X compotype
search_for_X_compotypes = 1;
results_table = results_table(results_table.num_compotypes == search_for_X_compotypes, :);
results_table.beta_seed = categorical(results_table.beta_seed);
sumarray = grpstats(results_table.num_compotypes, results_table.beta_seed, @sum);
groupindices = grp2idx(results_table.beta_seed);
results_table.sums = sumarray(groupindices);
results_table = results_table(results_table.sums == search_for_X_compotypes * num_init_composition,:);
selected_beta_seeds = unique(results_table.beta_seed)

%% examples on how to inspect data
% examining a run (y is the molecule type, x - genenration, color how
% nuch that molecule type is occupied
subplot(3,1,1)
imagesc(output(1).trace)
xlabel('generation')
ylabel('molecule type')

% beta matrix for a run
subplot(3,1,2)
imagesc(output(1).Beta)
% or
%beta_seed_index = 1;
%imagesc(beta_matrices{beta_seed_index})

% initial composition
subplot(3,1,3)
imagesc(init_cond_comp_mat)

