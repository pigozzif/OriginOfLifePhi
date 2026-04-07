%% Programmer : Lior Segev
% Date : 12 Feb 2020
% The Plan for this script:
% Choose beta seed,
% Choose initial conditions 
% Simulate
% output is in composition cell {beta_index, composition_index}

clearvars
%% Supply time series of compositional vectors
%1.1 Generate a beta matrix
p=tgs_parameters_v10();
p.sigma = 4;
p.mu = -4;
p.gen = 30;
p.splitsize = 1;
p.NG = 100;
p.hthresh = 0.9;
num_init_composition = 100;
beta_matrices_seeds = [5];
is_dt = true; % is_dt = true -> dt is with in generation , false -> dt is generations
p.randomRun = true;  % this means that the GARD simulation will be random but with fixed beta matrix and fixed initial conditions


%% Generate synthetic conditions
% generate Initial condition vectors, should be read from a mat file
% generated in python by Amit
nmax=ceil(p.splitsize*p.NG); %the size at which the assembly splits
nmin=floor(nmax/2);
init_cond_comp_mat = histc(rand(nmin ,num_init_composition)*p.NG, 0:p.NG);
init_cond_comp_mat = init_cond_comp_mat(1:p.NG,:);

% THIS IS WHERE you should load init_cond_comp_mat FILENAME.MAT ->
% init_cond_comp_mat = load(directory); % that will replace init_cond_comp_mat

% uncomment next lines to load your own database of initial compositions:
% % init_cond_comp_mat = csvread('beta 5 100 runs equimolar rho step 500.csv');
% % init_cond_comp_mat = init_cond_comp_mat';
% % num_init_composition = size(init_cond_comp_mat,2);

% Generate Beta matrices
beta_matrices={};
for b_index = 1:size(beta_matrices_seeds, 2)
    p.seed = [beta_matrices_seeds(b_index) 1 1];
    beta_matrices{b_index} = tgs_newbeta_v10(p);
    % For control on catalysis:
%     beta_counter = beta_matrices_seeds(b_index);
%     beta_matrices{b_index} = zeros(p.NG);
%     beta_matrices{b_index} = readtable(['beta_matrix_standard_' num2str(beta_counter) '.csv']);
%     beta_matrices{b_index} = table2array(beta_matrices{b_index}).*(10);
end
p.Beta = beta_matrices;

%% Simulate - generate generations:

compositions = {};
fluxes = {}; % added by Amit 1May20 to record the fluxes

for b_index = 1:size(beta_matrices_seeds, 2)
    rho = p.rho;
%     Here we input specific rho for rho-controlled experiments:
%     rho = [0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;
%         0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;
%         0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;
%         0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;
%         0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;
%         0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;
%         0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;
%         0.01;0.01;0.02;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;0.01;
%         0.01;0.01;0.01;0.01]
%     p.rho = rho / sum(rho);
    for c_index = 1:num_init_composition
        p.seed = [beta_matrices_seeds(b_index) 1 1];
        p.Beta = beta_matrices{b_index};
        p.n = init_cond_comp_mat(:,c_index);
        o = tgs_agard_v10(p, 2); % second parameter selects the cluster norm => 1 cosine,2 sqEuclidean dist
        fprintf('initial composition number %d/%d\n', c_index, num_init_composition)

        if (is_dt)
            % generate fluxes (Amit Kahana)
            fluxes{b_index, c_index} = o.fluxes;
            % generate compositions within generations
            c_in_gens =[];
            numberOfElements_inGens=[];
            for i=1:p.gen
                [c_in_gen, numberOfElements_inGen]=tgs_recreate_compositions_within_a_generation(o.trace(:,i), o.asshist{i});
                c_in_gens = [c_in_gens; c_in_gen'];
            end 
            compositions{b_index, c_index} = c_in_gens';
        else
            compositions{b_index, c_index} = o.trace;
        end
    end
    fprintf('beta seed number %d/%d done\n', b_index, size(beta_matrices_seeds, 2))
end

filename = [datestr(now, 'YYYY-mm-dd_HH-MM-SS') '.mat'];
fprintf(['Saving compositions variable to file: ' filename]);
% For big data files, add '-v7.3' in the save function.
save(filename, 'compositions', 'beta_matrices_seeds', 'p', 'rho', ...
    'fluxes','num_init_composition', 'beta_matrices', '-v7.3');

%% plot results
%subplot_index = 1;
%for b_index = 1:size(beta_matrices_seeds, 2)
%    for c_index = 1:num_init_composition
%       subplot(size(beta_matrices_seeds, 2), num_init_composition, subplot_index)
%       imagesc(compositions{b_index, c_index})
%       subplot_index = subplot_index + 1;
%       title(sprintf("b_{seed}: %d, init index: %d", [beta_matrices_seeds(b_index) c_index]));
%    end
%end


