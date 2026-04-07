%% Following instructions from Gard10 readme file
p = tgs_parameters_v10();
p.seed = [1 1 1];
p.Beta = tgs_newbeta_v10(p);
p.gen = 100;

o=tgs_agard_v10(p, 1);