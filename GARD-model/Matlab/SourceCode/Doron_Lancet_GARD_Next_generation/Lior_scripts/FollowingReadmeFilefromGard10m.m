
p = tgs_parameters_v10();
p.seed = [1 1 1];
p.NG = 100;
p.Beta = tgs_newbeta_v10(p);

%%
o=tgs_agard_v10(p, 1);
c=tgs_carpet(o.trace); title('GARD'); xlabel('Generation'); ylabel('Generation');
