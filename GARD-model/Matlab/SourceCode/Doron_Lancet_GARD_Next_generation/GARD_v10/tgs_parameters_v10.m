function p=tgs_parameters_v10(p)
% p=tgs_parameters_v10(p)
% returns a set of default parameters and initializes the random number generator
% does not generate a beta matrix.
% 02/06/2011 GARD10, by Omer Markovitch

s1 = rand('state');
s2 = randn('state');

p.gen=1000; %number of generations
p.NG=1000; %number of different amphiphiles
p.splitsize=0.1; %Nmax=splitsize*NG
p.mu=-4; %mu of beta matrix lognormal distribution
p.sigma=4; %sigma of beta matrix lognormal distribution
p.Kf=10^-2; %forward, entering
p.Kb=10^-4; %backward, leaving
p.n=zeros(p.NG, 1);
p.seed = [0 0 0]; %1 is the seed for the beta matrix and 2 is the seed for gard and 3 is the seed for clustering
p.version=10.0;
p.randomRun = false;

%basic clustering parameters
p.ks = [1:10]; % the various possible k values to test in k-means
p.hthresh=0.9; % the cutoff for drift/nondrift
p.replicas=10; % the number of replicas per k in k-means
p.mink=4; % if no improvement for this many k values in a row then no new k values are tested.

p.rho = []; %environmental concentration
%state = randn('state');
%randn('state', p.seed(1));
p.Beta=[];

p=orderfields(p);

rand('state', s1);
randn('state', s2);
return;
