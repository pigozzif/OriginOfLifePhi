function beta=tgs_newbeta_v10(params)
% beta=tgs_newbeta_v10(params)
% Will generate a beta matrix from a lognormal distribution.
% You can check the distribution with hist(log(beta(:)), 30), plot using surf(beta).
% 16/06/2011 GARD10, by Omer Markovitch

s1 = rand('state');
s2 = randn('state');

if ~exist('params', 'var') || isempty(params); params=tgs_parameters; end;
randn('state', params.seed(1));
rand('state', params.seed(1));
beta=exp(randn(params.NG, params.NG).*params.sigma + params.mu); %http://en.wikipedia.org/wiki/Lognormal#Generating_log-normally-distributed_random_variates

rand('state', s1);
randn('state', s2);
return;
