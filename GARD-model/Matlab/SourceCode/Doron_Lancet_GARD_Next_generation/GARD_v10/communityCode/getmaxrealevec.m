function [evec, maxeval, realevals]=getmaxrealevec(beta)
% [evec, maxeval, realevals]=getmaxrealevec(beta)
% Receives a network (beta or pseudo-beta) and return the eigenvector of the largest real eigenvalue
% Accompanying data for  paper "Predicting Proto-Species Emergence in Simulated Complex Pre-Biotic Network", by Markovitch & Krasnogor

% maxeval & realevals are indices
[evec,evals]=eig(beta);
evals=diag(evals);
realevals=[]; for i=1:length(evals); if(isreal(evals(i))); realevals(end+1)=i; end; end;
maxeval=max(evals(realevals));
maxeval=realevals(find(evals(realevals)==maxeval));
evec=evec(:,maxeval);
if sum(abs(evec))~=abs(sum(evec)); error('eigen vector of the highest real eigen value is not all same signed!.'); end;
return

