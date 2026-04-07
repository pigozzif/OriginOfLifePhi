function out=comty_beta_louvain(beta)
  %  Given a beta matrix, find indices of communities for each species in NG
  comty=cluster_jl_orient(beta);
  out.commind=squeeze(cell2mat(comty.COM(end)));
