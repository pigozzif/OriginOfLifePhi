function [t_in_gen, numberOfElements] = tgs_recreate_compositions_within_a_generation(t, ass)
% Programmer: Lior Segev
% Creation Date: 18 Feb 2019
% This function recreates the partial compositions being created in a
% single generation. inputs => n = initial composition, asshist = single
% joins and leaves in a single dt up until n reaches n_max. just before 
% split.
% out = is a cluster of compositions that resulted in a single dt at times
%       from the split to the time it composition reached n_max
    t_in_gen = zeros(length(t), length(ass)+1);
    t_in_gen(:, end) = t;
    numberOfElements = zeros(length(ass)+1,1);
    numberOfElements(end) = sum(t);
    for i=length(ass):-1:1
        % generate older composition
        t(abs(ass(i))) = t(abs(ass(i))) - sign(ass(i));
        t_in_gen(:, i) = t;
        numberOfElements(i) = sum(t);
    end

end