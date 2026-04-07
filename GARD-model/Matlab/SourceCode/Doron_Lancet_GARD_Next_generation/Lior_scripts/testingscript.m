a = zeros(1000,1);
a([c_critical(:).range]) = 1;
plot(a)
[index angles] = tgs_nondrift(o.trace, 0.9, 3);


trace = o.trace ./ (ones(p.NG,1) * (sum(o.trace.*o.trace).^.5)); 
avgH = sum(trace(:,1:end-1).*trace(:,2:end)); % scalar multiplication of two vectors
 
H_mat = tgs_H(trace(:,1:end-1), trace(:,2:end));

avgH_usedbyLior = sum(H_mat)/p.gen;
subplot(2,1,2)
plot(avgH_usedbyLior)

subplot(2,1,1)
plot(avgH)


eps=0.0000001;
norm1 = o.trace ./ (ones(size(o.trace,1),1) * max(sqrt(sum(o.trace.^2)),eps));
result = norm1 - trace;