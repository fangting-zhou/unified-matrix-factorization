Rcpp::sourceCpp('count.cpp')

library(e1071); library(ggplot2); library(reshape2)

# Initialize parameters
r = 0.5
A = matrix(as.numeric(runif(nrow(X0)) < 0.5), nrow(X0))
B = matrix(as.numeric(runif(ncol(X0)) < 0.5), ncol(X0))
Z = matrix(runif(nrow(X0) * ncol(X0)) < 0.5, nrow(X0), ncol(X0))
W = matrix(rgamma(ncol(X0), 1, 0.1), ncol(X0))
z = rnorm(ncol(X0), 0, 10)
p = rbeta(ncol(X0), 1, 1)
x = rbeta(ncol(X0), 1, 1)
l = rgamma(ncol(X0), 1, 1)
s = rgamma(ncol(X0), 1, 0.1)

recordA = list(A)
recordB = list(B)

# MCMC iterations
iter = 1; maxit = 20000

while(iter < maxit) {
  # latent indicator
  Z = updateZ(X0, A, B, W, z, p, l, s, x)
  # latent association A
  RA = updateA(Z, A, B, W, z, r)
  A = RA$A; B = RA$B; W = RA$W
  # latent association B
  B = updateB(Z, A, B, W, z, r)
  # probability of B
  r = updater(B)
  # effect size
  W = updateW(Z, A, B, W, z)
  # residual size
  z = updatez(Z, A, B, W, z)
  # zero inflation probability
  p = updatep(X0, Z, p, l)
  # rate in poisson
  l = updatel(X0, Z, p, l, s, x)
  # probability in NB
  x = updatex(X0, Z, s, l)
  # size in NB
  s = updates(X0, Z, s, x, l)
  
  recordA = c(recordA, list(A))
  recordB = c(recordB, list(B))
  
  iter = iter + 1
}

perE = function(A, B) {
  ## list all permutations
  permutation = permutations(ncol(A))

  DHamming = rep(NA, nrow(permutation))
  for(l in 1 : nrow(permutation)) {
    NB = as.vector(B[, permutation[l, ]])
    DHamming[l] = sum(abs(as.vector(A) - NB))
  }
  
  return(permutation[which.min(DHamming), ])
}

A = A[, perE(A0, A)]
B = B[, perE(B0, B)]
