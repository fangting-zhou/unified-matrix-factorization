Rcpp::sourceCpp('beta.cpp')

library(e1071); library(ggplot2); library(reshape2)

# Initialize parameters
r = 0.5
A = matrix(as.numeric(runif(nrow(X0)) < 0.5), nrow(X0))
B = matrix(as.numeric(runif(ncol(X0)) < 0.5), ncol(X0))
W = matrix(rgamma(ncol(X0), 1, 0.1), ncol(X0))
z = rnorm(ncol(X0), 0, 10)
u1 = runif(ncol(X0), 0, 0.5); u0 = 1 - u1
v0 = rgamma(ncol(X0), 1, 0.1); v1 = rgamma(ncol(X0), 1, 0.1)

recordA = list(A)
recordB = list(B)

# MCMC iterations
iter = 1; maxit = 20000

while(iter < maxit) {
  # latent indicator
  Z = updateZ(X0, A, B, W, z, u0, u1, v0, v1)
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
  # beta parameter
  u0 = updateu0(X0, Z, u0, u1, v0)
  u1 = updateu1(X0, Z, u0, u1, v1)
  v0 = updatev0(X0, Z, u0, v0)
  v1 = updatev1(X0, Z, u1, v1)
  
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