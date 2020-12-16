# generate sample <-> latent feature matrix A
generateA = function(n, a, k) {
  flag = 0
  
  while(flag == 0) {
    # first sample generate features
    nones = rpois(1, a)
    while(nones == 0) nones = rpois(1, a)
    A = matrix(rep(1, nones), nrow = 1)
    
    # subsequent samples
    for(i in 2 : n) {
      # update existing features
      A = rbind(A, runif(ncol(A)) < colSums(A) / i)
      # generate new features
      nones = rpois(1, a / i)
      if(nones != 0) {
        A = cbind(A, matrix(0, nrow(A), nones))
        A[i, (ncol(A) - nones) : ncol(A)] = 1
      }
    }
    
    # accepting criteria
    if(ncol(A) == k) flag = 1
  }
  
  return(A)
}

# sample size n = 1000; sparsity parameter a = 2; number of latent features k = 6
A0 = generateA(1000, 2, 6)
colSums(A0)

# generate feature <-> latent feature matrix B (binary): number of features p = 50
B0 = matrix(as.numeric(runif(50 * 6) < 0.3), 50, 6)

# generate feature <-> latent feature matrix C (ternary): number of features p = 50
C0 = matrix(as.numeric(runif(50 * 6) < 0.15), 50, 6)
C0[C0 == 1] = 2 * (runif(sum(C0 == 1)) < 0.5) - 1
colSums(C0==1)
colSums(C0==-1)

# residual z and latent effect size w
z0 = log(0.1); w0 = c(3.0, 3.5, 4.0, 4.5, 5.0, 5.5)

# generate latent binary indicator Z
odds = exp(A0 %*% diag(w0) %*% t(B0) + z0); Z0 = matrix(as.numeric(runif(nrow(A0) * nrow(B0)) < odds / (1 + odds)), nrow(A0), nrow(B0))

# generate latent ternary indicator Y
oddsm = exp(A0 %*% diag(w0) %*% t(C0 == 1) + z0); oddsn = exp(A0 %*% diag(w0) %*% t(C0 == -1) + z0)
U0 = runif(nrow(A0) * nrow(C0)); Y0 = (U0 > (oddsn + 1) / (oddsm + oddsn + 1)) - (U0 < (oddsn) / (oddsm + oddsn + 1))

# generate observation X from mixture of ZIP and NB
X0 = matrix(0, nrow(A0), nrow(B0))
# NB - size r = 5; probability prob = 0.2
X0[Z0 == 1] = rnbinom(sum(Z0), 5, 0.2)
# ZIP - inflation parameter pi = 0.9; rate lambda = 1
R0 = runif(sum(1 - Z0)) < 0.9; X0[Z0 == 0] = (1 - R0) * rpois(sum(1 - Z0), 1)

# generate observation X from mixture of beta
X0 = matrix(0, nrow(A0), nrow(B0))
# BE(2, 8) & BE(8, 2)
X0[Z0 == 0] = rbeta(sum(1 - Z0), 8, 2); X0[Z0 == 1] = rbeta(sum(Z0), 2, 8)

# generate observation X from mixture of three normals
X0 = matrix(0, nrow(A0), nrow(C0))
# N(0, 1) - mean m = 0; sd s = 1
X0[Y0 == 0] = rnorm(sum(Y0 == 0), 0, 1)
# N(10, 5) - mean m = 10; sd s = 5
X0[Y0 == 1] = rnorm(sum(Y0 == 1), 10, 5)
# N(-10, 5) - mean m = -10; sd s = 5
X0[Y0 == -1] = rnorm(sum(Y0 == -1), -10, 5)
