Rcpp::sourceCpp('continuous.cpp')

library(e1071); library(ggplot2); library(reshape2); library(truncnorm)

# Initialize parameters
r = c(0.2, 0.6, 0.2)
u = rep(0, ncol(X0))
Y = matrix(0, nrow(X0), ncol(X0))
s = rgamma(ncol(X0), 1, 0.1); k1 = k2 = 5 * sqrt(s)
A = matrix(as.numeric(runif(nrow(X0)) < 0.5), nrow(X0))
z1 = rnorm(ncol(X0), 0, 10); z2 = rnorm(ncol(X0), 0, 10)
W1 = matrix(rgamma(ncol(X0), 1, 0.1), ncol(X0)); W2 = matrix(rgamma(ncol(X0), 1, 0.1), ncol(X0))
C = matrix(as.numeric(runif(ncol(X0)) < 0.5), ncol(X0)); C[C == 1] = C[C == 1] - 2 * (runif(sum(C)) < 0.5)

recordA = list(A); recordC = list(C); iter = 1; maxit = 20000

while(iter < maxit) {
  # update uniform bound and indicator in a block
  RY = updateY(X0, Y, A, C, W1, W2, z1, z2, u, k1, k2, s)
  k1 = RY$k1; k2 = RY$k2; Y = RY$Y
  # mean parameter
  u = updateu(X0, Y, s, k1, k2)
  # variance parameter
  s = updates(X0, Y, u, k1, k2)
  # latent association A
  RA = updateA(Y, A, C, W1, W2, z1, z2, r)
  A = RA$A; C = RA$C; W1 = RA$W1; W2 = RA$W2;
  # latent association C
  C = updateC(Y, A, C, W1, W2, z1, z2, r)
  # probability of indicator
  r = updater(C)
  # effect size
  W1 = updateW1(Y, A, C, W1, W2, z1, z2)
  W2 = updateW2(Y, A, C, W1, W2, z1, z2)
  # residual size
  z1 = updatez1(Y, A, C, W1, W2, z1, z2)
  z2 = updatez2(Y, A, C, W1, W2, z1, z2)
  
  recordA = c(recordA, list(A))
  recordC = c(recordC, list(C))
  
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
C = C[, perE(C0, C)]

pA0 = melt(t(A0), varnames = c("Factor", "Sample"), value.name = "Indicator")
pC0 = melt(t(C0), varnames = c("Factor", "Feature"), value.name = "Indicator")

ggplot(pA0, aes(x = Sample, y = Factor)) + geom_tile(aes(fill = Indicator), color = "black") +
  scale_fill_gradient(low = "black", high = "green") + theme(legend.position = "none") +
  scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) + labs(x = "", y = "")

ggplot(pC0, aes(x = Feature, y = Factor)) + geom_tile(aes(fill = Indicator), color = "black") +
  scale_fill_gradientn(colors = c("red", "black", "green"), breaks = c(-1, 0, 1)) + theme(legend.position = "none") +
  scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) + labs(x = "", y = "")

pA = melt(t(A), varnames = c("Factor", "Sample"), value.name = "Indicator")
pC = melt(t(C), varnames = c("Factor", "Feature"), value.name = "Indicator")

ggplot(pA, aes(x = Sample, y = Factor)) + geom_tile(aes(fill = Indicator), color = "black") +
  scale_fill_gradient(low = "black", high = "green") + theme(legend.position = "none") +
  scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) + labs(x = "", y = "")

ggplot(pC, aes(x = Feature, y = Factor)) + geom_tile(aes(fill = Indicator), color = "black") +
  scale_fill_gradientn(colors = c("red", "black", "green"), breaks = c(-1, 0, 1)) + theme(legend.position = "none") +
  scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) + labs(x = "", y = "")

