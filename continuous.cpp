// [[Rcpp::depends(RcppArmadillo)]]
# include <RcppArmadillo.h>

using namespace arma;
using namespace Rcpp;

// Accessing R function from Rcpp
// [[Rcpp::export]]
vec rtnorm(double n, double a, double b, double mean, double sd){
  
  // Obtain environment containing function
  Environment truncnorm("package:truncnorm"); 
  
  // Make function callable from Cpp
  Function rtruncnorm = truncnorm["rtruncnorm"];
  
  // Call the function and receive its output
  vec rnorm = as<vec>(rtruncnorm(n, a, b, mean, sd));
  
  // Return object
  return rnorm;
}

// [[Rcpp::export]]
List updateA(mat Y, mat A, mat C, mat W1, mat W2, vec z1, vec z2, vec r, double a = 2, double s = 1e-1, double K = 1e2) {
  // new matrix
  double c0 = 1; mat N = A; vec numk = rpois(A.n_rows, a / A.n_rows);
  
  for(int i = 0; i < A.n_rows; i ++) {
    vec index = ones<vec>(A.n_cols);
    // update existing latent features
    for(int k = 0; k < A.n_cols; k ++) {
      if((sum(A.col(k)) - A(i, k)) == 0) index(k) = 0; else {
        N(i, k) = 1 - A(i, k);
        
        // posterior 0 & 1 probability
        double prold = A(i, k) * log(sum(A.col(k)) - A(i, k) + a / K) + (1 - A(i, k)) * log(A.n_rows - sum(A.col(k)) + A(i, k));
        double prnew = N(i, k) * log(sum(N.col(k)) - N(i, k) + a / K) + (1 - N(i, k)) * log(N.n_rows - sum(N.col(k)) + N(i, k));
        
        for(int l = 0; l < C.n_rows; l ++) {
          double oddsold1 = sum(A.row(i) % W1.row(l) % (C.row(l) == 1)) + z1(l);
          double oddsnew1 = sum(N.row(i) % W1.row(l) % (C.row(l) == 1)) + z1(l);
          double oddsold2 = sum(A.row(i) % W2.row(l) % (C.row(l) == -1)) + z2(l);
          double oddsnew2 = sum(N.row(i) % W2.row(l) % (C.row(l) == -1)) + z2(l);
          
          prold += oddsold1 * (Y(i, l) == 1) + oddsold2 * (Y(i, l) == -1) - log(1 + exp(oddsold1) + exp(oddsold2));
          prnew += oddsnew1 * (Y(i, l) == 1) + oddsnew2 * (Y(i, l) == -1) - log(1 + exp(oddsnew1) + exp(oddsnew2));
        }
        
        if((prold - prnew) <= log(1 / randu() - 1)) A = N; else N = A;
      }
    }
    
    if((sum(index) != 0) & (sum(index) != A.n_cols)) {
      A = A.cols(find(index)); N = A; C = C.cols(find(index)); W1 = W1.cols(find(index)); W2 = W2.cols(find(index));
    }
    
    // generate new features
    if(numk(i) != 0) {
      double den = 0, num = 0;
      
      mat AN = zeros<mat>(A.n_rows, numk(i)); AN.row(i) = ones<rowvec>(numk(i)); AN = join_rows(A, AN);
      mat W1N = randg<mat>(W1.n_rows, numk(i), distr_param(c0, 1 / s)); W1N = join_rows(W1, W1N);
      mat W2N = randg<mat>(W2.n_rows, numk(i), distr_param(c0, 1 / s)); W2N = join_rows(W2, W2N);
      mat CN = zeros<mat>(C.n_rows, numk(i));
      for(int t = 0; t < C.n_rows; t ++) {
        for(int l = 0; l < numk(i); l ++) {
          double u = randu(); 
          
          if(u < r(2)) CN(t, l) = -1;
          if(u > (r(0) + r(2))) CN(t, l) = 1;
        }
      } CN = join_rows(C, CN);
      
      for(int l = 0; l < C.n_rows; l ++) {
        double oddsold1 = sum(A.row(i) % W1.row(l) % (C.row(l) == 1)) + z1(l);
        double oddsnew1 = sum(AN.row(i) % W1N.row(l) % (CN.row(l) == 1)) + z1(l);
        double oddsold2 = sum(A.row(i) % W2.row(l) % (C.row(l) == -1)) + z2(l);
        double oddsnew2 = sum(AN.row(i) % W2N.row(l) % (CN.row(l) == -1)) + z2(l);
        
        den += oddsold1 * (Y(i, l) == 1) + oddsold2 * (Y(i, l) == -1) - log(1 + exp(oddsold1) + exp(oddsold2));
        num += oddsnew1 * (Y(i, l) == 1) + oddsnew2 * (Y(i, l) == -1) - log(1 + exp(oddsnew1) + exp(oddsnew2));
      }
      
      if((num - den) >= log(randu())) {
        A = AN; N = A; C = CN; W1 = W1N; W2 = W2N;
      }
    }
  }
  
  return List::create(Named("A") = A, Named("C") = C, Named("W1") = W1, Named("W2") = W2);
}

// [[Rcpp::export]]
mat updateC(mat Y, mat A, mat C, mat W1, mat W2, vec z1, vec z2, vec r) {
  // new matrix
  mat C0 = C, C1 = C, C2 = C;
  
  for(int j = 0; j < C.n_rows; j ++) {
    for(int k = 0; k < C.n_cols; k ++) {
      C0(j, k) = 0; C1(j, k) = 1; C2(j, k) = -1;
      
      // posterior probability 0 & 1 & -1
      vec odds01 = A * trans(W1.row(j) % (C0.row(j) == 1)) + z1(j);
      vec odds02 = A * trans(W2.row(j) % (C0.row(j) == -1)) + z2(j);
      
      vec odds11 = A * trans(W1.row(j) % (C1.row(j) == 1)) + z1(j);
      vec odds12 = A * trans(W2.row(j) % (C1.row(j) == -1)) + z2(j);
      
      vec odds21 = A * trans(W1.row(j) % (C2.row(j) == 1)) + z1(j);
      vec odds22 = A * trans(W2.row(j) % (C2.row(j) == -1)) + z2(j);
      
      double prob0 = sum(odds01 % (Y.col(j) == 1) + odds02 % (Y.col(j) == -1) - log(1 + exp(odds01) + exp(odds02))) + log(r(0));
      double prob1 = sum(odds11 % (Y.col(j) == 1) + odds12 % (Y.col(j) == -1) - log(1 + exp(odds11) + exp(odds12))) + log(r(1));
      double prob2 = sum(odds21 % (Y.col(j) == 1) + odds22 % (Y.col(j) == -1) - log(1 + exp(odds21) + exp(odds22))) + log(r(2));;
      
      double u = randu(); C(j, k) = 0;
      
      if(u < (exp(prob2) / (exp(prob0) + exp(prob1) + exp(prob2)))) C(j, k) = -1;
      if(u > ((exp(prob0) + exp(prob2)) / (exp(prob0) + exp(prob1) + exp(prob2)))) C(j, k) = 1;
      
      C0 = C; C1 = C; C2 = C;
    }
  }
  
  return(C);
}

// [[Rcpp::export]]
vec updater(mat C, double a0 = 1, double a1 = 1, double a2 = 1) {
  double b = 1; vec r = zeros<vec>(3);
  
  // conjugate dirichlet distribution
  double a0new = a0 + accu(C == 0);
  double a1new = a1 + accu(C == 1);
  double a2new = a2 + accu(C == -1);
  
  double r0 = randg(distr_param(a0new, b));
  double r1 = randg(distr_param(a1new, b));
  double r2 = randg(distr_param(a2new, b));
  
  r(0) = r0 / (r0 + r1 + r2); r(1) = r1 / (r0 + r1 + r2); r(2) = r2 / (r0 + r1 + r2);
  
  return(r);
}

// [[Rcpp::export]]
mat updateW1(mat Y, mat A, mat C, mat W1, mat W2, vec z1, vec z2, double s = 1e-1, double s0 = 1) {
  // new matrix
  mat N1 = W1;
  
  for(int j = 0; j < W1.n_rows; j ++) {
    for(int k = 0; k < W1.n_cols; k ++) {
      // sample from random walk centered at current value
      do N1(j, k) = W1(j, k) + s0 * randn(); while(N1(j, k) < 0);
      
      // calculate acceptance ratio
      vec prob1 = A * trans(W1.row(j) % (C.row(j) == 1)) + z1(j);
      vec prob2 = A * trans(N1.row(j) % (C.row(j) == 1)) + z1(j);
      vec prob3 = A * trans(W2.row(j) % (C.row(j) == -1)) + z2(j);
      
      double den = sum(prob1 % (Y.col(j) == 1) + prob3 % (Y.col(j) == -1) - log(1 + exp(prob1) + exp(prob3))) - s * W1(j, k);
      double num = sum(prob2 % (Y.col(j) == 1) + prob3 % (Y.col(j) == -1) - log(1 + exp(prob2) + exp(prob3))) - s * N1(j, k);
      
      if((num - den) >= log(randu())) W1 = N1; else N1 = W1;
    }
  }
  
  return(W1);
}

// [[Rcpp::export]]
mat updateW2(mat Y, mat A, mat C, mat W1, mat W2, vec z1, vec z2, double s = 1e-1, double s0 = 1) {
  // new matrix
  mat N2 = W2;
  
  for(int j = 0; j < W2.n_rows; j ++) {
    for(int k = 0; k < W2.n_cols; k ++) {
      // sample from random walk centered at current value
      do N2(j, k) = W2(j, k) + s0 * randn(); while(N2(j, k) < 0);
      
      // calculate acceptance ratio
      vec prob1 = A * trans(W1.row(j) % (C.row(j) == 1)) + z1(j);
      vec prob2 = A * trans(W2.row(j) % (C.row(j) == -1)) + z2(j);
      vec prob3 = A * trans(N2.row(j) % (C.row(j) == -1)) + z2(j);
      
      double den = sum(prob1 % (Y.col(j) == 1) + prob2 % (Y.col(j) == -1) - log(1 + exp(prob1) + exp(prob2))) - s * W2(j, k);
      double num = sum(prob1 % (Y.col(j) == 1) + prob3 % (Y.col(j) == -1) - log(1 + exp(prob1) + exp(prob3))) - s * N2(j, k);
      
      if((num - den) >= log(randu())) W2 = N2; else N2 = W2;
    }
  }
  
  return(W2);
}

// [[Rcpp::export]]
vec updatez1(mat Y, mat A, mat C, mat W1, mat W2, vec z1, vec z2, double s = 1e2, double s0 = 1) {
  for(int j = 0; j < z1.n_elem; j ++) {
    // sample from random walk centered at current value
    double znew = z1(j) + s0 * randn();
    
    // calculate acceptance ratio
    vec prob1 = A * trans(W1.row(j) % (C.row(j) == 1)) + z1(j);
    vec prob2 = A * trans(W1.row(j) % (C.row(j) == 1)) + znew;
    vec prob3 = A * trans(W2.row(j) % (C.row(j) == -1)) + z2(j);
    
    double den = sum(prob1 % (Y.col(j) == 1) + prob3 % (Y.col(j) == -1) - log(1 + exp(prob1) + exp(prob3))) - pow(z1(j), 2) / (2 * s);
    double num = sum(prob2 % (Y.col(j) == 1) + prob3 % (Y.col(j) == -1) - log(1 + exp(prob2) + exp(prob3))) - pow(znew, 2) / (2 * s);
    
    if((num - den) >= log(randu())) z1(j) = znew;
  }
  
  return(z1);
}

// [[Rcpp::export]]
vec updatez2(mat Y, mat A, mat C, mat W1, mat W2, vec z1, vec z2, double s = 1e2, double s0 = 1) {
  for(int j = 0; j < z2.n_elem; j ++) {
    // sample from random walk centered at current value
    double znew = z2(j) + s0 * randn();
    
    // calculate acceptance ratio
    vec prob1 = A * trans(W1.row(j) % (C.row(j) == 1)) + z1(j);
    vec prob2 = A * trans(W2.row(j) % (C.row(j) == -1)) + z2(j);
    vec prob3 = A * trans(W2.row(j) % (C.row(j) == -1)) + znew;
    
    double den = sum(prob1 % (Y.col(j) == 1) + prob2 % (Y.col(j) == -1) - log(1 + exp(prob1) + exp(prob2))) - pow(z2(j), 2) / (2 * s);
    double num = sum(prob1 % (Y.col(j) == 1) + prob3 % (Y.col(j) == -1) - log(1 + exp(prob1) + exp(prob3))) - pow(znew, 2) / (2 * s);
    
    if((num - den) >= log(randu())) z2(j) = znew;
  }
  
  return(z2);
}

// [[Rcpp::export]]
List updateY(mat X, mat Y, mat A, mat C, mat W1, mat W2, vec z1, vec z2, vec u, vec k1, vec k2, vec s, double k0 = 5, double b0 = 1e-1) {
  mat N = Y; double pi = 3.14; double a0 = 1;
  
  for(int j = 0; j < X.n_cols; j ++) {
    double k1new = randg(distr_param(a0, 1 / b0)), k2new = randg(distr_param(a0, 1 / b0));
    
    if(k1new < sqrt(s(j)) * k0) k1new = sqrt(s(j)) * k0;
    if(k2new < sqrt(s(j)) * k0) k2new = sqrt(s(j)) * k0;
    
    vec odds1 = zeros<vec>(X.n_rows), odds2 = zeros<vec>(X.n_rows);
    
    for(int i = 0; i < X.n_rows; i ++) {
      N(i, j) = 0;
      
      // posterior probability 0 & 1 & -1
      odds1(i) = sum(A.row(i) % W1.row(j) % (C.row(j) == 1)) + z1(j);
      odds2(i) = sum(A.row(i) % W2.row(j) % (C.row(j) == -1)) + z2(j);
      
      double prob0 = pow(X(i, j) - u(j), 2) / (2 * s(j)) + log(2 * pi * s(j)) / 2;
      double prob1 = log(k1new) - odds1(i);
      double prob2 = log(k2new) - odds2(i);
      
      if(X(i, j) > u(j) & X(i, j) < (u(j) + k1new)) {if((prob1 - prob0) < log(1 / randu() - 1)) N(i, j) = 1;}
      if(X(i, j) < u(j) & X(i, j) > (u(j) - k2new)) {if((prob2 - prob0) < log(1 / randu() - 1)) N(i, j) = -1;}
    }
    
    double num = sum(log(exp(odds1) % (X.col(j) < u(j) + k1new) % (X.col(j) > u(j)) + 
                     exp(odds2) % (X.col(j) > u(j) - k2new) % (X.col(j) < u(j)) +
                     exp(- pow(X.col(j) - u(j), 2) / (2 * pi * s(j))) / sqrt(2 * pi * s(j))));
    double den = sum(log(exp(odds1) % (X.col(j) < u(j) + k1(j)) % (X.col(j) > u(j)) + 
                     exp(odds2) % (X.col(j) > u(j) - k2(j)) % (X.col(j) < u(j)) +
                     exp(- pow(X.col(j) - u(j), 2) / (2 * pi * s(j))) / sqrt(2 * pi * s(j))));
    
    if((num - den) >= log(randu())) {
      k1(j) = k1new; k2(j) = k2new; Y = N;
    } else N = Y;
  }
  
  return List::create(Named("Y") = Y, Named("k1") = k1, Named("k2") = k2);
}

// [[Rcpp::export]]
vec updates(mat X, mat Y, vec u, vec k1, vec k2, double k0 = 5, double a0 = 1e-3, double b0 = 1e-3) {
  vec s = zeros<vec>(X.n_cols);
  
  for(int j = 0; j < X.n_cols; j ++) {
    // upper bound
    vec uk = zeros<vec>(2); uk(0) = k1(j) / k0; uk(1) = k2(j) / k0;
    
    // conjugate inverse gamma
    double anew = a0 + sum(Y.col(j) == 0) / 2;
    double bnew = b0 + sum((Y.col(j) == 0) % pow(X.col(j) - u(j), 2)) / 2;
    
    s(j) = 1 / randg(distr_param(anew, 1 / bnew));
    // upper bound constrain
    if(s(j) > pow(min(uk), 2)) s(j) = pow(min(uk), 2);
  }
  
  return(s);
}

// [[Rcpp::export]]
vec updateu(mat X, mat Y, vec s, vec k1, vec k2, double s0 = 100) {
  vec u = zeros<vec>(X.n_cols); double n = 1;
  
  for(int j = 0; j < X.n_cols; j ++) {
    double low1 = -1e10, low2 = -1e10, up1 = 1e10, up2 = 1e10;
    
    if(sum(Y.col(j) == 1) != 0) {
      vec lb1 = X.col(j) - k1(j); lb1 = lb1(find(Y.col(j) == 1)); low1 = max(lb1);
      vec ub1 = X.col(j); ub1 = ub1(find(Y.col(j) == 1)); up1 = min(ub1);
    }
    
    if(sum(Y.col(j) == -1) != 0) {
      vec lb2 = X.col(j); lb2 = lb2(find(Y.col(j) == -1)); low2 = max(lb2);
      vec ub2 = X.col(j) + k2(j); ub2 = ub2(find(Y.col(j) == -1)); up2 = min(ub2);
    }
    
    vec lb = zeros<vec>(2), ub = zeros<vec>(2); lb(0) = low1; lb(1) = low2; ub(0) = up1; ub(1) = up2;
    
    // conjugate normal
    double snew = 1 / (1 / s0 + sum(Y.col(j) == 0) / s(j));
    double mnew = snew * sum((Y.col(j) == 0) % X.col(j)) / s(j);
    
    u(j) = rtnorm(n, max(lb), min(ub), mnew, sqrt(snew))(0);
  }
  
  return(u);
}
