// [[Rcpp::depends(RcppArmadillo)]]
# include <RcppArmadillo.h>

using namespace arma;
using namespace Rcpp;

// [[Rcpp::export]]
List updateA(mat Z, mat A, umat B, mat W, vec z, double r, double a = 1, double s = 0.1, double K = 100) {
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
        
        for(int l = 0; l < B.n_rows; l ++) {
          double oddsold = sum(A.row(i) % W.row(l) % B.row(l)) + z(l);
          double oddsnew = sum(N.row(i) % W.row(l) % B.row(l)) + z(l);
          
          prold += oddsold * Z(i, l) - log(1 + exp(oddsold));
          prnew += oddsnew * Z(i, l) - log(1 + exp(oddsnew));
        }
        
        if((prold - prnew) <= log(1 / randu() - 1)) A = N; else N = A;
      }
    }
    
    if((sum(index) != 0) & (sum(index) != A.n_cols)) {
      A = A.cols(find(index)); N = A; B = B.cols(find(index)); W = W.cols(find(index));
    }
    
    // generate new features
    if(numk(i) != 0) {
      double den = 0, num = 0;
      
      mat AN = zeros<mat>(A.n_rows, numk(i)); AN.row(i) = ones<rowvec>(numk(i)); AN = join_rows(A, AN);
      mat WN = randg<mat>(W.n_rows, numk(i), distr_param(c0, 1 / s)); WN = join_rows(W, WN);
      umat BN = randu<mat>(B.n_rows, numk(i)) < r; BN = join_rows(B, BN);
      
      for(int l = 0; l < B.n_rows; l ++) {
        double oddsden = sum(A.row(i) % W.row(l) % B.row(l)) + z(l);
        double oddsnum = sum(AN.row(i) % WN.row(l) % BN.row(l)) + z(l);
        
        den += oddsden * Z(i, l) - log(1 + exp(oddsden));
        num += oddsnum * Z(i, l) - log(1 + exp(oddsnum));
      }
      
      if((num - den) >= log(randu())) {
        A = AN; N = A; B = BN; W = WN;
      }
    }
  }
  
  return List::create(Named("A") = A, Named("B") = B, Named("W") = W);
}

// [[Rcpp::export]]
mat updateB(mat Z, mat A, mat B, mat W, vec z, double r) {
  // new matrix
  mat N = B;
  
  for(int j = 0; j < B.n_rows; j ++) {
    for(int k = 0; k < B.n_cols; k ++) {
      N(j, k) = 1 - B(j, k);
      
      // posterior 0 & 1 probability
      vec oddsold = A * trans(W.row(j) % B.row(j)) + z(j);
      vec oddsnew = A * trans(W.row(j) % N.row(j)) + z(j);
      
      double prold = sum(oddsold % Z.col(j) - log(1 + exp(oddsold))) + log(r) * B(j, k) + log(1 - r) * (1 - B(j, k));
      double prnew = sum(oddsnew % Z.col(j) - log(1 + exp(oddsnew))) + log(r) * N(j, k) + log(1 - r) * (1 - N(j, k));
      
      if((prold - prnew) <= log(1 / randu() - 1)) B = N; else N = B;
    }
  }
  
  return(B);
}

// [[Rcpp::export]]
double updater(mat B, double a0 = 1, double b0 = 1) {
  double c0 = 1;
  
  // conjugate beta distribution
  double anew = a0 + accu(B);
  double bnew = b0 + accu(1 - B);
  
  double x = randg(distr_param(anew, c0));
  double y = randg(distr_param(bnew, c0));
  
  return(x / (x + y));
}

// [[Rcpp::export]]
mat updateW(mat Z, mat A, mat B, mat W, vec z, double s = 0.1, double s0 = 1) {
  // new matrix
  mat N = W;
  
  for(int j = 0; j < W.n_rows; j ++) {
    for(int k = 0; k < W.n_cols; k ++) {
      // sample from random walk centered at current value
      do N(j, k) = W(j, k) + s0 * randn(); while (N(j, k) <= 0);
      
      // calculate acceptance ratio
      vec oddsden = A * trans(W.row(j) % B.row(j)) + z(j);
      vec oddsnum = A * trans(N.row(j) % B.row(j)) + z(j);
      
      double prden = sum(oddsden % Z.col(j) - log(1 + exp(oddsden))) - s * W(j, k);
      double prnum = sum(oddsnum % Z.col(j) - log(1 + exp(oddsnum))) - s * N(j, k);
      
      if((prnum - prden) >= log(randu())) W = N; else N = W;
    }
  }
  
  return(W);
}

// [[Rcpp::export]]
vec updatez(mat Z, mat A, mat B, mat W, vec z, double s = 100, double s0 = 1) {
  for(int j = 0; j < z.n_elem; j ++) {
    // sample from random walk centered at current value
    double znew = z(j) + s0 * randn();
    
    // calculate acceptance ratio
    vec oddsden = A * trans(W.row(j) % B.row(j)) + z(j);
    vec oddsnum = A * trans(W.row(j) % B.row(j)) + znew;
    
    double prden = - pow(z(j), 2) / (2 * s) + sum(oddsden % Z.col(j) - log(1 + exp(oddsden)));
    double prnum = - pow(znew, 2) / (2 * s) + sum(oddsnum % Z.col(j) - log(1 + exp(oddsnum)));
    
    if((prnum - prden) >= log(randu())) z(j) = znew;
  }
  
  return(z);
}

// [[Rcpp::export]]
mat updateZ(mat X, mat A, mat B, mat W, vec z, vec u0, vec u1, vec v0, vec v1) {
  mat Z = zeros<mat>(X.n_rows, X.n_cols);
  
  for(int i = 0; i < X.n_rows; i ++) {
    for(int j = 0; j < X.n_cols; j ++) {
      // posterior 0 & 1 probability
      double odds = sum(A.row(i) % W.row(j) % B.row(j)) + z(j);
      
      double prob1 = odds + log(v1(j)) + (u1(j) * v1(j) - 1) * log(X(i, j)) + ((1 - u1(j)) * v1(j) - 1) * log(1 - X(i, j)) + lgamma(v1(j)) - lgamma(u1(j) * v1(j)) - lgamma((1 - u1(j)) * v1(j));
      double prob0 = log(v0(j)) + (u0(j) * v0(j) - 1) * log(X(i, j)) + ((1 - u0(j)) * v0(j) - 1) * log(1 - X(i, j)) + lgamma(v0(j)) - lgamma(u0(j) * v0(j)) - lgamma((1 - u0(j)) * v0(j));
      
      Z(i, j) = (prob0 - prob1) <= log(1 / randu() - 1);
    }
  }
  
  return(Z);
}

// [[Rcpp::export]]
vec updateu0(mat X, mat Z, vec u0, vec u1, vec v0, double a0 = 1, double b0 = 1) {
  double c0 = 1; double u0new;
  
  for(int j = 0; j < u0.n_elem; j ++) {
    do {
      double x = randg(distr_param(a0, c0));
      double y = randg(distr_param(b0, c0));
      
      // update by a MH step
      u0new = x / (x + y);
    } while (u0new > u1(j));
    
    double den = sum((1 - Z.col(j)) % ((u0(j) * v0(j) - 1) * log(X.col(j)) + ((1 - u0(j)) * v0(j) - 1) * log(1 - X.col(j)) - lgamma(u0(j) * v0(j)) - lgamma((1 - u0(j)) * v0(j))));
    double num = sum((1 - Z.col(j)) % ((u0new * v0(j) - 1) * log(X.col(j)) + ((1 - u0new) * v0(j) - 1) * log(1 - X.col(j)) - lgamma(u0new * v0(j)) - lgamma((1 - u0new) * v0(j))));
    
    if((num - den) >= log(randu())) u0(j) = u0new;
  }
  
  return(u0);
}

// [[Rcpp::export]]
vec updateu1(mat X, mat Z, vec u0, vec u1, vec v1, double a0 = 1, double b0 = 1) {
  double c0 = 1; double u1new;
  
  for(int j = 0; j < u1.n_elem; j ++) {
    do {
      double x = randg(distr_param(a0, c0));
      double y = randg(distr_param(b0, c0));
      
      // update by a MH step
      u1new = x / (x + y);
    } while (u1new < u0(j));
    
    double den = sum(Z.col(j) % ((u1(j) * v1(j) - 1) * log(X.col(j)) + ((1 - u1(j)) * v1(j) - 1) * log(1 - X.col(j)) - lgamma(u1(j) * v1(j)) - lgamma((1 - u1(j)) * v1(j))));
    double num = sum(Z.col(j) % ((u1new * v1(j) - 1) * log(X.col(j)) + ((1 - u1new) * v1(j) - 1) * log(1 - X.col(j)) - lgamma(u1new * v1(j)) - lgamma((1 - u1new) * v1(j))));
    
    if((num - den) >= log(randu())) u1(j) = u1new;
  }
  
  return(u1);
}

// [[Rcpp::export]]
vec updatev0(mat X, mat Z, vec u0, vec v0, double a0 = 1, double b0 = 0.1) {
  for(int j = 0; j < v0.n_elem; j ++) {
    // update by a MH step
    double v0new = randg(distr_param(a0, 1 / b0));
    
    double den = sum((1 - Z.col(j)) % (log(v0(j)) + (u0(j) * v0(j) - 1) * log(X.col(j)) + ((1 - u0(j)) * v0(j) - 1) * log(1 - X.col(j)) + lgamma(v0(j)) - lgamma(u0(j) * v0(j)) - lgamma((1 - u0(j)) * v0(j))));
    double num = sum((1 - Z.col(j)) % (log(v0new) + (u0(j) * v0new - 1) * log(X.col(j)) + ((1 - u0(j)) * v0new - 1) * log(1 - X.col(j)) + lgamma(v0new) - lgamma(u0(j) * v0new) - lgamma((1 - u0(j)) * v0new)));
    
    if((num - den) >= log(randu())) v0(j) = v0new;
  }
  
  return(v0);
}

// [[Rcpp::export]]
vec updatev1(mat X, mat Z, vec u1, vec v1, double a0 = 1, double b0 = 0.1) {
  for(int j = 0; j < v1.n_elem; j ++) {
    // update by a MH step
    double v1new = randg(distr_param(a0, 1 / b0));
    
    double den = sum(Z.col(j) % (log(v1(j)) + (u1(j) * v1(j) - 1) * log(X.col(j)) + ((1 - u1(j)) * v1(j) - 1) * log(1 - X.col(j)) + lgamma(v1(j)) - lgamma(u1(j) * v1(j)) - lgamma((1 - u1(j)) * v1(j))));
    double num = sum(Z.col(j) % (log(v1new) + (u1(j) * v1new - 1) * log(X.col(j)) + ((1 - u1(j)) * v1new - 1) * log(1 - X.col(j)) + lgamma(v1new) - lgamma(u1(j) * v1new) - lgamma((1 - u1(j)) * v1new)));
    
    if((num - den) >= log(randu())) v1(j) = v1new;
  }
  
  return(v1);
}