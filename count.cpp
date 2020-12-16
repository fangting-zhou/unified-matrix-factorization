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
mat updateZ(mat X, mat A, mat B, mat W, vec z, vec p, vec l, vec s, vec x) {
  mat Z = zeros<mat>(X.n_rows, X.n_cols);
  
  for(int i = 0; i < X.n_rows; i ++) {
    for(int j = 0; j < X.n_cols; j ++) {
      // posterior 0 & 1 probability
      double odds = sum(A.row(i) % W.row(j) % B.row(j)) + z(j);
      
      double prob1 = odds + s(j) * log(x(j)) + X(i, j) * log(1 - x(j)) + lgamma(X(i, j) + s(j)) - lgamma(s(j)) - lgamma(X(i, j) + 1);
      double prob0 = (X(i, j) == 0) * log((1 - p(j)) + p(j) * exp(- l(j))) + (X(i, j) != 0) * (X(i, j) * log(l(j)) - l(j) - lgamma(X(i, j) + 1));
      
      Z(i, j) = (prob0 - prob1) <= log(1 / randu() - 1);
    }
  }
  
  return(Z);
}

// [[Rcpp::export]]
vec updatep(mat X, mat Z, vec p, vec l, double a0 = 10, double b0 = 1) {
  double c0 = 1;
  
  for(int j = 0; j < p.n_elem; j ++) {
    double x = randg(distr_param(a0, c0));
    double y = randg(distr_param(b0, c0));
    
    // update by a MH step
    double pnew = x / (x + y);
    
    double den = sum((1 - Z.col(j)) % ((X.col(j) == 0) * log((1 - p(j)) + p(j) * exp(- l(j))) + (X.col(j) != 0) % (X.col(j) * log(l(j)) - l(j) - lgamma(X.col(j) + 1))));
    double num = sum((1 - Z.col(j)) % ((X.col(j) == 0) * log((1 - pnew) + pnew * exp(- l(j))) + (X.col(j) != 0) % (X.col(j) * log(l(j)) - l(j) - lgamma(X.col(j) + 1))));
    
    if((num - den) >= log(randu())) p(j) = pnew;
  }
  
  return(p);
}

// [[Rcpp::export]]
vec updatel(mat X, mat Z, vec p, vec l, vec s, vec x, double a0 = 1, double b0 = 0.1) {
  for(int j = 0; j < l.n_elem; j ++) {
    // update by a MH step
    double lnew = randg(distr_param(a0, 1 / b0));
    
    if(lnew > s(j) * (1 - x(j)) / x(j)) lnew = s(j) * (1 - x(j)) / x(j);
    
    double den = sum((1 - Z.col(j)) % ((X.col(j) == 0) * log((1 - p(j)) + p(j) * exp(- l(j))) + (X.col(j) != 0) % (X.col(j) * log(l(j)) - l(j) - lgamma(X.col(j) + 1))));
    double num = sum((1 - Z.col(j)) % ((X.col(j) == 0) * log((1 - p(j)) + p(j) * exp(- lnew)) + (X.col(j) != 0) % (X.col(j) * log(lnew) - lnew - lgamma(X.col(j) + 1))));
    
    if((num - den) >= log(randu())) l(j) = lnew;
  }
  
  return(l);
}

// [[Rcpp::export]]
vec updates(mat X, mat Z, vec s, vec x, vec l, double a0 = 1, double b0 = 0.1) {
  for(int j = 0; j < s.n_elem; j ++) {
    // update by a MH step
    double snew = randg(distr_param(a0, 1 / b0));
    
    if(snew < l(j) * x(j) / (1 - x(j))) snew = l(j) * x(j) / (1 - x(j));
    
    double den = sum(Z.col(j) % (lgamma(X.col(j) + s(j)) - lgamma(s(j)) - lgamma(X.col(j) + 1) + s(j) * log(x(j)) + X.col(j) * log(1 - x(j))));
    double num = sum(Z.col(j) % (lgamma(X.col(j) + snew) - lgamma(snew) - lgamma(X.col(j) + 1) + snew * log(x(j)) + X.col(j) * log(1 - x(j))));
    
    if((num - den) >= log(randu())) s(j) = snew;
  }
  
  return(s);
}

// [[Rcpp::export]]
vec updatex(mat X, mat Z, vec s, vec l, double a0 = 1, double b0 = 1) {
  double c0 = 1; vec x = zeros<vec>(Z.n_cols);
  
  for(int j = 0; j < x.n_elem; j ++) {
    // sample from conjugate beta distribution
    double anew = a0 + s(j) * sum(Z.col(j));
    double bnew = b0 + sum(Z.col(j) % X.col(j));
    
    double z = randg(distr_param(anew, c0));
    double y = randg(distr_param(bnew, c0));
    
    x(j) = z / (z + y);
    
    if(x(j) > s(j) / (l(j) + s(j))) x(j) = s(j) / (l(j) + s(j));
  }
  
  return(x);
}