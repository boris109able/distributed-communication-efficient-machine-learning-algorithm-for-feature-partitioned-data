#include <Rcpp.h>
#include "omp.h"
#include<string>
#include<iostream>
#include<cmath>
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::export]]
NumericVector timesTwo(NumericVector x) {
  return x * 2;
}

// [[Rcpp::export]]
double phiEval(std::vector<double> xbeta, std::vector<double> Y, std::vector<double> beta, double regParam, std::string type) {
  double sum1 = 0;
  double sum2 = 0;
  int n = xbeta.size();
  int d = beta.size();
  int i, j;
  if (type.compare("SVM") == 0) {
    double tmp;
    for (j = 0; j < n; j++) {
      tmp = 1 - Y[j] * xbeta[j];
      if (tmp > 0) {
        sum1 += pow(tmp, 2);
      }
    }
    sum1 *= 0.5;
    for (i = 0; i < d; i++) {
      sum2 += fabs(beta[i]);
    }
    sum2 *= regParam;
  }
  
  if (type.compare("Logistic") == 0) {
    for (j = 0; j < n; j++) {
      sum1 += log(1 + exp(-Y[j] * xbeta[j]));
    }
    for (i = 0; i < d; i++) {
      sum2 += fabs(beta[i]);
    }
    sum2 *= regParam;
  }

  if (type.compare("Lasso") == 0) {
    for (j = 0; j < n; j++) {
      sum1 += pow(Y[j] - xbeta[j],2);
    }
    sum1 *= 0.5;
    for (i = 0; i < d; i++) {
      sum2 += fabs(beta[i]);
    }
    sum2 *= regParam;
  }
  
  return sum1 + sum2;
}

double norm1(std::vector<double> beta) {
  int d = beta.size();
  int i;
  double sum = 0;
  for (i = 0; i < d; i++) {
    sum += beta[i];
  }
  return sum;
}
// [[Rcpp::export]]
void computeGrad(NumericMatrix X, std::vector<double> Y, std::vector<double> xbeta, std::vector<double>& grad, std::string type) {
  int n = X.nrow();
  int d = X.ncol();
  int i, j;
  if (type.compare("SVM") == 0) {
    double tmp;
    std::fill(grad.begin(), grad.end(), 0);
    //std::cout<<"grad norm 1: "<<norm1(xbeta)<<" "<<X(0,1)<<" "<<X[n]<<std::endl;
    //notice X(i, j) is X[j * n + i] or X[i + j * n]
    for (j = 0; j < n; j++) {
      tmp = 1 - Y[j] * xbeta[j];
      if (tmp > 0) {
        for (i = 0; i < d; i++) {
          grad[i] += -Y[j] * tmp * X[j + i * n];
        }
      }
    }
  }
  if (type.compare("Logistic") == 0) {
    std::fill(grad.begin(), grad.end(), 0);
    double scalar;
    for (j = 0; j < n; j++) {
      scalar = - Y[j] /(1 + exp(Y[j] * xbeta[j]));
      for (i = 0; i < d; i++) {
        grad[i] += scalar * X[j + i * n];
      }
    }
  }
  if (type.compare("Lasso") == 0) {
    std::fill(grad.begin(), grad.end(), 0);
    for (i = 0; i < d; i++) {
      for (j = 0; j < n; j++) {
        grad[i] += X[j + i * n] * (xbeta[j] - Y[j]);
      }
    }
  }
}

// [[Rcpp::export]]
std::vector<double> gradTest(NumericMatrix X, std::vector<double> Y, std::vector<double> beta, std::string type) {
  int n = X.nrow();
  int d = X.ncol();
  std::vector<double> grad(d, 0);
  std::vector<double> xbeta(n, 0);
  int i, j;
  for (j = 0; j < n; j++) {
    for (i = 0; i < d; i++) {
      xbeta[j] += X[i * n + j] * beta[i];
    }
  }
  //std::cout<<"xbeta norm 1: "<<norm1(xbeta)<<std::endl;
  computeGrad(X, Y, xbeta, grad, type);
  return grad;
}
// [[Rcpp::export]]
double SquaredFrobeniusNorm(NumericMatrix X) {
  int n = X.nrow();
  int d = X.ncol();
  int i, j;
  double sum = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++) {
      sum += pow(X[i + j * n], 2);
    }
  }
  return sum;
}

// [[Rcpp::export]]
List minL1_synthesize(NumericMatrix X, std::vector<double> Y, double regParam, double lambdaMax, double lambdaMin, std::string type, double epsOuter = 0.001, int loopOuter = 1000, bool loopForever = false) {
  double L = 2 * lambdaMax;
  if (type.compare("Logistic") == 0) {
    L = lambdaMax / 2;
  }
  int n = X.nrow();
  int d = X.ncol();
  std::vector<double> betaOld(d, 0);
  std::vector<double> betaNew(d, 0);
  std::vector<double> grad(d, 0);
  std::vector<double> xbeta(n, 0);
  double phiOld = phiEval(xbeta, Y, betaOld, regParam, type);
  double phiNew = 0;
  double decrease = 0;
  std::vector<double> phiSequence;
  phiSequence.push_back(phiOld);
  int i = 1;
  
  //setting up helper variables
  int l, j;
  double tmp, b;
  while (loopForever || i <= loopOuter) {
    computeGrad(X, Y, xbeta, grad, type);
    for (l = 0; l < d; l++) {
      //update betaNew
      tmp = betaOld[l] - grad[l] / L;
      b = regParam / L;
      if (tmp > b) {
        betaNew[l] = tmp - b;
      }
      else if (tmp < -b) {
        betaNew[l] = tmp + b;
      }
      else {
        betaNew[l] = 0;
      }
    }
    //Caution: do not forget to update xbeta
    for (j = 0; j < n; j++) {
      xbeta[j] = 0;
      for (l = 0; l < d; l++) {
        xbeta[j] += X[j + l * n] * betaNew[l];
      }
    }
    //calculate new objective function value
    phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
    phiSequence.push_back(phiNew);
    decrease = phiOld - phiNew;
    if (fabs(decrease) <= epsOuter) {
      break;
    }
    phiOld = phiNew;
    betaOld = betaNew;
    i++;
    //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
  }
  List result;
  result["i"] = i;
  result["phiNew"] = phiNew;
  result["phiSequence"] = phiSequence;
  result["beta"] = betaNew;
  return result;
}
// [[Rcpp::export]]
double dist(std::vector<double> v1, std::vector<double> v2) {
  int n = v1.size();
  int i;
  double sum = 0;
  for (i = 0; i < n; i++) {
    sum += pow(v1[i] - v2[i], 2);
  }
  return sqrt(sum);
}

double pow2(double a, double b) {
  return pow(a, b);
}
// [[Rcpp::export]]
List minL1_fast_synthesize(NumericMatrix X, std::vector<double> Y, double regParam, double lambdaMax, double lambdaMin, std::string type, double epsOuter = 0.001, int loopOuter = 1000, bool loopForever = true) {
  double L = 2 * lambdaMax;
  double mu = 2 * lambdaMin;
  if (type.compare("Logistic") == 0) {
    L = lambdaMax / 2;
    mu = lambdaMin / 2;
  }
  double mu_0 = mu / 2;
  int n = X.nrow();
  int d = X.ncol();
  std::vector<double> betaOld(d, 0);
  std::vector<double> betaNew(d, 0);
  std::vector<double> grad(d, 0);
  std::vector<double> xbeta(n, 0);
  std::vector<double> betaFixed(d, 0);
  double phiOld = phiEval(xbeta, Y, betaOld, regParam, type);
  double phiNew = 0;
  double decrease = 0;
  std::vector<double> phiSequence;
  phiSequence.push_back(phiOld);
  int i = 1;
  int k = 1;
  double D = sqrt(2 * d + sqrt(d) / 2);
  
  //setting up helper variables
  int l, j;
  double tmp, b;
  int firstFewRoundExact = 5;
  
  while (loopForever || i <= loopOuter) {
    if (dist(betaOld, betaNew) > (mu_0 / L) * pow(1 - mu_0 / L, i) * D || i <= firstFewRoundExact) {
      betaOld = betaNew;
      
      computeGrad(X, Y, xbeta, grad, type);
      for (l = 0; l < d; l++) {
        //update betaNew
        tmp = betaOld[l] - grad[l] / L;
        b = regParam / L;
        if (tmp > b) {
          betaNew[l] = tmp - b;
        }
        else if (tmp < -b) {
          betaNew[l] = tmp + b;
        }
        else {
          betaNew[l] = 0;
        }
      }
      //Caution: do not forget to update xbeta
      for (j = 0; j < n; j++) {
        xbeta[j] = 0;
        for (l = 0; l < d; l++) {
          xbeta[j] += X[j + l * n] * betaNew[l];
        }
      }
      //calculate new objective function value
      phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
      phiSequence.push_back(phiNew);
      decrease = phiOld - phiNew;
      if (fabs(decrease) <= epsOuter) {
        break;
      }
      phiOld = phiNew;
      i++;
      //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
    }
    else {
      betaFixed = betaOld;
      while (dist(betaNew, betaFixed) <= (mu_0 / L) * pow(1 - mu_0 / L, i) * D) {
        betaOld = betaNew;
      
        for (l = 0; l < d; l++) {
          //update betaNew
          tmp = betaOld[l] - grad[l] / L;
          b = regParam / L;
          if (tmp > b) {
            betaNew[l] = tmp - b;
          }
          else if (tmp < -b) {
            betaNew[l] = tmp + b;
          }
          else {
            betaNew[l] = 0;
          }
        }
        
        //Caution: do not forget to update xbeta
        for (j = 0; j < n; j++) {
          xbeta[j] = 0;
          for (l = 0; l < d; l++) {
            xbeta[j] += X[j + l * n] * betaNew[l];
          }
        }
        
        //calculate new objective function value
        phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
        phiSequence.push_back(phiNew);
        decrease = phiOld - phiNew;
        if (fabs(decrease) <= epsOuter) {
          break;
        }
        phiOld = phiNew;
        i++;
        k++;
        //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
      }
      if (fabs(decrease) <= epsOuter) {
        break;
      }
    }
  }
  List result;
  result["i"] = i;
  result["k"] = k;
  result["phiNew"] = phiNew;
  result["phiSequence"] = phiSequence;
  result["beta"] = betaNew;
  return result;
}

// [[Rcpp::export]]
List minL1_fast_real(NumericMatrix X, std::vector<double> Y, double regParam, double alpha, double D, std::string type, double epsOuter = 0.001, int loopOuter = 1000, bool loopForever = true, double eval = -1) {
  double fnorm = SquaredFrobeniusNorm(X);
  double L = 2 * fnorm;
  if (type.compare("Logistic") == 0) {
    L = fnorm / 2;
  }
  //std::cout<< "L: "<<L<<std::endl;
  int n = X.nrow();
  int d = X.ncol();
  std::vector<double> betaOld(d, 0);
  std::vector<double> betaNew(d, 0);
  std::vector<double> grad(d, 0);
  std::vector<double> xbeta(n, 0);
  std::vector<double> betaFixed(d, 0);
  double phiOld = phiEval(xbeta, Y, betaOld, regParam, type);
  double phiNew = 0;
  double decrease = 0;
  std::vector<double> phiSequence;
  phiSequence.push_back(phiOld);
  int i = 1;
  int k = 1;
  
  //setting up helper variables
  int l, j;
  double tmp, b;
  int firstFewRoundExact = 5;
  
  while (loopForever || i <= loopOuter) {
    if (dist(betaOld, betaNew) > D * pow(alpha, k) || i <= firstFewRoundExact) {
      betaOld = betaNew;
      
      computeGrad(X, Y, xbeta, grad, type);
      for (l = 0; l < d; l++) {
        //update betaNew
        tmp = betaOld[l] - grad[l] / L;
        b = regParam / L;
        if (tmp > b) {
          betaNew[l] = tmp - b;
        }
        else if (tmp < -b) {
          betaNew[l] = tmp + b;
        }
        else {
          betaNew[l] = 0;
        }
      }
      //Caution: do not forget to update xbeta
      for (j = 0; j < n; j++) {
        xbeta[j] = 0;
        for (l = 0; l < d; l++) {
          xbeta[j] += X[j + l * n] * betaNew[l];
        }
      }
      //calculate new objective function value
      phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
      phiSequence.push_back(phiNew);
      decrease = phiOld - phiNew;
      if (eval <= 0) {
        if (fabs(decrease) <= epsOuter) {
          break;
        }
      }
      else {
        if (phiNew < eval) {
          break;
        }
      }
      phiOld = phiNew;
      i++;
      //if (i % 100 == 0) {
      //  std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<"\t"<<D * pow(alpha, i)<<std::endl;
      //}
      //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
    }
    else {
      betaFixed = betaOld;
      while (dist(betaNew, betaFixed) <= D * pow(alpha, k)) {
        betaOld = betaNew;
        
        for (l = 0; l < d; l++) {
          //update betaNew
          tmp = betaOld[l] - grad[l] / L;
          b = regParam / L;
          if (tmp > b) {
            betaNew[l] = tmp - b;
          }
          else if (tmp < -b) {
            betaNew[l] = tmp + b;
          }
          else {
            betaNew[l] = 0;
          }
        }
        
        //Caution: do not forget to update xbeta
        for (j = 0; j < n; j++) {
          xbeta[j] = 0;
          for (l = 0; l < d; l++) {
            xbeta[j] += X[j + l * n] * betaNew[l];
          }
        }
        
        //calculate new objective function value
        phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
        phiSequence.push_back(phiNew);
        decrease = phiOld - phiNew;
        if (eval <= 0) {
          if (fabs(decrease) <= epsOuter) {
            break;
          }
        }
        else {
          if (phiNew < eval) {
            break;
          }
        }
        phiOld = phiNew;
        i++;
        k++;
        //if (i % 100 == 0) {
        //  std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<"\t"<<D * pow(alpha, i)<<std::endl;
        //}
        //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
      }
      if (eval <= 0) {
        if (fabs(decrease) <= epsOuter) {
          break;
        }
      }
      else {
        if (phiNew < eval) {
          break;
        }
      }
    }
  }
  List result;
  result["i"] = i;
  result["k"] = k;
  result["phiNew"] = phiNew;
  result["phiSequence"] = phiSequence;
  result["beta"] = betaNew;
  return result;
}

// [[Rcpp::export]]
List minL1_standard_real(NumericMatrix X, std::vector<double> Y, double regParam, std::string type, double epsOuter = 0.001, int loopOuter = 1000, bool loopForever = true) {
  double fnorm = SquaredFrobeniusNorm(X);
  double L = 2 * fnorm;
  if (type.compare("Logistic") == 0) {
    L = fnorm / 2;
  }
  //std::cout<< "L: "<<L<<std::endl;
  int n = X.nrow();
  int d = X.ncol();
  std::vector<double> betaOld(d, 0);
  std::vector<double> betaNew(d, 0);
  std::vector<double> grad(d, 0);
  std::vector<double> xbeta(n, 0);
  double phiOld = phiEval(xbeta, Y, betaOld, regParam, type);
  double phiNew = 0;
  double decrease = 0;
  std::vector<double> phiSequence;
  phiSequence.push_back(phiOld);
  int i = 1;
  
  //setting up helper variables
  int l, j;
  double tmp, b;
  
  while (loopForever || i <= loopOuter) {
      betaOld = betaNew;
      computeGrad(X, Y, xbeta, grad, type);
      for (l = 0; l < d; l++) {
        //update betaNew
        tmp = betaOld[l] - grad[l] / L;
        b = regParam / L;
        if (tmp > b) {
          betaNew[l] = tmp - b;
        }
        else if (tmp < -b) {
          betaNew[l] = tmp + b;
        }
        else {
          betaNew[l] = 0;
        }
      }
      //Caution: do not forget to update xbeta
      for (j = 0; j < n; j++) {
        xbeta[j] = 0;
        for (l = 0; l < d; l++) {
          xbeta[j] += X[j + l * n] * betaNew[l];
        }
      }
      //calculate new objective function value
      phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
      phiSequence.push_back(phiNew);
      decrease = phiOld - phiNew;
      if (fabs(decrease) <= epsOuter) {
        break;
      }
      phiOld = phiNew;
      i++;
      //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
  }
  List result;
  result["i"] = i;
  result["phiNew"] = phiNew;
  result["phiSequence"] = phiSequence;
  result["beta"] = betaNew;
  return result;
}

// [[Rcpp::export]]
List minL1_fast_synthesize_lower(NumericMatrix X, std::vector<double> Y, double regParam, double lambdaMax, double lambdaMin, std::string type, double lower = 100) {
  double L = 2 * lambdaMax;
  double mu = 2 * lambdaMin;
  if (type.compare("Logistic") == 0) {
    L = lambdaMax / 2;
    mu = lambdaMin / 2;
  }
  double mu_0 = mu / 2;
  int n = X.nrow();
  int d = X.ncol();
  std::vector<double> betaOld(d, 0);
  std::vector<double> betaNew(d, 0);
  std::vector<double> grad(d, 0);
  std::vector<double> xbeta(n, 0);
  std::vector<double> betaFixed(d, 0);
  double phiOld = phiEval(xbeta, Y, betaOld, regParam, type);
  double phiNew = 0;
  std::vector<double> phiSequence;
  phiSequence.push_back(phiOld);
  int i = 1;
  int k = 1;
  double D = sqrt(2 * d + sqrt(d) / 2);
  
  //setting up helper variables
  int l, j;
  double tmp, b;
  int firstFewRoundExact = 5;
  
  while (1) {
    if (dist(betaOld, betaNew) > (mu_0 / L) * pow(1 - mu_0 / L, i) * D || i <= firstFewRoundExact) {
      betaOld = betaNew;
      
      computeGrad(X, Y, xbeta, grad, type);
      for (l = 0; l < d; l++) {
        //update betaNew
        tmp = betaOld[l] - grad[l] / L;
        b = regParam / L;
        if (tmp > b) {
          betaNew[l] = tmp - b;
        }
        else if (tmp < -b) {
          betaNew[l] = tmp + b;
        }
        else {
          betaNew[l] = 0;
        }
      }
      //Caution: do not forget to update xbeta
      for (j = 0; j < n; j++) {
        xbeta[j] = 0;
        for (l = 0; l < d; l++) {
          xbeta[j] += X[j + l * n] * betaNew[l];
        }
      }
      //calculate new objective function value
      phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
      phiSequence.push_back(phiNew);
      if (phiNew <= lower) {
        break;
      }
      phiOld = phiNew;
      i++;
      //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
    }
    else {
      betaFixed = betaOld;
      while (dist(betaNew, betaFixed) <= (mu_0 / L) * pow(1 - mu_0 / L, i) * D) {
        betaOld = betaNew;
        
        for (l = 0; l < d; l++) {
          //update betaNew
          tmp = betaOld[l] - grad[l] / L;
          b = regParam / L;
          if (tmp > b) {
            betaNew[l] = tmp - b;
          }
          else if (tmp < -b) {
            betaNew[l] = tmp + b;
          }
          else {
            betaNew[l] = 0;
          }
        }
        
        //Caution: do not forget to update xbeta
        for (j = 0; j < n; j++) {
          xbeta[j] = 0;
          for (l = 0; l < d; l++) {
            xbeta[j] += X[j + l * n] * betaNew[l];
          }
        }
        
        //calculate new objective function value
        phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
        phiSequence.push_back(phiNew);
        if (phiNew <= lower) {
          break;
        }
        phiOld = phiNew;
        i++;
        k++;
        //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
      }
      if (phiNew <= lower) {
        break;
      }
    }
  }
  List result;
  result["i"] = i;
  result["k"] = k;
  result["phiNew"] = phiNew;
  result["phiSequence"] = phiSequence;
  result["beta"] = betaNew;
  return result;
}

// [[Rcpp::export]]
List minL1_fast_real_DrXu(NumericMatrix X, std::vector<double> Y, double regParam, double alpha, double D, std::string type, int M = 20, double epsOuter = 0.001, int loopOuter = 1000, bool loopForever = true, double eval = -1) {
  //std::cout<<"eval: "<<eval<<std::endl;
  double fnorm = SquaredFrobeniusNorm(X);
  double L = 2 * fnorm;
  if (type.compare("Logistic") == 0) {
    L = fnorm / 2;
  }
  //std::cout<< "L: "<<L<<std::endl;
  int n = X.nrow();
  int d = X.ncol();
  std::vector< std::vector<double> > interval;
  int k2 = d % M;
  int k1 = d / M;
  int ktmp;
  
  for (ktmp = 0; ktmp < M; ktmp++) {
    std::vector<double> each;
    int len = ktmp < k2 ? k1 + 1: k1;
    if (ktmp == 0) {
      each.push_back(0);
      each.push_back(len);
    }
    else {
      each.push_back(interval[ktmp - 1][1]);
      each.push_back(each[0] + len);
    }
    interval.push_back(each);
  }
  
  // for (ktmp = 0; ktmp < M; ktmp++) {
  //   std::cout<<interval[ktmp][0] << " " << interval[ktmp][1]<<std::endl;
  // }
  // 
  // if (ktmp == M) {
  //   List result;
  //   return result;
  // }
  
  std::vector<double> betaOld(d, 0);
  std::vector<double> betaNew(d, 0);
  std::vector<double> grad(d, 0);
  std::vector<double> xbeta(n, 0);
  std::vector<double> betaFixed(d, 0);
  std::vector<double> xbetaFixed;
  std::vector< std::vector<double> > eachxbeta;
  double phiOld = phiEval(xbeta, Y, betaOld, regParam, type);
  double phiNew = 0;
  double decrease = 0;
  std::vector<double> phiSequence;
  phiSequence.push_back(phiOld);
  int i = 1;
  int k = 1;
  
  //setting up helper variables
  int l, j, machine;
  double tmp, b;
  int firstFewRoundExact = 5;
  
  while (loopForever || i <= loopOuter) {
    if (dist(betaOld, betaNew) > D * pow(alpha, k) || i <= firstFewRoundExact) {
      betaOld = betaNew;
      
      computeGrad(X, Y, xbeta, grad, type);
      for (l = 0; l < d; l++) {
        //update betaNew
        tmp = betaOld[l] - grad[l] / L;
        b = regParam / L;
        if (tmp > b) {
          betaNew[l] = tmp - b;
        }
        else if (tmp < -b) {
          betaNew[l] = tmp + b;
        }
        else {
          betaNew[l] = 0;
        }
      }
      //Caution: do not forget to update xbeta
      for (j = 0; j < n; j++) {
        xbeta[j] = 0;
        for (l = 0; l < d; l++) {
          xbeta[j] += X[j + l * n] * betaNew[l];
        }
      }
      //calculate new objective function value
      phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
      phiSequence.push_back(phiNew);
      decrease = phiOld - phiNew;
      if (eval <= 0) {
        if (fabs(decrease) <= epsOuter) {
          break;
        }
      }
      else {
        if (phiNew <= eval) {
          break;
        }
      }
      
      phiOld = phiNew;
      i++;
      //if (i % 100 == 0) {
      //  std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<"\t"<<D * pow(alpha, i)<<std::endl;
      //}
      //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
    }
    else {
      betaFixed = betaOld;
      xbetaFixed = xbeta;
      while (dist(betaNew, betaFixed) <= D * pow(alpha, k)) {
        betaOld = betaNew;
        
        for (machine = 0; machine < M; machine++) {
          xbeta = xbetaFixed;
          for (j = 0; j < n; j++) {
            for (l = interval[machine][0]; l < interval[machine][1]; l++) {
              xbeta[j] += X[j + l * n] * betaNew[l];
            }
          }
          
          computeGrad(X, Y, xbeta, grad, type);
          for (l = interval[machine][0]; l < interval[machine][1]; l++) {
            //update betaNew
            tmp = betaOld[l] - grad[l] / L;
            b = regParam / L;
            if (tmp > b) {
              betaNew[l] = tmp - b;
            }
            else if (tmp < -b) {
              betaNew[l] = tmp + b;
            }
            else {
              betaNew[l] = 0;
            }
          }
        }
      
        //Caution: do not forget to update xbeta
        for (j = 0; j < n; j++) {
          xbeta[j] = 0;
          for (l = 0; l < d; l++) {
            xbeta[j] += X[j + l * n] * betaNew[l];
          }
        }
        
        //calculate new objective function value
        phiNew = phiEval(xbeta, Y, betaNew, regParam, type);
        phiSequence.push_back(phiNew);
        decrease = phiOld - phiNew;
        if (eval <= 0) {
          if (fabs(decrease) <= epsOuter) {
            break;
          }
        }
        else {
          if (phiNew <= eval) {
            break;
          }
        }
        phiOld = phiNew;
        i++;
        k++;
        //if (i % 100 == 0) {
        //  std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<"\t"<<D * pow(alpha, i)<<std::endl;
        //}
        //std::cout<<i<<"\t"<<phiNew<<"\t"<<"decrease: "<<decrease<<std::endl;
      }
      if (eval <= 0) {
        if (fabs(decrease) <= epsOuter) {
          break;
        }
      }
      else {
        if (phiNew <= eval) {
          break;
        }
      }
    }
  }
  List result;
  result["i"] = i;
  result["k"] = k;
  result["phiNew"] = phiNew;
  result["phiSequence"] = phiSequence;
  result["beta"] = betaNew;
  return result;
}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
timesTwo(42)
*/
