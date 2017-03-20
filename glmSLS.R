rm(list = ls())

# Scaled Least Squares(SLS) Estimator. NIPS 2016.
SLS <- function(y, X, subset_size, g_deriv2, g_deriv3,
                c_init, tol = 1e-7) {
  n <- nrow(X)
  p <- ncol(X)
  X <- as.matrix(X)
  if (missing(subset_size)) subset_size <- 100 * p * log(p)
  if (missing(c_init)) c_init <- 2.0 / var(y)
  
  #Compute the least square estimates beta_ols and y_ols
  subset_index <- sample(n, subset_size)
  Xs <- X[subset_index, ]
  beta_ols <- subset_size / n * solve(t(Xs) %*% Xs, t(X) %*% y)
  y_ols <- X %*% beta_ols
  
  #Recursive root finding
  c_old <- Inf
  c_new <- c_init
  while (abs(c_new - c_old) > tol) {
    c_old <- c_new
    c_y <- c_old * y_ols 
    d2 <- g_deriv2(c_y)
    d3 <- g_deriv3(c_y)
    
    numer <- c_old / n * sum(d2) - 1.0
    denom <- 1.0 / n * sum(d2 + c_y * d3)  #seems to have a typo in the paper.
    
    c_new <- c_old - numer / denom
    cat("Old:", c_old, "New:", c_new, "\n")
  }
  
  beta_sls <- c_new * beta_ols
  beta_sls
}


# glmnet helper function
cv.glmnet_coef <- function(...) {
  require(glmnet)
  glmnet_fit <- cv.glmnet(...)
  return(coef(glmnet_fit, s = "lambda.min"))
}
glmnet_coef <- function(...) {
  require(glmnet)
  glmnet_fit <- glmnet(...)
  return(coef(glmnet_fit, s = min(glmnet_fit$lambda)))
}


# Minimal example - Poisson regression
#devtools::install_github("kcf-jackson/glmSimData")
library(glmSimData)
library(glmnet)
X <- generate_independent_covariates(10000, 10)
true_beta <- rnorm(ncol(X), sd = 0.5)
my_data <- generate_response(X, beta = true_beta, family = poisson())

beta_SLS <- SLS(my_data$resp_var, X, g_deriv2 = exp, g_deriv3 = exp, subset_size = 3000)
beta_GLM <- glm(resp_var ~ . - 1, data = my_data, family = poisson())$coefficients
beta_GLMNET <- glmnet_coef(x = as.matrix(X), y = as.matrix(my_data$resp_var),
                           family = "poisson", intercept = FALSE)
beta_GLMNET <- beta_GLMNET[-1]  #dropping intercept
data.frame(SLS = beta_SLS, GLM = beta_GLM, GLMNET = beta_GLMNET, true = true_beta)


# Speed comparison
X <- generate_independent_covariates(100000, 100)
true_beta <- rnorm(ncol(X), sd = 0.02)
my_data <- generate_response(X, beta = true_beta, family = poisson())

system.time(
  beta_SLS <- SLS(my_data$resp_var, X, g_deriv2 = exp, g_deriv3 = exp, 
                  subset_size = 3000)
)
system.time(
  beta_GLM <- glm(resp_var ~ . - 1, data = my_data, family = poisson())$coefficients
)
system.time(
  beta_GLMNET <- glmnet_coef(x = as.matrix(X), y = my_data$resp_var,
                             family = "poisson", intercept = FALSE)[-1]
)
system.time(
  beta_GLMNET_cv <- cv.glmnet_coef(
    x = as.matrix(X), y = my_data$resp_var,
    family = "poisson", intercept = FALSE
  )[-1]
)


# Accuracy comparison
round(data.frame(SLS = beta_SLS, GLM = beta_GLM, 
                 GLMNET_cv = beta_GLMNET_cv, GLMNET = beta_GLMNET,
                 true = true_beta), 4)
SLS_rel <- mean(abs((beta_SLS - true_beta) / true_beta))
GLM_rel <- mean(abs((beta_GLM - true_beta) / true_beta))
GLMNET_cv_rel <- mean(abs((beta_GLMNET_cv - true_beta) / true_beta))
GLMNET_rel <- mean(abs((beta_GLMNET - true_beta) / true_beta))
cat(" SLS relative error:", SLS_rel,
    "GLM relative error:", GLM_rel, "\n",
    "GLMNET_cv relative error:", GLMNET_cv_rel,
    "GLMNET relative error:", GLMNET_rel, "\n")


# Notes / References:
# 1. Paper link: https://papers.nips.cc/paper/6522-scaled-least-squares-estimator-for-glms-in-large-scale-problems
# 2. For logistic regression, g(w) = log(1 + exp(w))
# deriv(w) = exp(w) / (1 + exp(w))
# deriv2(w) = exp(w) / (1 + exp(w))^2
# deriv3(w) = (1 - exp(w)) / (1 + exp(w))^2
# 3. glm function would take a long time if one sets n = 100000 and p = 1000.
