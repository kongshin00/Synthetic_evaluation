# Generate simulation data
# Ten variables will be consisted of
# 1. Two non-negative continuous variables multivariate exponential distribution
# 2. Four count variables from multivaraite poisson distribution
# 3. One binary variable from Bernoulli distribution
# 4. Two categorical variables from multinomial distribution
# 5. one ordinal variable from ranked Gaussian distribution

library(LaplacesDemon) # for multivariate exponential 
library(fourPNO)
library(dplyr)


# Define function for multivariate Poisson.
.fixInf <- function(data) {
  # hacky way of replacing infinite values with the col max + 1
  if (any(is.infinite(data))) {
    data <-  apply(data, 2, function(x) {
      if (any(is.infinite(x))) {
        x[ind<-which(is.infinite(x))] <- NA
        x[ind] <- max(x, na.rm=TRUE)+1
      }
      x
    })
  }
  data
}

rmvpois <- function(n, mu, Sigma, ...) {
  Cor <- cov2cor(Sigma)
  SDs <- sqrt(diag(Sigma))
  d   <- length(SDs)
  if (length(mu) != length(SDs)) stop("Sigma and mu/lambdas dimensions don't match")
  if (length(mu) == 1) stop("Need more than 1 variable")
  normd  <- rmvnorm(n, rep(0, d), Cor)
  unif   <- pnorm(normd)
  data <- t(matrix(qpois(t(unif), mu, ...), d, n))
  data <- .fixInf(data)
  return(data)
}

# Define function that generate data with given parameters.

generate_data = function(n_obs,mve_mu, mve_sigma, mvp_mu_1, mvp_sigma_1, mvp_mu_2, mvp_sigma_2, Ber_p, multi_p1, multi_p2, G_mu, G_sigma){
  
  
  mve = t(t(abs(rmvl(n_obs, c(0,0), mve_sigma))) + mve_mu)
  mve[,1] = as.integer(mve[,1])
  mve[,2] = as.integer(mve[,2])
  
  mvp_1 = rmvpois(n_obs, mvp_mu_1, mvp_sigma_1)
  mvp_2 = rmvpois(n_obs, mvp_mu_2, mvp_sigma_2)
  
  Ber =  sample(c('M','F'), n_obs, replace = TRUE, prob = Ber_p)
  
  multi_1 =  sample(c('A','B','C','D'), n_obs, replace = TRUE, prob = multi_p1)
  multi_2 =  sample(c('IC','PO','AZ','ST','CB','EY','CJ'), n_obs, replace = TRUE, prob = multi_p2)
  
  Gaus = rnorm(n_obs, G_mu, G_sigma)
  
  data = cbind(mve, mvp_1,mvp_2, Ber, multi_1, multi_2, Gaus)
  
  return(data)
  
}


# Obtain number of observations for each cluster from the prior probability.

# Set parameters for Simulation data
n = 5000 # 10000, 50000, 100000
p = 5 # 5, 10, 15, 20

set.seed(42)
n_obs = rmultinom(1, n, c(0.2,0.5,0.3))   

# Set parameters for each cluster.
n_obs_a = n_obs[1]
n_obs_b = n_obs[2]
n_obs_c = n_obs[3]


p_a = list(mve_mu = c(0.75,25),
           mve_sigma = matrix(c(50,30,30,100),nrow = 2),
           
           mvp_mu_1 = c(1,1),
           mvp_sigma_1 = matrix(c(1,0.6,0.6,2),nrow = 2),
           
           mvp_mu_2 = c(1,1),
           mvp_sigma_2 = matrix(c(1,0.2,0.2,1),nrow = 2),
           
           Ber_p = c(0.2,0.8),
           
           multi_p1 = c(0.7,0.1,0.15,0.05),
           multi_p2 = c(0.2,0.1,0.3,0.1,0.1,0.1,0.1),
           
           G_mu = 80,
           G_sigma = 20)


p_b = list(mve_mu = c(1,15),
           mve_sigma = matrix(c(100,100,100,150),nrow = 2),
           
           mvp_mu_1 = c(1,1),
           mvp_sigma_1 = matrix(c(1,0.2,0.2,3),nrow = 2),
           
           mvp_mu_2 = c(1,3),
           mvp_sigma_2 = matrix(c(1,0.5,0.5,1),nrow = 2),
           
           Ber_p = c(0.4,0.6),
           
           multi_p1 = c(0.3,0.5,0.15,0.05),
           multi_p2 = c(0.1,0.2,0.2,0.05,0.15,0.2,0.1),
           
           G_mu = 100,
           G_sigma = 10)


p_c = list(mve_mu = c(4,15),
           mve_sigma = matrix(c(100,50,50,150),nrow = 2),
           
           mvp_mu_1 = c(2,1),
           mvp_sigma_1 = matrix(c(1,0.2,0.2,3),nrow = 2),
           
           mvp_mu_2 = c(3,2),
           mvp_sigma_2 = matrix(c(1,0.5,0.5,1),nrow = 2),
           
           Ber_p = c(0.55,0.45),
           
           multi_p1 = c(0.2,0.4,0.3,0.1),
           multi_p2 = c(0.05,0.4,0.2,0.05,0.15,0.05,0.1),
           
           G_mu = 130,
           G_sigma = 5)


# Generate data from each cluster with given parameters.
set.seed(42)

data_a = generate_data(n_obs_a,p_a$mve_mu, p_a$mve_sigma, p_a$mvp_mu_1, p_a$mvp_sigma_1, p_a$mvp_mu_2, p_a$mvp_sigma_2, 
                       p_a$Ber_p, p_a$multi_p1, p_a$multi_p2, p_a$G_mu, p_a$G_sigma)

data_b = generate_data(n_obs_b,p_b$mve_mu, p_b$mve_sigma, p_b$mvp_mu_1, p_b$mvp_sigma_1, p_b$mvp_mu_2, p_b$mvp_sigma_2, 
                       p_b$Ber_p, p_b$multi_p1, p_b$multi_p2, p_b$G_mu, p_b$G_sigma)

data_c = generate_data(n_obs_c,p_c$mve_mu, p_c$mve_sigma, p_c$mvp_mu_1, p_c$mvp_sigma_1, p_c$mvp_mu_2, p_c$mvp_sigma_2, 
                       p_c$Ber_p, p_c$multi_p1, p_c$multi_p2, p_c$G_mu, p_c$G_sigma)


#########################
# for p = 15, 20
if (p %in% c(15, 20)){
  data_d = generate_data(n_obs_a,p_a$mve_mu, p_a$mve_sigma, p_a$mvp_mu_1, p_a$mvp_sigma_1, p_a$mvp_mu_2, p_a$mvp_sigma_2, 
                         p_a$Ber_p, p_a$multi_p1, p_a$multi_p2, p_a$G_mu, p_a$G_sigma)
  
  data_e = generate_data(n_obs_b,p_b$mve_mu, p_b$mve_sigma, p_b$mvp_mu_1, p_b$mvp_sigma_1, p_b$mvp_mu_2, p_b$mvp_sigma_2, 
                         p_b$Ber_p, p_b$multi_p1, p_b$multi_p2, p_b$G_mu, p_b$G_sigma)
  
  data_f = generate_data(n_obs_c,p_c$mve_mu, p_c$mve_sigma, p_c$mvp_mu_1, p_c$mvp_sigma_1, p_c$mvp_mu_2, p_c$mvp_sigma_2, 
                         p_c$Ber_p, p_c$multi_p1, p_c$multi_p2, p_c$G_mu, p_c$G_sigma)
  
}

#########################

# Concatenate the observations to one dataset.
data = rbind(data_a, data_b, data_c)
data = data.frame(data)
# set each data type
data[,c(1,10)] = sapply(data[, c(1,10)], as.numeric)
data[,c(2,3,4,5,6)] = sapply(data[, c(2,3,4,5,6)], as.integer)

# Coerce Gaus variable to ordinal.
#perform binning with custom breaks
data = data %>% mutate(ordinal = cut(Gaus, breaks=c(-Inf, 50, 70, 100,120,140, Inf), labels = FALSE)) %>% select(- Gaus)

# Change variable order and permute row order.
data = data[,c(7,8,9,2,3,4,5,6,1,10)]
col = c()
for (i in 1:10){
  col = c(col,paste('Var',i, sep = ''))
}

colnames(data) = col

data = data[sample(nrow(data)),]

row.names(data) = NULL

#########################


# for p=15, 20 Concatenate the observations to one dataset.
if (p %in% c(15, 20)){
  data2 = rbind(data_d, data_e, data_f)
  data2 = data.frame(data2)
  # set each data type
  data2[,c(1,10)] = sapply(data2[, c(1,10)], as.numeric)
  data2[,c(2,3,4,5,6)] = sapply(data2[, c(2,3,4,5,6)], as.integer)
  
  # Coerce Gaus variable to ordinal.
  #perform binning with custom breaks
  data2 = data2 %>% mutate(ordinal = cut(Gaus, breaks=c(-Inf, 50, 70, 100,120,140, Inf), labels = FALSE)) %>% select(- Gaus)
  
  # Change variable order and permute row order.
  data2 = data2[,c(7,8,9,2,3,4,5,6,1,10)]
  col = c()
  for (i in 1:10){
    col = c(col,paste('Var',i+10, sep = ''))
  }
  
  colnames(data2) = col
  
  data2 = data2[sample(nrow(data2)),]
  
  row.names(data2) = NULL
  
  data = cbind(data, data2)
}

#########################
# Aggregation

colnames(data)

find_duplicate = function(data){
  duplicates = duplicated(data)
  n_duplicates = sum(duplicates)
  return(n_duplicates)
}

find_duplicate(data)

# for p = 5 or 15
if (p %in% c(5, 15)){
  data = data %>% select(-Var3, -Var5, -Var6, -Var7, -Var8)
}

find_duplicate(data_subset)
dim(data);dim(data_subset)


# save
n = format(n, scientific = FALSE)
write.csv(data, paste0("C:/Users/kongs/Rprojects/project_synthetic/cw_toy/toy_original_", n, "_", p, ".csv"), row.names=FALSE)



