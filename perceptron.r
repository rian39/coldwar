
######################################################
#
# R implementation of a simple perceptron with
# visualization in 2D corresponding to exercise 1.3 
# and 1.4 of
# http://www.dbs.ifi.lmu.de/cms/Maschinelles_Lernen_und_Data_Mining_12
#
# programming language R available at
#    http://cran.r-project.org/
#
# available in the CIP pool by
# $ R
#
# This file was created by Marisa Petri (contact: 
# petri[aT}dbs.ifi.lmu.de)
#
######################################################

######################################################
#
# w   <- vector of weights = separating hyperplane
# col <- color of straight line
# lwd <- thickness of straight line
#
# Plots a seperating hyperplane w for two dimensions
# into an existing plot.
#
######################################################
plotMe <- function(w, col=1, lwd=1) {
  if (length(w) == 3) {
    if (w[3] == 0) {
      if (w[2] == 0) {
        abline(v=0, col=col, lwd=lwd);
      } else {
        abline(v=-w[1]/w[2], col=col, lwd=lwd);
      }
    } else if (is.infinite(w[2])) {
      abline(v=0, col=col, lwd=lwd);
    } else if (is.infinite(w[3])) {
      abline(h=0, col=col, lwd=lwd);
    } else {
      abline(-w[1]/w[3], -w[2]/w[3], col=col, lwd=lwd);
    }
  }
}

######################################################
#
# m   <- matrix of feature vectors (stored in columns)
# eta <- step size for each correction of a miss-
#          classified pattern
#
# Header for Plots resulting from trainPerceptron()
# and trainViaGradientDescent()
#
######################################################
plotHeader <- function(m, eta=0.1) {
  plot(m[1,], m[2,], xlab=expression(x[1]),
    ylab=expression(x[2]), xlim=range(-1,5),
    ylim=range(-1,5), col=ifelse(y==-1,2,3),
    main="Perceptron Iterations",
    sub=substitute(list(eta) == x, list(x=eta)))
  text(m[1,], m[2,], 1:dim(m)[2],
    col=ifelse(y==-1, 2, 3), pos=4)
  abline(h=0, lty=3)
  abline(v=0, lty=3)
}

######################################################
#
# x   <- miss-classified pattern to be inserted
# y   <- class labels for pattern x
# w   <- vector of weights
# eta <- step size for each correction of a miss-
#          classified pattern
#
# Performs one step of a pattern-based descent for the
# missclassified pattern x on the given weight vector 
# w and returns the adapted w.
#
######################################################
pStep <- function(x, y, w, eta=.1) {
  stopifnot(length(x) == length(w)-1);
  w[1] = w[1] + eta * y;
  for (j in 2:length(w)) {
    w[j] = w[j] + eta * y * x[j-1];
  }
  w;
}

######################################################
#
# x   <- an instance to be activated
# w   <- vector of weights
#
# Computes the activation function h of the perceptron
# represented by w.
#
######################################################
activate <- function(x, w) {
  sum(c(1, x) * w) # do not forget the bias offset
}

######################################################
#
# h   <- the activation value of the instance to be
#          classified
#
# Computes the class estimate for the given activation
# value h.
#
######################################################
classify <- function(h) {
  sign(h)
}

######################################################
#
# m   <- matrix of feature vectors (stored in columns)
# y   <- vector of class labels for feature vectors;
#          these must be in {-1, 1}
# w   <- initialization vector of weights; defaults to
#          the diagonal of the given input space + 1
# eta <- step size for each correction of a miss-
#          classified pattern
# ord <- order of repeated "random" pattern sampling
#          procedure; if == -1, uses a real random 
#          pattern selection; defaults to 1:dim(m)[2]
# plot <- boolean, if TRUE, each learning step's 
#          separating hyperplane w is visualized in the
#          first two dimensions
# maxIt <- savety threshold in case the perceptron
#          cannot be terminated within a reasonable
#          number of iterations < maxIt
#
# Pattern-based Perceptron implementation; returns the
# trained perceptron model w.
#
######################################################
trainPerceptron<-function(m, y, w=NULL, eta=.1,
                   ord=NULL, plot=TRUE, maxIt=1e4) {
  if (plot) {
    plotMe(w, 8, 3)
  }
  if (is.null(w)) { # default to diagonal
    w <- rep(1, dim(m)[1]+1)
  }
  random <- FALSE
  if (is.null(ord)) { # default linear order traversal
    ord <- 1:(dim(m)[2])
  } else if (ord[1] == -1) { # random traversals
    random <- TRUE
    ord <- sample(dim(m)[2], dim(m)[2])
  }
  numIt <- 0 # iteration counter
  while (1) { # until convergence
    wtemp <- w;
	num_errors <- 0
    for (i in ord) {
      h <- activate(m[,i], w)
      if (classify(h) == y[i]) {
        next # correctly classified
      }
	  num_errors <- num_errors + 1
      w <- pStep(m[,i], y[i], w, eta)
      numIt <- numIt + 1
      if (plot) {
        print (sprintf("%2d : % 3.3f,% 3.3f,% 3.3f",
              i, w[1], w[2], w[3]))
        plotMe(w, cols[numIt %% length(cols)])
      }
      if (random) { # re-shuffle after each adaption
        break
      }      
    }
    if (num_errors == 0) {
      break
    }
	if (!random && isTRUE(all.equal(wtemp, w))) {
	  # we need to check this, as this is not the
	  # classical perceptron
	  stop(paste("The given fixed order results in an endless loop after", numIt, "iterations"))
	}
	if (numIt == maxIt) {
	  stop(paste("Terminating after maxIt =",numIt,"iterations"))
	}
    if (random) { # re-shuffle
      ord <- sample(dim(m)[2], dim(m)[2])
    }
  }
  
  print(paste(numIt, "iterations"))
  if (plot) {
    plotMe(w, 1, 3)
  }
  w
}

######################################################
#
# Gradient Descent variant of trainPerceptron() - this
# is NOT the classical perceptron.
#
######################################################
trainViaGradientDescent <- function(m, y, w=NULL, 
                       eta=.1, plot=TRUE, maxIt=1e4) {
  if (plot) {
    plotMe(w,8,3)
  }
  if (is.null(w)) { # default to diagonal
    w <- rep(1, dim(m)[1]+1)
  }
  nullers <- rep(0,length(w)); # measure progress
  numIt <- 0
  k <- 0
  while (1) {
    change <- rep(0,length(w));
    for (i in 1:length(y)) {
      h <- activate(m[,i], w)
      if (classify(h) == y[i]) {
        next
      }
      change[1] = change[1] + y[i];
      for (j in 2:length(w)) {
        change[j] = change[j] + y[i] * m[j-1,i];
      }
      k <- k+1;
    }
    if (isTRUE(all.equal(change, nullers))) {
      break
    }
    numIt <- numIt+1;
	if (numIt == maxIt) {
	  stop(paste("Terminating after maxIt =",numIt,"iterations"))
	}
    w <- w + eta * change;
    if (plot) {
      print (sprintf("Iter %2d - %3d : % 3.3f,% 3.3f,% 3.3f",
                    numIt, k, w[1], w[2], w[3]))
      plotMe(w, cols[numIt %% length(cols)])
    }
  }
  print(paste(numIt, "iterations,", k, "used patterns"))
  if (plot) {
    plotMe(w, 1, 3)
  }
  w
}

######################################################
#
# Initialize the data of exercise 1-3
#
######################################################

examples <- function() {
    eta  <- .1
    cols <- rainbow(10) # colors for visualization
    w    <- c(0,1,-1)
    m    <- matrix(c(2,4, 1,.5, .5,1.5, 0,.5), nrow=2)
    y    <- c(1,1,-1,-1)


    # 1-3 a)
    plotHeader(m, eta)
    trainPerceptron(m, y, w, eta)

    # output:
    # [1] " 1 :  0.100, 1.200,-0.600"
    # [1] 1 iterations
    # [1]  0.1  1.2 -0.6

    #
    # 1-3 b)
    #
    # order {1,3,3,1,3,3}
#
eta  <-.25
cols <- rainbow(5)
cols <- c(cols,1)
plotHeader(m, eta)
trainPerceptron(m, y, w, eta)
legend("bottomright", legend=1:6, col=cols, lwd=c(rep(1,5),3))

# output:
# [1] " 1 :  0.250, 1.500, 0.000"
# [1] " 3 :  0.000, 1.375,-0.375"
# [1] " 3 : -0.250, 1.250,-0.750"
# [1] " 1 :  0.000, 1.750, 0.250"
# [1] " 3 : -0.250, 1.625,-0.125"
# [1] " 3 : -0.500, 1.500,-0.500"
# [1] 6 iterations
# [1] -0.5  1.5 -0.5

#
# order {1,4,3,2}
#
cols <- rainbow(2)
cols <- c(cols,1)
plotHeader(m, eta)
trainPerceptron(m, y, w, eta, c(1,4,3,2))
legend("bottomright", legend=1:3, col=cols, lwd=c(rep(1,2),3))

# output:
# [1] " 1 :  0.250, 1.500, 0.000"
# [1] " 4 :  0.000, 1.500,-0.125"
# [1] " 3 : -0.250, 1.375,-0.500"
# [1] 3 iterations
# [1] -0.250  1.375 -0.500

#
# random order (call multiple times)
#
plotHeader(m, eta)
trainPerceptron(m, y, w, eta, ord=-1)

#
# 1-3 c)
#
# eta .1
eta  <- .1
plotHeader(m, eta)
trainViaGradientDescent(m, y, w, eta)

# output:
# [1] "Iter  1 -   1 :  0.100, 1.200,-0.600"
# [1] "1 iterations, 1 used patterns"
# [1]  0.1  1.2 -0.6

# eta .25
eta  <- .25
plotHeader(m,eta)
trainViaGradientDescent(m,y,w,eta)

# output:
# [1] "Iter  1 -   1 :  0.250, 1.500, 0.000"
# [1] "Iter  2 -   3 : -0.250, 1.375,-0.500"
# [1] "2 iterations, 3 used patterns"
# [1] -0.250  1.375 -0.500


######################################################
#
# Exercise 1-4
#
######################################################

num <- matrix(c(
0,1,0,0,1,0,0,1,0,0,1,0,0,1,0, # 1
1,1,1,0,0,1,1,1,1,1,0,0,1,1,1, # 2
1,1,1,0,0,1,0,1,1,0,0,1,1,1,1, # 3
1,0,1,1,0,1,1,1,1,0,0,1,0,0,1, # 4
1,1,1,1,0,0,1,1,1,0,0,1,1,1,1, # 5
1,1,1,1,0,0,1,1,1,1,0,1,1,1,1, # 6
1,1,1,0,0,1,0,0,1,0,0,1,0,0,1, # 7
1,1,1,1,0,1,1,1,1,1,0,1,1,1,1, # 8
1,1,1,1,0,1,1,1,1,0,0,1,1,1,1, # 9
1,1,1,1,0,1,1,0,1,1,0,1,1,1,1  # 0
), ncol=15, byrow=TRUE)
# visualize numbers:
plotNumber <- function(number, x=num, dx=3, dy=5) {
  stopifnot(length(x[number,]) == dx * dy) # test dimensions
  # the matrix must be rotated by 90°; transpose causes the
  # image to be flipped, thus the flip transformation [dy:1,]
  image(t(matrix(x[number,],ncol=dx,byrow=T)[dy:1,]))
}
# visualize all in an array of images
previousPar <- par(mfrow=c(5,2), # configure the array
    mar = c(1, 1, 1, 1) + 0.1) # cut space around the edges
for (i in 1:10)
  plotNumber(i)
par(previousPar) # reset image device to default values

# self-made plotting method
plotNumber2 <- function(number, x=num, dx=3, dy=5) {
  stopifnot(length(x[number,]) == dx * dy) # test dimensions
  plot(rep(1:dx, dy), rep(dy:1, each=dx), type="p", 
    pch=ifelse(x[number,]==1,22,NA), cex=7, bg=1,
   axes=FALSE, xlab=NA, ylab=NA, asp=1)
  box()
}
previousPar <- par(mfrow=c(5,2), # configure the array
    mar = c(1, 1, 1, 1) + 0.1) # cut space around the edges
for (i in 1:10)
  plotNumber2(i)
par(previousPar) # reset image device to default values


y_orig <- c(1:9,0) # original class labels
y <- ifelse(y_orig %% 2 == 0,1,-1) # pair vs. impair
w <- rep(1,16)
eta <- .25

# train perceptron (with fixed testing order)
w_new <- trainPerceptron(t(num), y, w, eta, plot=FALSE)
# test result
w_new
for (i in 1:10) {
  print(sprintf("%2d is % d => % d", i, y[i], classify(activate(num[i,], w_new))))
}
# analyze w
matrix(w_new[2:16],ncol=3,byrow=T)


# random pattern order
w_new <- trainPerceptron(t(num), y, w, eta, -1, plot=FALSE)

# with varying w
w_new <- trainPerceptron(t(num), y, runif(16), eta, -1, plot=FALSE)
w_new <- trainPerceptron(t(num), y, sample(c(-100,1),16,replace=T), eta, -1, plot=FALSE)
w_new <- trainPerceptron(t(num), y, sample(c(-1,1),16,replace=T), eta, -1, plot=FALSE)
w_new <- trainPerceptron(t(num), y, 1:16, eta, -1, plot=FALSE)
w_new <- trainPerceptron(t(num), y, rep(-100,16), eta, -1, plot=FALSE)
w_new <- trainPerceptron(t(num), y, rep(-100,16), 100, -1, plot=FALSE)

# with varying eta
w_new <- trainPerceptron(t(num), y, w, runif(1), -1, plot=FALSE)

# gradient descent
w_new <- trainViaGradientDescent(t(num), y, w, eta, plot=FALSE)

y <- ifelse(y_orig %% 3 == 0,1,-1) # divisible by 3
w_new <- trainPerceptron(t(num), y, w, eta, plot=FALSE)



# visualize 2 classes (pair, impair)
cum_num <- apply(num[(1:5)*2,],2,sum) # first row: pair
cum_num <- rbind(cum_num, apply(num[(1:5)*2-1,],2,sum)) # 2nd: impair
# visualize both classes in an array of images
previousPar <- par(mfrow=c(2,1), # configure the array
    mar = c(1, 1, 1, 1) + 0.1) # cut space around the edges
for (i in 1:2)
  plotNumber(i, cum_num)
par(previousPar) # reset image device to default values

# visualize 2 classes (divides by 3, or does not)
cum_num <- apply(num[c(1:9,0) %% 3 == 0,],2,sum) # divides by 3
cum_num <- rbind(cum_num, apply(num[c(1:9,0) %% 3 != 0,],2,sum)) # does not
previousPar <- par(mfrow=c(2,1), # configure the array
    mar = c(1, 1, 1, 1) + 0.1) # cut space around the edges
for (i in 1:2)
  plotNumber(i, cum_num)
par(previousPar) # reset image device to default values
}
