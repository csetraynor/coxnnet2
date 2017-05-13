library(tensorflow)

#' Calculate C-index of a model
#' @param pred Prediction from model
#' @param time Event time
#' @param status Censoring status
#' @export
#' @examples
#' cindex()
cindex <- function(pred, time, status) {
    wh <- which(status==1)
    time_mat <- outer(time, time[wh], ">")
    pred_mat <- outer(pred, pred[wh], "<")
    pred_mat_eq <- outer(pred, pred[wh], "==")
    total <- sum(time_mat)
    concord <- sum(time_mat & pred_mat) + sum(time_mat & pred_mat_eq)*0.5
    concord/total
}

#' Hidden Llayer helper function
#' @export
#' @examples
#' HiddenLayer()
HiddenLayer <- function(input, input_eval, input_size, output_size, dropout, activation=tf$tanh, seed=1, save_state=NULL) {
    set.seed(seed)
    
    W_init <- if(is.null(save_state)) {
        wt_lim <- 4 * sqrt(6/(input_size + output_size))
        tf$random_uniform(shape(input_size,output_size),minval=-wt_lim, maxval=wt_lim, seed=seed)
        # tf$random_normal(shape(input_size,output_size),stddev=wt_lim, seed=seed)
    } else {
        save_state[["node_weights"]]
    }
    B_init <- if(is.null(save_state)) {
        tf$zeros(shape(1L,output_size))
    } else {
        save_state[["node_bias"]]
    }
    W <- tf$Variable(W_init, dtype=tf$float32, name="node_weights")
    B <- tf$Variable(B_init, dtype=tf$float32, name="node_bias")
    
    output <- activation(tf$matmul(input, W) + B, name="node_output")
    if(dropout != 1) {
        output <- tf$nn$dropout(output, keep_prob=dropout, seed=seed)
    }
    
    output_eval <- activation(tf$matmul(input_eval, W) + B, name="node_output")

    return(list(output=output, output_eval=output_eval, weights=W, bias=B))
}

#' Cox output layer helper function
#' @export
#' @examples
#' CoxLayer()
CoxLayer <- function(input, input_eval, input_size, C, R, seed=1, save_state=NULL) {
    set.seed(seed)
    
    W_init <- if(is.null(save_state)) {
        wt_lim <- 6/(input_size + 1)
        tf$random_normal(shape(input_size,1),stddev=wt_lim, seed=seed)
    } else {
        save_state[["cox_weights"]]
    }
    W <- tf$Variable(W_init, dtype=tf$float32)
    
    theta <- tf$matmul(input,W, name="output")
    exp_theta <- tf$exp(theta)
    p1 <- theta * C
    p2 <- tf$log(tf$reduce_sum(R * tf$transpose(exp_theta), axis=1L, keep_dims=T)) * C
    npl <- -(tf$reduce_sum(p1)-tf$reduce_sum(p2))
    
    theta_eval <- tf$matmul(input_eval,W, name="output")
    exp_theta_eval <- tf$exp(theta_eval)
    p1_eval <- theta_eval * C #first part
    p2_eval <- tf$log(tf$reduce_sum(R * tf$transpose(exp_theta_eval), axis=1L, keep_dims=T)) * C #second part
    npl_eval <- -(tf$reduce_sum(p1_eval)-tf$reduce_sum(p2_eval))
    
    return(list(output=theta, output_eval=theta_eval, npl=npl, npl_eval=npl_eval, weights=W))
}

#' Create a Cox-nnet model
#' @param X matrix of covariates
#' @param time Event time
#' @param status Censoring status
#' @param n_nodes Number of hidden nodes in the hidden layer
#' @param dropout dropout parameter of hidden layer (probability of keeping)
#' @param L2 Weight decay/ridge regulariztaion parameter
#' @param input_dropout dropout parameter for the input layer
#' @param activation non-linear activation function
#' @param seed random seed used for intializing weights
#' @param sess TensorFlow session.  Leave as NULL.
#' @param save_state list of weight parameters for restoring a model
#' @export
#' @examples
#' CoxNnetModel()
CoxNnetModel <- function(X, time, event, n_nodes=as.integer(ceiling(sqrt(ncol(X)))), dropout=1, L2=0, input_dropout=1, activation=tf$tanh, seed=1, sess=NULL, save_state=NULL) {
    
    if(is.null(sess)) {
        sess = tf$Session()
    }
    
    input_size <- ncol(X)
    
    R <- lapply(1:length(time), function(i) {
        ifelse(time[i] <= time, 1,0)
    }); 
    R <- do.call(rbind,R)
    
    x_tf_eval <- tf$placeholder(tf$float32, shape(NULL, ncol(X)), name="x_tf_eval")
    x_tf <- tf$constant(X, tf$float32, name="x_tf")
    if(input_dropout != 1) {
        x_tf <- tf$nn$dropout(x_tf, keep_prob=input_dropout, seed=seed)
    }
    R_tf <- tf$constant(R, tf$float32, name="R_tf")
    C_tf <- tf$constant(matrix(event, ncol=1), tf$float32, name="C_tf")
    
    hidden_layer <- HiddenLayer(x_tf, x_tf_eval, input_size, n_nodes, dropout, activation, seed, save_state=save_state)
    cox_layer <- CoxLayer(hidden_layer[["output"]], hidden_layer[["output_eval"]], n_nodes, C_tf, R_tf, seed, save_state=save_state)
    
    if(L2 == 0) {
        cost <- cox_layer[["npl"]]
        cost_eval <- cox_layer[["npl_eval"]]
    } else {
        cost <- L2 * (tf$nn$l2_loss(hidden_layer[["weights"]]) + tf$nn$l2_loss(cox_layer[["weights"]])) + cox_layer[["npl"]]
        cost_eval <- L2 * (tf$nn$l2_loss(hidden_layer[["weights"]]) + tf$nn$l2_loss(cox_layer[["weights"]])) + cox_layer[["npl_eval"]]
    }
    cost <- tf$identity(cost, name="cost")
    cost_eval <- tf$identity(cost_eval, name="cost_eval")
    
    if(!is.null(save_state)) {
        init = tf$global_variables_initializer()
        sess$run(init)
    }
    
    return(list(X=X, time=time, event=event, cost=cost, cost_eval=cost_eval, output_eval=cox_layer[["output_eval"]], node_output_eval=hidden_layer[["output_eval"]], x_tf_eval=x_tf_eval, node_weights=hidden_layer[["weights"]], node_bias=hidden_layer[["bias"]], cox_weights=cox_layer[["weights"]], sess=sess))
}

#' Train a Cox-nnet model
#' @param model Model created from CoxNnetModel
#' @param max_iters Maximum number of iterations to train
#' @param optimizer TensorFlow optimizer used to train mdoel
#' @param eval_step Number of gradient descent steps before calculating cost function
#' @param device device to use (e.g., /cpu:0, /gpu:0)
#' @param verbose print out training iterations
#' @export
#' @examples
#' train_coxnnet()
train_coxnnet <- function(model, max_iters=10000, optimizer=tf$train$AdamOptimizer(), eval_step=347, device="/cpu:0", verbose=F) {
    
    timer <- proc.time()[3]
    
    sess <- model[["sess"]]
    train_step <- optimizer$minimize(model[["cost"]])
    init = tf$global_variables_initializer()
    sess$run(init)
    
    x_tf_eval <- model[["x_tf_eval"]]
    with(tf$device(device), {
        last_score <- Inf
        for(ep in 1:max_iters) {
            sess$run(train_step)
            if(ep %% eval_step == 0) {
                score <- sess$run(model[["cost_eval"]], feed_dict=dict(x_tf_eval=model[["X"]]))
                if(verbose) {
                    message("Cost: ", appendLF=F)
                    message(signif(score))
                }
                if(!is.finite(score)) stop("Something went wrong :(")
                if(score > last_score) {
                    sess$run(model[["node_weights"]]$assign(prev_wts[["node_weights"]]))
                    sess$run(model[["node_bias"]]$assign(prev_wts[["node_bias"]]))
                    sess$run(model[["cox_weights"]]$assign(prev_wts[["cox_weights"]]))
                    score <- sess$run(model[["cost_eval"]], feed_dict=dict(x_tf_eval=model[["X"]]))
                    message("Final cost: ", appendLF=F)
                    message(signif(score))
                    break
                }
                prev_wts <- weights_coxnnet(model)
                last_score <- score
            }
        }
    })
    
    message("Training time: ", appendLF=F)
    message(signif(proc.time()[3] - timer), appendLF=F)
    message("s")
    
    return(model)
}


#' Helper function to cross validate a model
#' @param X matrix of covariates
#' @param time Event time
#' @param status Censoring status
#' @param n_folds Number of cross-validation folds
#' @param CV_var model parameter to profile through cross-validation (e.g., dropout or L2)
#' @param var_steps sequence of values of CV_var to cross-validate
#' @param model_params List of additional parameters passed to CoxNnetModel
#' @param train_parmas List of additional parameters passed to train_coxnnet
#' @param cv_seed cross validation seed used to create folds
#' @param plot Generate a plot of the cross-validated model performance over the sequence of var_steps (requires ggplot2)
#' @export
#' @examples
#' cross_validate_coxnnet()
cross_validate_coxnnet <- function(X, time, event, nfolds=5, CV_var="dropout", var_steps=seq(0.1,0.9,0.1), model_params=list(), train_params=list(), CV_seed=1, plot=F) {
    set.seed(CV_seed)
    
    cv_matrix <- matrix(NA, nrow=length(var_steps), ncol=nfolds)
    colnames(cv_matrix) <- seq_len(nfolds)
    rownames(cv_matrix) <- var_steps
    
    folds <- sample(rep(seq_len(nfolds), length.out = nrow(X)))
    for(f in seq_len(nfolds)) {
        model_params[["X"]] <- X[folds != f,]
        model_params[["time"]] <- time[folds != f]
        model_params[["event"]] <- event[folds != f]
        X_val <- X[folds == f,]
        time_val <- time[folds == f]
        event_val <- event[folds == f]
        
        for(st in var_steps) {
            model_params[[CV_var]] <- st
            model <- train_params[["model"]] <- do.call(CoxNnetModel, model_params)
            model <- do.call(train_coxnnet, train_params)
            PI <- predict_coxnnet(model, X=X_val)[,1]
            model["sess"]$close()
            rm(model)
            gc()
            cv_matrix[as.character(st), as.character(f)] <- cindex(PI, time_val, event_val)
        }
    }
    
    sess$close()
    
    best_step <- var_steps[which.max(rowMeans(cv_matrix))]
    if(plot) {
        require(ggplot2)
        title <- sprintf("%s-fold CV, %s", nfolds, CV_var)
        df <- data.frame(c_index=as.vector(cv_matrix), var_step=rep(var_steps, times=nfolds))
        df2 <- data.frame(c_index=rowMeans(cv_matrix), var_step=var_steps)
        g <- ggplot(data=df) + geom_point( aes(x=var_step, y=c_index, col=var_step)) + geom_line(data=df2, aes(x=var_step,y=c_index)) + ggtitle(title)
        return(list(cv_matrix=cv_matrix, best_step=best_step, cv_plot=g))
    }
    return(list(cv_matrix=cv_matrix,  best_step=best_step))
}

predict_coxnnet <- function(model, X=model[["X"]]) {
        x_tf_eval <- model[["x_tf_eval"]]
        return(model[["sess"]]$run(model[["output_eval"]], feed_dict=dict(x_tf_eval=X)))
}

# npl <- function(model, X, time, train) {
    # sess <- model[["sess"]]
    # theta <- predict_coxnnet(model, X)
    # exp_theta <- exp(theta)
    
    # node_weights <- sess$run(model[["model_weights"]])
    # node_bias <- sess$run(model[["node_bias"]])
    # cox_weights <- sess$run(model[["cox_weights"]])
    
    # p1_eval <- theta_eval * C #first part
    # p2_eval <- tf$log(tf$reduce_sum(R * tf$transpose(exp_theta_eval), axis=1L, keep_dims=T)) * C #second part
    # npl_eval <- -(tf$reduce_sum(p1_eval)-tf$reduce_sum(p2_eval))
    
# }

#' Extract Cox-nnet model weights
#' @param model A trained model from which to extract the weights
#' @export
#' @examples
#' weights_coxnnet()
weights_coxnnet <- function(model) {
    sess <- model[["sess"]]
    node_weights <- sess$run(model[["node_weights"]])
    node_bias <- sess$run(model[["node_bias"]])
    cox_weights <- sess$run(model[["cox_weights"]])
    return(list(node_weights=node_weights, node_bias=node_bias, cox_weights=cox_weights))
}

#' Calculate partial derivatives of Cox-nnet model
#' @param model A trained model from which to extract the weights
#' @param X a matrix of data points where the partial derivatives should be calculated
#' @export
#' @examples
#' pad_coxnnet()
pad_coxnnet <- function(model, X) {
    sess <- model[["sess"]]
    x_tf_eval <- model[["x_tf_eval"]]
    node_output <- model[["node_output"]]
    model_output <- model[["output_eval"]]
    n_nodes <- sess$run(tf$shape(node_output), feed_dict=dict(x_tf_eval=X))[2]
    node_output_split <- tf$split(node_output, as.integer(n_nodes),1L)
    
    node_output_wrt_input <- list()
    
    for(i in 1:length(node_output_split)) {
        nos <- node_output_split[[i]]
        node_output_wrt_input[[i]] <- sess$run(tf$gradients(nos, x_tf_eval), feed_dict=dict(x_tf_eval=X))[[1]]
    }
    model_output_wrt_input <- sess$run(tf$gradients(model_output, x_tf_eval), feed_dict=dict(x_tf_eval=X))[[1]]
    return(list(node_output_wrt_input=node_output_wrt_input, model_output_wrt_input=model_output_wrt_input))
}

#' Calculate the node outputs from a model
#' @param model A trained model from which to extract the weights
#' @param X a matrix of data points which the node outputs should be calculated from
#' @export
#' @examples
#' node_output_coxnnet()
node_output_coxnnet <- function(model, X) {
    sess <- model[["sess"]]
    x_tf_eval <- model[["x_tf_eval"]]
    node_output <- model[["node_output"]]
    node_output <- sess$run(node_output, feed_dict=dict(x_tf_eval=X))
}

#' Helper function in order to save a model to disk
#' @param model A trained model from which to extract the weights
#' @param filename Name of the file to save to
#' @param weights_only Save only model weights, or also the data.  If TRUE, data must be provided in order to reconstruct the model.  
#' @export
#' @examples
#' save_model_coxnnet()
save_model_coxnnet <- function(model, filename, weights_only=T) {
    weights <- weights_coxnnet(model)
    if(weights_only) {
        saveRDS(list(weights), file=filename)
    } else {
        saveRDS(list(X=model[["X"]], time=model[["time"]], event=model[["event"]], weights=weights), file=filename)
    }
}

#' Helper function to load a model from disk
#' @param filename Name of the file containing model
#' @param X Covariate matrix if the model was saved with weights_only
#' @param time Survival time if the model was saved with weights_only
#' @param event Survival event if the model was saved with weights_only
#' @param sess The TensorFlow session can be specified if desire
#' @export
#' @examples
#' load_model_coxnnet()
load_model_coxnnet <- function(filename, X=NULL, time=NULL, event=NULL, sess=NULL) {
    save_state <- readRDS(filename)
    if(is.null(X)) {
        X <- save_state[["X"]]
    }
    if(is.null(time)) {
        time <- save_state[["time"]]
    }
    if(is.null(event)) {
        event <- save_state[["event"]]
    }
    model <- CoxNnetModel(X, time, event, save_state=save_state[[1]], sess=sess)
    return(model)
}






