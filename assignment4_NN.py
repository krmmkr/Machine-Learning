import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
import tensorflow as tf

def error_rate(p, t):
    return np.mean(p != t)

def NN(Xtrain, Xtest, Ytrain, Ytest):
    max_iter = 1000
    #print_period = 50

    lr = 0.00004
    
    N, D = Xtrain.shape
    
    # No of layers and nodes
    M1 = 100
    M2 = 100
    K = 1
    # bias and weights
    W1_init = np.random.randn(D, M1)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K)
    b3_init = np.zeros(K)

    # define variables and expressions
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

     # define the model
    Z1 = tf.nn.sigmoid( tf.matmul(X, W1) + b1 )
    Z2 = tf.nn.sigmoid( tf.matmul(Z1, W2) + b2 )
    calcY = tf.matmul(Z2, W3) + b3 # remember, the cost function does the softmaxing! weird, right?

    #Softmax
    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=calcY, labels=T))

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    predictedprob = tf.nn.sigmoid(calcY)
    predictor = tf.round(predictedprob)

    #predict_op = tf.argmax(Yish, 1)
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    test_costs = []
    train_costs = []
    test_accuracy = []
    train_accuracy = []
    # initializing session
    init = tf.global_variables_initializer()
    with tf.Session(config = config) as session:
        session.run(init)

        for i in range(max_iter):
            session.run(optimizer, feed_dict={X: Xtrain, T: Ytrain})
            pred_train = session.run(predictor, feed_dict={X: Xtrain})
            train_cost = session.run(cost, feed_dict={X: Xtrain, T: Ytrain})
            test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest})
            pred_test = session.run(predictor, feed_dict={X: Xtest})
            train_costs.append(train_cost)
            test_costs.append(test_cost)
            train_accuracy.append(accuracy_score(Ytrain,pred_train))
            test_accuracy.append(accuracy_score(Ytest,pred_test))
            pred_prob_test = session.run(predictedprob, feed_dict={X: Xtest})
            if i%10 == 0:
                #cal_accuracy(Ytrain, pred_train)
                #cal_accuracy(Ytest, pred_test)
        #test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest})
        #prediction = session.run(predict_op, feed_dict={X: Xtest})
                err = error_rate(pred_test, Ytest)
                print(err)