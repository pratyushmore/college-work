import numpy as np
import random
import matplotlib.pyplot as plt
import warnings

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    def determine(self, inputs):
        act = inputs
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,act) + b
            act = self.sigma(z)
        return act

    def calc_accur(self, testing_data):
        num_correct = 0
        for inputs, outp in testing_data:
            prediction = self.determine(inputs)
            result = np.argmax(prediction)
            if result == outp:
                num_correct += 1
        print "Cifar Accuracy:", num_correct,"/ 10000"
        return num_correct

    def stoch_minibatch_GD(self, training_data, epochs, alpha, testing_data, batch_size=None):
        if not batch_size:
            batch_size = len(training_data)
        plt.ion()
        fig, ax = plt.subplots()
        plot = ax.scatter([], [])
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        for i in xrange(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in xrange(0, len(training_data), batch_size)]
            for batch in batches:
                self.update_params_for_batch(batch, alpha)
            #total_error = self.calc_error(training_data)
            #print "Epoch", i, ", Error:", total_error
            total_error = self.calc_accur(testing_data)
            array = plot.get_offsets()
            array = np.append(array, [i, total_error])
            plot.set_offsets(array)
                # update x and ylim to show all points:
            ax.set_xlim(0, array[:,0].max() + 0.5)
            ax.set_ylim(0.0, array[:, 1].max() + 0.5)
            # update the figure
            fig.canvas.draw()

    def update_params_for_batch(self, batch, alpha):
        input1, output1 = batch[0]
        del_w, del_b = self.backprop(input1, output1)
        for input_x, output_x in batch[1:]:
            del_w2, del_b2 = self.backprop(input_x, output_x)
            del_w = [w1 + w2 for w1,w2 in zip(del_w, del_w2)]
            del_b = [b1 + b2 for b1,b2 in zip(del_b, del_b2)]
        n = len(batch)
        self.weights = [w1 - alpha*w2/float(n) for w1,w2 in zip(self.weights, del_w)]
        self.biases = [b1 - alpha*b2/float(n) for b1,b2 in zip(self.biases, del_b)]

    def calc_error(self, training_data):
        total_error = 0
        for i, o in training_data:
            out = self.determine(i)
            error = out - o
            total_error += np.sqrt(np.mean(np.square(error)))
        return total_error/len(outputs)

    def backprop(self, inputs, outputs):
        act = inputs
        act_vals = [act]
        z_vals = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,act) + b
            z_vals.append(z)
            act = self.sigma(z)
            act_vals.append(act)
        del_w = range(len(self.weights))
        del_b = range(len(self.biases))
        d = self.backprop_1(z_vals[-1], act_vals[-1], outputs)
        del_w[-1] = self.backprop_3(d, act_vals[-2])
        del_b[-1] = self.backprop_4(d)
        for offset in xrange(2, self.num_layers):
            d = self.backprop_2(d, self.weights[-offset + 1], z_vals[-offset])
            del_w[-offset] = self.backprop_3(d, act_vals[-offset - 1])
            del_b[-offset] = self.backprop_4(d)
        return del_w, del_b

    def backprop_1(self, z, a, y):
        return self.error_derivative(y, a) * self.sigma_prime(z)

    def backprop_2(self, d, w, z):
        return np.dot(w.transpose(), d) * self.sigma_prime(z)

    def backprop_3(self, d, a):
        return np.outer(d, a)

    def backprop_4(self, d):
        return d

    def sigma(self, x):
        with warnings.catch_warnings():
            try:
                exp_x = 1.0/(1.0+np.exp(-x))
                return exp_x
            except Warning:
                print x
                print 'Raised'
                return exp_x

    def sigma_prime(self, x):
        return self.sigma(x)*(1-self.sigma(x))

    def error_derivative(self, y, a):
        return a - y


#FOR XOR

# inputs = [[0,0], [0,1], [1,0], [1,1]]
# outputs = [[0], [1], [1], [0]]
# formatted_inputs = []
# formatted_outputs = []
#
# for i in inputs:
#     formatted_inputs.append(np.array([i]).T)
# for o in outputs:
#     formatted_outputs.append(np.array([o]))
# neural_net = Network([2,2,1])
# neural_net.stoch_minibatch_GD(zip(formatted_inputs, formatted_outputs), 2500, 2)

#For CIFAR

#Helper from https://www.cs.toronto.edu/~kriz/cifar.html
#Dataset from tech report associated with https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#Helpers based on above helpers

def format_cifar_inp(x):
    #rgb = np.array([x[k:k+1024] for k in range(0, 3072, 1024)]).T
    #greyscale = np.array([np.mean(rgb, axis=1)]).T
    #return greyscale / 255.
    return np.reshape(x/255., (3072,1))

def cifar_load_data_wrapper():
    training_data = []
    for i in range(1,6):
        batch_data = unpickle('cifar_data/cifar-10-batches-py/data_batch_' + str(i))
        training_inputs = batch_data.get('data')
        tr_l = batch_data.get('labels')
        training_results = [cifar_vectorized_result(y) for y in tr_l]
        training_inputs_formatted = [format_cifar_inp(x) for x in training_inputs]
        training_data.extend(zip(training_inputs_formatted, training_results))
    test_batch = unpickle('cifar_data/cifar-10-batches-py/test_batch')
    testing_inputs = test_batch.get('data')
    tst_l = test_batch.get('labels')
    testing_inputs_formatted = [format_cifar_inp(x) for x in testing_inputs]
    testing_data = zip(testing_inputs_formatted, tst_l)
    return (training_data, testing_data)

def cifar_vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

training_data, testing_data = cifar_load_data_wrapper()
#print (training_data[0])
#print np.exp(training_data[0][0])
neural_net = Network([3072, 30,20, 10])
neural_net.stoch_minibatch_GD(training_data, 30, 3, testing_data, batch_size=10)
