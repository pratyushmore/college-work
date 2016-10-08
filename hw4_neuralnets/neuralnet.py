import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def determine(self, a):
        """Return the output of the network if ``a`` is input."""
        print a, "\n"
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
            print a
        return a

    def calc_error(self, training_data):
        total_error = 0
        for i, o in training_data:
            print "\n",i
            out = self.determine(i)
            print out
            error = out - o
            total_error += np.sqrt(np.mean(np.square(error)))
        return total_error/len(outputs)


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            total_error = self.calc_error(training_data)
            print "Epoch", j, ", Error:", total_error
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop0(x, y)
            # for b in delta_nabla_b:
            #     print b.shape
            # print "\n"
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-eta*nw/len(mini_batch)
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-eta*nb/len(mini_batch)
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop0(self, inputs, outputs):
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
            return del_b, del_w

    def backprop_1(self, z, a, y):
            #print "z =", z
            return self.error_derivative(y, a) * self.sigma_prime(z)

    def backprop_2(self, d, w, z):
            return np.dot(w.transpose(), d) * self.sigma_prime(z)

    def backprop_3(self, d, a):
            return np.outer(d, a)

    def backprop_4(self, d):
            return d

    def sigma(self, x):
            return 1.0/(1.0+np.exp(-x))

    def sigma_prime(self, x):
            exp_x = np.exp(-x)
            return exp_x/np.square(1.0 + exp_x)

    def error_derivative(self, y, a):
            #print a-y
            return a - y


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


inputs = [[0,0], [0,1], [1,0], [1,1]]
outputs = [[0], [1], [1], [0]]
formatted_inputs = []
formatted_outputs = []

for i in inputs:
    formatted_inputs.append(np.array([i]).T)
for o in outputs:
    formatted_outputs.append(np.array([o]))
neural_net = Network([2,2,1])
neural_net.SGD(zip(formatted_inputs, formatted_outputs), 2500, 4, 10)
print neural_net.determine(formatted_inputs[0])
print neural_net.determine(formatted_inputs[1])
print neural_net.determine(formatted_inputs[2])
print neural_net.determine(formatted_inputs[3])
