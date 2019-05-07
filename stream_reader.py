import numpy
from pyspark import SparkContext
from numpy import genfromtxt
from pyspark.streaming import StreamingContext
import os
os.environ['PYSPARK_PYTHON'] = 'virt/bin/python'


sc = SparkContext(appName="FacialRecognizer")
ssc = StreamingContext(sc, 5)
socket_stream = ssc.socketTextStream("0.0.0.0", 5555)


def run_visible(data, weights, hidden_nodes):
    """
    Assuming the RBM has been trained (so that weights for the network have been learned),
    run the network on a set of visible units, to get a sample of the hidden units.

    Taken from https://github.com/echen/restricted-boltzmann-machines/blob/master/rbm.py
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.

    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = numpy.ones((num_examples, hidden_nodes + 1))

    # Insert bias units of 1 into the first column of data.
    data = numpy.insert(data, 0, 1)

    # Calculate the activations of the hidden units.
    hidden_activations = numpy.dot(data, weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = _logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > numpy.random.rand(num_examples, hidden_nodes + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1

    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states

def _logistic(x):
    return 1.0 / (1 + numpy.exp(-x))


weights = genfromtxt('rbmWeights.csv', delimiter=',')

# socket_stream.map(lambda v: "Embedding: " + v).pprint() # prints the embedding sent on the socket
# prints the hidden units calculated from the weights
socket_stream.map(lambda v: run_visible(numpy.fromstring(v, count=128, sep=", "), weights, 20)).pprint()

# Next steps would be to take these hidden states, backpropogate, and look at the resulting visible states.
# Run the cost function to determine if the states are sufficiently similar to be the same person as the neural neetwork
# was trained on

ssc.start()
ssc.awaitTermination()
