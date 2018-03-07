# Jason
BRNN and LSTM

Machine Learning Mastery

Start Here       Blog       Books       About       Contact

Search...
 
Need help with LSTMs in Python? Take the FREE Mini-Course.

How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras
by Jason Brownlee on June 16, 2017 in Long Short-Term Memory Networks
Bidirectional LSTMs are an extension of traditional LSTMs that can improve model performance on sequence classification problems.

In problems where all timesteps of the input sequence are available, Bidirectional LSTMs train two instead of one LSTMs on the input sequence. The first on the input sequence as-is and the second on a reversed copy of the input sequence. This can provide additional context to the network and result in faster and even fuller learning on the problem.

In this tutorial, you will discover how to develop Bidirectional LSTMs for sequence classification in Python with the Keras deep learning library.

After completing this tutorial, you will know:

How to develop a small contrived and configurable sequence classification problem.
How to develop an LSTM and Bidirectional LSTM for sequence classification.
How to compare the performance of the merge mode used in Bidirectional LSTMs.
Let’s get started.

How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras
How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras
Photo by Cristiano Medeiros Dalbem, some rights reserved.

Overview
This tutorial is divided into 6 parts; they are:

Bidirectional LSTMs
Sequence Classification Problem
LSTM For Sequence Classification
Bidirectional LSTM For Sequence Classification
Compare LSTM to Bidirectional LSTM
Comparing Bidirectional LSTM Merge Modes
Environment
This tutorial assumes you have a Python SciPy environment installed. You can use either Python 2 or 3 with this example.

This tutorial assumes you have Keras (v2.0.4+) installed with either the TensorFlow (v1.1.0+) or Theano (v0.9+) backend.

This tutorial also assumes you have scikit-learn, Pandas, NumPy, and Matplotlib installed.

If you need help setting up your Python environment, see this post:

How to Setup a Python Environment for Machine Learning and Deep Learning with Anaconda
Need help with LSTMs for Sequence Prediction?
Take my free 7-day email course and discover 6 different LSTM architectures (with sample code).

Click to sign-up and also get a free PDF Ebook version of the course.

Start Your FREE Mini-Course Now!
Bidirectional LSTMs
The idea of Bidirectional Recurrent Neural Networks (RNNs) is straightforward.

It involves duplicating the first recurrent layer in the network so that there are now two layers side-by-side, then providing the input sequence as-is as input to the first layer and providing a reversed copy of the input sequence to the second.

To overcome the limitations of a regular RNN […] we propose a bidirectional recurrent neural network (BRNN) that can be trained using all available input information in the past and future of a specific time frame.

…

The idea is to split the state neurons of a regular RNN in a part that is responsible for the positive time direction (forward states) and a part for the negative time direction (backward states)

— Mike Schuster and Kuldip K. Paliwal, Bidirectional Recurrent Neural Networks, 1997

This approach has been used to great effect with Long Short-Term Memory (LSTM) Recurrent Neural Networks.

The use of providing the sequence bi-directionally was initially justified in the domain of speech recognition because there is evidence that the context of the whole utterance is used to interpret what is being said rather than a linear interpretation.

… relying on knowledge of the future seems at first sight to violate causality. How can we base our understanding of what we’ve heard on something that hasn’t been said yet? However, human listeners do exactly that. Sounds, words, and even whole sentences that at first mean nothing are found to make sense in the light of future context. What we must remember is the distinction between tasks that are truly online – requiring an output after every input – and those where outputs are only needed at the end of some input segment.

— Alex Graves and Jurgen Schmidhuber, Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures, 2005

The use of bidirectional LSTMs may not make sense for all sequence prediction problems, but can offer some benefit in terms of better results to those domains where it is appropriate.

We have found that bidirectional networks are significantly more effective than unidirectional ones…

— Alex Graves and Jurgen Schmidhuber, Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures, 2005

To be clear, timesteps in the input sequence are still processed one at a time, it is just the network steps through the input sequence in both directions at the same time.

Bidirectional LSTMs in Keras
Bidirectional LSTMs are supported in Keras via the Bidirectional layer wrapper.

This wrapper takes a recurrent layer (e.g. the first LSTM layer) as an argument.

It also allows you to specify the merge mode, that is how the forward and backward outputs should be combined before being passed on to the next layer. The options are:

‘sum‘: The outputs are added together.
‘mul‘: The outputs are multiplied together.
‘concat‘: The outputs are concatenated together (the default), providing double the number of outputs to the next layer.
‘ave‘: The average of the outputs is taken.
The default mode is to concatenate, and this is the method often used in studies of bidirectional LSTMs.

Sequence Classification Problem
We will define a simple sequence classification problem to explore bidirectional LSTMs.

The problem is defined as a sequence of random values between 0 and 1. This sequence is taken as input for the problem with each number provided one per timestep.

A binary label (0 or 1) is associated with each input. The output values are all 0. Once the cumulative sum of the input values in the sequence exceeds a threshold, then the output value flips from 0 to 1.

A threshold of 1/4 the sequence length is used.

For example, below is a sequence of 10 input timesteps (X):


0.63144003 0.29414551 0.91587952 0.95189228 0.32195638 0.60742236 0.83895793 0.18023048 0.84762691 0.29165514
1
0.63144003 0.29414551 0.91587952 0.95189228 0.32195638 0.60742236 0.83895793 0.18023048 0.84762691 0.29165514
The corresponding classification output (y) would be:


0 0 0 1 1 1 1 1 1 1
1
0 0 0 1 1 1 1 1 1 1
We can implement this in Python.

The first step is to generate a sequence of random values. We can use the random() function from the random module.


# create a sequence of random numbers in [0,1]
X = array([random() for _ in range(10)])
1
2
# create a sequence of random numbers in [0,1]
X = array([random() for _ in range(10)])
We can define the threshold as one-quarter the length of the input sequence.


# calculate cut-off value to change class values
limit = 10/4.0
1
2
# calculate cut-off value to change class values
limit = 10/4.0
The cumulative sum of the input sequence can be calculated using the cumsum() NumPy function. This function returns a sequence of cumulative sum values, e.g.:


pos1, pos1+pos2, pos1+pos2+pos3, ...
1
pos1, pos1+pos2, pos1+pos2+pos3, ...
We can then calculate the output sequence as whether each cumulative sum value exceeded the threshold.


# determine the class outcome for each item in cumulative sequence
y = array([0 if x < limit else 1 for x in cumsum(X)])
1
2
# determine the class outcome for each item in cumulative sequence
y = array([0 if x < limit else 1 for x in cumsum(X)])
The function below, named get_sequence(), draws all of this together, taking as input the length of the sequence, and returns the X and y components of a new problem case.


from random import random
from numpy import array
from numpy import cumsum

# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	return X, y
1
2
3
4
5
6
7
8
9
10
11
12
13
from random import random
from numpy import array
from numpy import cumsum
 
# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	return X, y
We can test this function with a new 10 timestep sequence as follows:


X, y = get_sequence(10)
print(X)
print(y)
1
2
3
X, y = get_sequence(10)
print(X)
print(y)
Running the example first prints the generated input sequence followed by the matching output sequence.


[ 0.22228819 0.26882207 0.069623 0.91477783 0.02095862 0.71322527
0.90159654 0.65000306 0.88845226 0.4037031 ]
[0 0 0 0 0 0 1 1 1 1]
1
2
3
[ 0.22228819 0.26882207 0.069623 0.91477783 0.02095862 0.71322527
0.90159654 0.65000306 0.88845226 0.4037031 ]
[0 0 0 0 0 0 1 1 1 1]
LSTM For Sequence Classification
We can start off by developing a traditional LSTM for the sequence classification problem.

Firstly, we must update the get_sequence() function to reshape the input and output sequences to be 3-dimensional to meet the expectations of the LSTM. The expected structure has the dimensions [samples, timesteps, features]. The classification problem has 1 sample (e.g. one sequence), a configurable number of timesteps, and one feature per timestep.

The classification problem has 1 sample (e.g. one sequence), a configurable number of timesteps, and one feature per timestep.

Therefore, we can reshape the sequences as follows.


# reshape input and output data to be suitable for LSTMs
X = X.reshape(1, n_timesteps, 1)
y = y.reshape(1, n_timesteps, 1)
1
2
3
# reshape input and output data to be suitable for LSTMs
X = X.reshape(1, n_timesteps, 1)
y = y.reshape(1, n_timesteps, 1)
The updated get_sequence() function is listed below.


# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y
1
2
3
4
5
6
7
8
9
10
11
12
# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y
We will define the sequences as having 10 timesteps.

Next, we can define an LSTM for the problem. The input layer will have 10 timesteps with 1 feature a piece, input_shape=(10, 1).

The first hidden layer will have 20 memory units and the output layer will be a fully connected layer that outputs one value per timestep. A sigmoid activation function is used on the output to predict the binary value.

A TimeDistributed wrapper layer is used around the output layer so that one value per timestep can be predicted given the full sequence provided as input. This requires that the LSTM hidden layer returns a sequence of values (one per timestep) rather than a single value for the whole input sequence.

Finally, because this is a binary classification problem, the binary log loss (binary_crossentropy in Keras) is used. The efficient ADAM optimization algorithm is used to find the weights and the accuracy metric is calculated and reported each epoch.


# define LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(10, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
1
2
3
4
5
# define LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(10, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
The LSTM will be trained for 1,000 epochs. A new random input sequence will be generated each epoch for the network to be fit on. This ensures that the model does not memorize a single sequence and instead can generalize a solution to solve all possible random input sequences for this problem.


# train LSTM
for epoch in range(1000):
	# generate new random sequence
	X,y = get_sequence(n_timesteps)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
1
2
3
4
5
6
# train LSTM
for epoch in range(1000):
	# generate new random sequence
	X,y = get_sequence(n_timesteps)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
Once trained, the network will be evaluated on yet another random sequence. The predictions will be then compared to the expected output sequence to provide a concrete example of the skill of the system.


# evaluate LSTM
X,y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
	print('Expected:', y[0, i], 'Predicted', yhat[0, i])
1
2
3
4
5
# evaluate LSTM
X,y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
	print('Expected:', y[0, i], 'Predicted', yhat[0, i])
The complete example is listed below.


from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y

# define problem properties
n_timesteps = 10
# define LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(1000):
	# generate new random sequence
	X,y = get_sequence(n_timesteps)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate LSTM
X,y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
	print('Expected:', y[0, i], 'Predicted', yhat[0, i])
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
 
# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y
 
# define problem properties
n_timesteps = 10
# define LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(1000):
	# generate new random sequence
	X,y = get_sequence(n_timesteps)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate LSTM
X,y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
	print('Expected:', y[0, i], 'Predicted', yhat[0, i])
Running the example prints the log loss and classification accuracy on the random sequences each epoch.

This provides a clear idea of how well the model has generalized a solution to the sequence classification problem.

We can see that the model does well, achieving a final accuracy that hovers around 90% and 100% accurate. Not perfect, but good for our purposes.

The predictions for a new random sequence are compared to the expected values, showing a mostly correct result with a single error.


...
Epoch 1/1
0s - loss: 0.2039 - acc: 0.9000
Epoch 1/1
0s - loss: 0.2985 - acc: 0.9000
Epoch 1/1
0s - loss: 0.1219 - acc: 1.0000
Epoch 1/1
0s - loss: 0.2031 - acc: 0.9000
Epoch 1/1
0s - loss: 0.1698 - acc: 0.9000
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
...
Epoch 1/1
0s - loss: 0.2039 - acc: 0.9000
Epoch 1/1
0s - loss: 0.2985 - acc: 0.9000
Epoch 1/1
0s - loss: 0.1219 - acc: 1.0000
Epoch 1/1
0s - loss: 0.2031 - acc: 0.9000
Epoch 1/1
0s - loss: 0.1698 - acc: 0.9000
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Bidirectional LSTM For Sequence Classification
Now that we know how to develop an LSTM for the sequence classification problem, we can extend the example to demonstrate a Bidirectional LSTM.

We can do this by wrapping the LSTM hidden layer with a Bidirectional layer, as follows:


model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
1
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
This will create two copies of the hidden layer, one fit in the input sequences as-is and one on a reversed copy of the input sequence. By default, the output values from these LSTMs will be concatenated.

That means that instead of the TimeDistributed layer receiving 10 timesteps of 20 outputs, it will now receive 10 timesteps of 40 (20 units + 20 units) outputs.

The complete example is listed below.


from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y

# define problem properties
n_timesteps = 10
# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(1000):
	# generate new random sequence
	X,y = get_sequence(n_timesteps)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate LSTM
X,y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
	print('Expected:', y[0, i], 'Predicted', yhat[0, i])
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
 
# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y
 
# define problem properties
n_timesteps = 10
# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(1000):
	# generate new random sequence
	X,y = get_sequence(n_timesteps)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate LSTM
X,y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
	print('Expected:', y[0, i], 'Predicted', yhat[0, i])
Running the example, we see a similar output as in the previous example.

The use of bidirectional LSTMs have the effect of allowing the LSTM to learn the problem faster.

This is not apparent from looking at the skill of the model at the end of the run, but instead, the skill of the model over time.


...
Epoch 1/1
0s - loss: 0.0967 - acc: 0.9000
Epoch 1/1
0s - loss: 0.0865 - acc: 1.0000
Epoch 1/1
0s - loss: 0.0905 - acc: 0.9000
Epoch 1/1
0s - loss: 0.2460 - acc: 0.9000
Epoch 1/1
0s - loss: 0.1458 - acc: 0.9000
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
...
Epoch 1/1
0s - loss: 0.0967 - acc: 0.9000
Epoch 1/1
0s - loss: 0.0865 - acc: 1.0000
Epoch 1/1
0s - loss: 0.0905 - acc: 0.9000
Epoch 1/1
0s - loss: 0.2460 - acc: 0.9000
Epoch 1/1
0s - loss: 0.1458 - acc: 0.9000
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [0] Predicted [0]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Expected: [1] Predicted [1]
Compare LSTM to Bidirectional LSTM
In this example, we will compare the performance of traditional LSTMs to a Bidirectional LSTM over time while the models are being trained.

We will adjust the experiment so that the models are only trained for 250 epochs. This is so that we can get a clear idea of how learning unfolds for each model and how the learning behavior differs with bidirectional LSTMs.

We will compare three different models; specifically:

LSTM (as-is)
LSTM with reversed input sequences (e.g. you can do this by setting the “go_backwards” argument to he LSTM layer to “True”)
Bidirectional LSTM
This comparison will help to show that bidirectional LSTMs can in fact add something more than simply reversing the input sequence.

We will define a function to create and return an LSTM with either forward or backward input sequences, as follows:


def get_lstm_model(n_timesteps, backwards):
	model = Sequential()
	model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
1
2
3
4
5
6
def get_lstm_model(n_timesteps, backwards):
	model = Sequential()
	model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
We can develop a similar function for bidirectional LSTMs where the merge mode can be specified as an argument. The default of concatenation can be specified by setting the merge mode to the value ‘concat’.


def get_bi_lstm_model(n_timesteps, mode):
	model = Sequential()
	model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
1
2
3
4
5
6
def get_bi_lstm_model(n_timesteps, mode):
	model = Sequential()
	model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
Finally, we define a function to fit a model and retrieve and store the loss each training epoch, then return a list of the collected loss values after the model is fit. This is so that we can graph the log loss from each model configuration and compare them.


def train_model(model, n_timesteps):
	loss = list()
	for _ in range(250):
		# generate new random sequence
		X,y = get_sequence(n_timesteps)
		# fit model for one epoch on this sequence
		hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
		loss.append(hist.history['loss'][0])
	return loss
1
2
3
4
5
6
7
8
9
def train_model(model, n_timesteps):
	loss = list()
	for _ in range(250):
		# generate new random sequence
		X,y = get_sequence(n_timesteps)
		# fit model for one epoch on this sequence
		hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
		loss.append(hist.history['loss'][0])
	return loss
Putting this all together, the complete example is listed below.

First a traditional LSTM is created and fit and the log loss values plot. This is repeated with an LSTM with reversed input sequences and finally an LSTM with a concatenated merge.


from random import random
from numpy import array
from numpy import cumsum
from matplotlib import pyplot
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y

def get_lstm_model(n_timesteps, backwards):
	model = Sequential()
	model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def get_bi_lstm_model(n_timesteps, mode):
	model = Sequential()
	model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def train_model(model, n_timesteps):
	loss = list()
	for _ in range(250):
		# generate new random sequence
		X,y = get_sequence(n_timesteps)
		# fit model for one epoch on this sequence
		hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
		loss.append(hist.history['loss'][0])
	return loss


n_timesteps = 10
results = DataFrame()
# lstm forwards
model = get_lstm_model(n_timesteps, False)
results['lstm_forw'] = train_model(model, n_timesteps)
# lstm backwards
model = get_lstm_model(n_timesteps, True)
results['lstm_back'] = train_model(model, n_timesteps)
# bidirectional concat
model = get_bi_lstm_model(n_timesteps, 'concat')
results['bilstm_con'] = train_model(model, n_timesteps)
# line plot of results
results.plot()
pyplot.show()
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
from random import random
from numpy import array
from numpy import cumsum
from matplotlib import pyplot
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
 
# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y
 
def get_lstm_model(n_timesteps, backwards):
	model = Sequential()
	model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
 
def get_bi_lstm_model(n_timesteps, mode):
	model = Sequential()
	model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
 
def train_model(model, n_timesteps):
	loss = list()
	for _ in range(250):
		# generate new random sequence
		X,y = get_sequence(n_timesteps)
		# fit model for one epoch on this sequence
		hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
		loss.append(hist.history['loss'][0])
	return loss
 
 
n_timesteps = 10
results = DataFrame()
# lstm forwards
model = get_lstm_model(n_timesteps, False)
results['lstm_forw'] = train_model(model, n_timesteps)
# lstm backwards
model = get_lstm_model(n_timesteps, True)
results['lstm_back'] = train_model(model, n_timesteps)
# bidirectional concat
model = get_bi_lstm_model(n_timesteps, 'concat')
results['bilstm_con'] = train_model(model, n_timesteps)
# line plot of results
results.plot()
pyplot.show()
Running the example creates a line plot.

Your specific plot may vary in the details, but will show the same trends.

We can see that the LSTM forward (blue) and LSTM backward (orange) show similar log loss over the 250 training epochs.

We can see that the Bidirectional LSTM log loss is different (green), going down sooner to a lower value and generally staying lower than the other two configurations.

Line Plot of Log Loss for an LSTM, Reversed LSTM and a Bidirectional LSTM
Line Plot of Log Loss for an LSTM, Reversed LSTM and a Bidirectional LSTM

Comparing Bidirectional LSTM Merge Modes
There a 4 different merge modes that can be used to combine the outcomes of the Bidirectional LSTM layers.

They are concatenation (default), multiplication, average, and sum.

We can compare the behavior of different merge modes by updating the example from the previous section as follows:


n_timesteps = 10
results = DataFrame()
# sum merge
model = get_bi_lstm_model(n_timesteps, 'sum')
results['bilstm_sum'] = train_model(model, n_timesteps)
# mul merge
model = get_bi_lstm_model(n_timesteps, 'mul')
results['bilstm_mul'] = train_model(model, n_timesteps)
# avg merge
model = get_bi_lstm_model(n_timesteps, 'ave')
results['bilstm_ave'] = train_model(model, n_timesteps)
# concat merge
model = get_bi_lstm_model(n_timesteps, 'concat')
results['bilstm_con'] = train_model(model, n_timesteps)
# line plot of results
results.plot()
pyplot.show()
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
n_timesteps = 10
results = DataFrame()
# sum merge
model = get_bi_lstm_model(n_timesteps, 'sum')
results['bilstm_sum'] = train_model(model, n_timesteps)
# mul merge
model = get_bi_lstm_model(n_timesteps, 'mul')
results['bilstm_mul'] = train_model(model, n_timesteps)
# avg merge
model = get_bi_lstm_model(n_timesteps, 'ave')
results['bilstm_ave'] = train_model(model, n_timesteps)
# concat merge
model = get_bi_lstm_model(n_timesteps, 'concat')
results['bilstm_con'] = train_model(model, n_timesteps)
# line plot of results
results.plot()
pyplot.show()
Running the example will create a line plot comparing the log loss of each merge mode.

Your specific plot may differ but will show the same behavioral trends.

The different merge modes result in different model performance, and this will vary depending on your specific sequence prediction problem.

In this case, we can see that perhaps a sum (blue) and concatenation (red) merge mode may result in better performance, or at least lower log loss.

Line Plot to Compare Merge Modes for Bidirectional LSTMs
Line Plot to Compare Merge Modes for Bidirectional LSTMs

Summary
In this tutorial, you discovered how to develop Bidirectional LSTMs for sequence classification in Python with Keras.

Specifically, you learned:

How to develop a contrived sequence classification problem.
How to develop an LSTM and Bidirectional LSTM for sequence classification.
How to compare merge modes for Bidirectional LSTMs for sequence classification.
Do you have any questions?
Ask your questions in the comments below and I will do my best to answer.

Develop LSTMs for Sequence Prediction Today!
Long Short-Term Memory Networks with Python

Develop Your Own LSTM models in Minutes
…with just a few lines of python code

Discover how in my new Ebook:
Long Short-Term Memory Networks with Python

It provides self-study tutorials on topics like:
CNN LSTMs, Encoder-Decoder LSTMs, generative models, data preparation, making predictions and much more…

Finally Bring LSTM Recurrent Neural Networks to
Your Sequence Predictions Projects
Skip the Academics. Just Results.

Click to learn more.

 
About Jason Brownlee
Jason Brownlee, Ph.D. is a machine learning specialist who teaches developers how to get results with modern machine learning methods via hands-on tutorials.
View all posts by Jason Brownlee →
 How to Get Reproducible Results with KerasData Preparation for Variable Length Input Sequences 
42 Responses to How to Develop a Bidirectional LSTM For Sequence Classification in Python with Keras

Siddharth June 18, 2017 at 4:33 pm # 
Great post! Do you think bidirectional LSTMs can be used for time series prediciton problems?

REPLY

Jason Brownlee June 19, 2017 at 8:33 am # 
Yes, the question is, can they lift performance on your problem. Try it and see.

REPLY

Orozcohsu June 18, 2017 at 6:38 pm # 
Greatest ,thank you

REPLY

Jason Brownlee June 19, 2017 at 8:43 am # 
I’m glad it helped!

REPLY

truongtrang June 19, 2017 at 3:17 pm # 
hi Jason,
In fact, I usually need to use multi thread ( multi worker) for load model Keras for improve performance for my system. But when I use multi thread to work with model Keras, its so error with graph, so I used multi process instead. I wana ask you have another solution for multi worker with Keras? Hope you can understand what i say.
Thank you.

REPLY

Jason Brownlee June 20, 2017 at 6:35 am # 
Yes, I would recommend using GPUs on AWS:
http://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/

REPLY

Yitzhak June 20, 2017 at 12:05 am # 
Thanks, Jason, such a great post !

REPLY

Jason Brownlee June 20, 2017 at 6:37 am # 
You’re welcome Yitzhak.

REPLY

John Jaro July 2, 2017 at 12:19 am # 
hi Jason, thanks greatly for your work. I’ve read probably 50 of your blog articles!

I’m still struggling to understand how to reshape lagged data for LSTM and would greatly appreciate your help.

I’m working on sequence classification on time-series data over multiple days. I’ve lagged the data together (2D) and created differential features using code very similar to yours, and generated multiple look forward and look backward features over a window of about +5 and -4:


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
df = DataFrame(data)
columns = [df.shift(i) for i in range(1, lag+1)]
columns.append(df)
df = concat(columns, axis=1)
return df

I’ve gotten decent results with Conv1D residual networks on my dataset, but my experiments with LSTM are total failures.

I reshape the data for Conv1D like so: X = X.reshape(X.shape[0], X.shape[1], 1)

Is this same data shape appropriate for LSTM or Bidirectional LSTM? I think it needs to be different, but I cannot figure out how despite hours of searching.

Thanks for your assistance if any!

REPLY

John Jaro July 2, 2017 at 12:27 am # 
By the way, my question is not a prediction task – it’s multi class classification: looking at a particular day’s data in combination with surrounding lagged/diff’d day’s data and saying it is one of 10 different types of events.

REPLY

Jason Brownlee July 2, 2017 at 6:30 am # 
Great. Sequence classification.

One day might be one sequence and be comprised of lots of time steps for lots of features.

REPLY

Jason Brownlee July 2, 2017 at 6:29 am # 
Thanks John!

Are you working on a sequence classification problem or sequence regression problem? Do you want to classify a whole sequence or predict the next value in the sequence? This will determine the type of LSTM you want.

The input to LSTMs is 3d with the form [samples, time steps, features].

Samples are sequences.
Time steps are lag obs.
Features are things measured at each time step.

Does that help?

REPLY

jilian July 2, 2017 at 11:57 pm # 
Hello Jason,
Thank you for this blog .
i want to use a 2D LSTM (the same as gridlstm or multi diagonal LSTM) after CNN,the input is image with 3D RGB (W * H * D)
does the keras develop GridLSTM or multi-directional LSTM.
i saw the tensorflow develop the GridLSTM.can link it into keras?
Thank you.

REPLY

Jason Brownlee July 3, 2017 at 5:34 am # 
You can use a CNN as a front-end model for LSTM.

Sorry, I’ve not heard of “grid lstm” or “multi-directional lstm”.

REPLY

Marianico August 3, 2017 at 7:32 pm # 
Nice post, Jason! I have two questions:

1.- May Bidirectional() work in a regression model without TimeDistributed() wrapper?
2.- May I have two Bidirectional() layers, or the model would be a far too complex?
3.- Does Bidirectional() requires more input data to train?

Thank you in advance! 🙂

REPLY

Jason Brownlee August 4, 2017 at 6:58 am # 
Hi Marianico,

1. sure.
2. you can if you want, try it.
3. it may, test this assumption.

REPLY

Thabet Ali September 28, 2017 at 10:10 pm # 
Thank you so much Mr Jason!

REPLY

Jason Brownlee September 29, 2017 at 5:04 am # 
You’re welcome.

REPLY

Zhong October 24, 2017 at 2:06 pm # 
very nice post, Jason. I am working on a CNN+LSTM problem atm. This post is really helpful.

By the way, do you have some experience of CNN + LSTM for sequence classification. The inputs are frames of medical scans over time.

We are experiencing a quick overfitting (95% accuracy after 5 epochs).

What is the best practice to slow down the overfitting?

REPLY

Jason Brownlee October 24, 2017 at 4:01 pm # 
Yes, I have a small example in my book.

Consider dropout and other forms of regularization. Also try larger batch sizes.

Let me know how you go.

REPLY

Ed December 15, 2017 at 3:13 am # 
Hi Jason! First of all congratulations on your work, I have been learning a lot through your posts.

Regarding this topic: I am handling a problem where I have time series with different size, and I want to binary classify each fixed-size window of each time series.

I know that n_timesteps should be the fixed-size of the window, but then I will have a different number of samples for each time series. A solution might be defining a fixed sample size and add “zero” windows to smaller time series, but I would like to know if there are other options. Do you have any suggestion to overcome this problem?

Thanks in advance!

REPLY

Jason Brownlee December 15, 2017 at 5:39 am # 
Hi Ed, yes, use zero padding and a mask to ignore zero values.

If you know you need to make a prediction every n steps, consider splitting each group of n steps into separate samples of length n. It will make modeling so much easier.

Let me know how you go.

REPLY

Saisaisun December 15, 2017 at 8:32 pm # 
Hello Jason,
I have a question on how to output each timestep of a sequence using LSTM. My problem is 0-1 classification. All I want to do is for one sample (a sequence) to output many labels (corresponding to this sequence). My data are all 3D, including labels and input. But the output I got is wrong. All of them (predicted labels) were 0.
Here is my code:

###
model = Sequential()
model.add(Masking(mask_value= 0,input_shape=(maxlen,feature_dim)))
model.add (LSTM(int(topos[0]), activation=act, kernel_initializer=’normal’, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation=’sigmoid’)))
model.compile(loss=’binary_crossentropy’, optimizer=opt, metrics=[‘accuracy’])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[tensorboard], validation_data=(x_val,y_val), shuffle=True, initial_epoch=0)
###

Is this right? Thank you so much.

REPLY

Jason Brownlee December 16, 2017 at 5:25 am # 
Perhaps you need a larger model, more training, more data, etc…

Here are some ideas:
http://machinelearningmastery.com/improve-deep-learning-performance/

REPLY

devi December 28, 2017 at 9:42 pm # 
very good post,I am trying to try it.

REPLY

Jason Brownlee December 29, 2017 at 5:21 am # 
Thanks.

REPLY

Namrata January 3, 2018 at 11:07 pm # 
Hi Jason

Are we feeding one sequence value[i.e sequence[i]] at each time step into the LSTM? If this is true, does this sequence[i] go through all the memory units in the LSTM? Or does it progressively go through each memory unit as timesteps are incremented?

Thanks

REPLY

Jason Brownlee January 4, 2018 at 8:11 am # 
Each input is passed through all units in the layer at the same time.

REPLY

Jacob January 20, 2018 at 2:41 pm # 
Great post, Jason. I learn best from examples. I bought a series of your books and feel like I have learned more from them then actual courses and am very satisfied with them.

REPLY

Jason Brownlee January 21, 2018 at 9:07 am # 
Thanks Jacob, I write to help people and it is great to hear that it is helping!

REPLY

Ami Tabak January 22, 2018 at 12:13 am # 
Hi Jason
I’m looking into predicting a cosine series based on input sine series
I tried redefining the get_sequence as follows:


def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    start = (random() + n_timesteps) % (2 * math.pi)
    end = start + (2 * math.pi)
    X = np.sin((np.arange(start,end, (end - start) / n_timesteps))) 
    # calculate cut-off value to change class values
    limit = n_timesteps/4.0
    # determine the class outcome for each item in cumulative sequence
    y = np.cos(X)
    # reshape input and output data to be suitable for LSTMs
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y
1
2
3
4
5
6
7
8
9
10
11
12
13
def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    start = (random() + n_timesteps) % (2 * math.pi)
    end = start + (2 * math.pi)
    X = np.sin((np.arange(start,end, (end - start) / n_timesteps))) 
    # calculate cut-off value to change class values
    limit = n_timesteps/4.0
    # determine the class outcome for each item in cumulative sequence
    y = np.cos(X)
    # reshape input and output data to be suitable for LSTMs
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y
in the main i changed the loss function to use MSE as ‘binary_crossentropy’ to my understanding produces a 0/1 loss and I’m looking at a continues function use case
model.compile(loss=’mean_squared_error’, optimizer=’adam’, metrics=[‘acc’])

Alas the model doesn’t converge and results in binary like results.

Expected: [ 0.55061242] Predicted [1]

Any suggestions ?

BR
Ami

Full modified code:
===============


import math
import numpy as np
from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
 
# create a sequence classification instance
def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    start = (random() + n_timesteps) % (2 * math.pi)
    end = start + (2 * math.pi)
    X = np.sin((np.arange(start,end, (end - start) / n_timesteps))) 
    # calculate cut-off value to change class values
    limit = n_timesteps/4.0
    # determine the class outcome for each item in cumulative sequence
    y = np.cos(X)
    # reshape input and output data to be suitable for LSTMs
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y
 
# define problem properties
n_timesteps = 100
# define LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(1000):
    # generate new random sequence
    X,y = get_sequence(n_timesteps)
    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate LSTM
X,y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
    print('Expected:', y[0, i], 'Predicted', yhat[0, i])
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
import math
import numpy as np
from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
 
# create a sequence classification instance
def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    start = (random() + n_timesteps) % (2 * math.pi)
    end = start + (2 * math.pi)
    X = np.sin((np.arange(start,end, (end - start) / n_timesteps))) 
    # calculate cut-off value to change class values
    limit = n_timesteps/4.0
    # determine the class outcome for each item in cumulative sequence
    y = np.cos(X)
    # reshape input and output data to be suitable for LSTMs
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y
 
# define problem properties
n_timesteps = 100
# define LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
# train LSTM
for epoch in range(1000):
    # generate new random sequence
    X,y = get_sequence(n_timesteps)
    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate LSTM
X,y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
    print('Expected:', y[0, i], 'Predicted', yhat[0, i])
REPLY

Jason Brownlee January 22, 2018 at 4:45 am # 
I have a long list of general ideas here:
http://machinelearningmastery.com/improve-deep-learning-performance/

REPLY

Harish Yadav January 27, 2018 at 5:54 pm # 
we dont need to modify anything ….just put bidirectional layer…that’t it ….is there anything we have to modify in different problems like [Neural machine translation]…

REPLY

Jason Brownlee January 28, 2018 at 8:22 am # 
Not really.

REPLY

Harish Yadav January 29, 2018 at 8:40 pm # 
how to add attention layer in decoder…..that thing also one line of code i think…

REPLY

Jason Brownlee January 30, 2018 at 9:49 am # 
See these posts:
https://machinelearningmastery.com/?s=attention&submit=Search

REPLY

Shivam February 12, 2018 at 9:48 pm # 
Jason,
Is bidirectional lstm and bidirectional rnn one and the same?

REPLY

Jason Brownlee February 13, 2018 at 8:00 am # 
A bidirectional LSTM is a bidirectional RNN. A bidirectional GRU is also a bidirectional RNN.

REPLY

Ehud Schreiber February 13, 2018 at 2:12 am # 
Hello Jason,

thanks for a very clear and informative post.

My data consists of many time-series of different lengths, which may be very different and grow quite large – from minutes to more than an hour of 1 Hz samples.

Is there a way, therefore, not to specify n_timesteps in the definition of the model, as it doesn’t really need it then, but only when fitting or predicting?

I really wouldn’t want to arbitrarily cut my sequences or pad them with a lot of unnecessary “zeros”.

By the way, I’m trying to detect rather rare occurrences of some event(s) along the sequence; hopefully a sigmoid predicted probability with an appropriate threshold would do the trick.

Thanks,
Ehud.

REPLY

Jason Brownlee February 13, 2018 at 8:05 am # 
You can pad with zeros and use a Mask to ignore the zero values.

I have many examples on the blog.

REPLY

shamsul February 26, 2018 at 12:26 pm # 
sir, can bidirectional lstm be used for sequence or time series forecasting?

REPLY

Jason Brownlee February 26, 2018 at 2:55 pm # 
Perhaps, try it and see.

REPLY
Leave a Reply



Name (required)


Email (will not be published) (required)


Website

Welcome to Machine Learning Mastery

Hi, I'm Jason Brownlee, Ph.D. 
My goal is to make practitioners like YOU awesome at applied machine learning.

Read More

Deep Learning for Sequence Prediction
Cut through the math and research papers.
Discover 4 Models, 6 Architectures, and 14 Tutorials.

Get Started With LSTMs in Python Today!

POPULAR
Your First Machine Learning Project in Python Step-By-Step Your First Machine Learning Project in Python Step-By-Step
JUNE 10, 2016
Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras
JULY 21, 2016
Line Plots of Air Pollution Time Series Multivariate Time Series Forecasting with LSTMs in Keras
AUGUST 14, 2017
How to Setup a Python Environment for Machine Learning and Deep Learning with Anaconda How to Setup a Python Environment for Machine Learning and Deep Learning with Anaconda
MARCH 13, 2017
Tour of Deep Learning Algorithms Develop Your First Neural Network in Python With Keras Step-By-Step
MAY 24, 2016
Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras
JULY 26, 2016
Time Series Forecasting with the Long Short-Term Memory Network in Python Time Series Forecasting with the Long Short-Term Memory Network in Python
APRIL 7, 2017
Multi-Class Classification Tutorial with the Keras Deep Learning Library Multi-Class Classification Tutorial with the Keras Deep Learning Library
JUNE 2, 2016
Regression Tutorial with Keras Deep Learning Library in Python Regression Tutorial with the Keras Deep Learning Library in Python
JUNE 9, 2016
How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras
AUGUST 9, 2016
© 2018 Machine Learning Mastery. All Rights Reserved.

Privacy | Contact | About

Get Your Start in Machine Learning
×
Get Your Start in Machine Learning
You can master applied Machine Learning 
without the math or fancy degree.
Find out how in this free and practical email course.

Email Address
