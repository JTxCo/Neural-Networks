import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from keras.layers import Dense
from keras.models import Sequential


# create input data, sine wave

x = np.arange(0, math.pi*2, 0.1)
y = (np.sin(x) + 1)/2

# build NN from scratch
model_sine = Sequential([
    Dense(1, activation='linear', input_shape=(1,)),
    Dense(30, activation='relu'), 
    Dense(1, activation='linear')
])
# dumps the shape and params of your NN
model_sine.summary()

# compile our model, specify loss and optimizer
#mean squared error is loss calculation
#adam is optimizer
# success print metric is mean average error
model_sine.compile(loss='mse', optimizer='adam', metrics=['mae'])

# train the model
#epochs is number of times it will run through the data
# batch size is how many data points it will look at at once
# verbose is how much output you want to see
model_sine.fit(x, y, epochs=500, batch_size=1, verbose=1)

# perdict 
# perdict is throw some data into the model and see what it outputs
output = model_sine.predict(x)

print(output.shape)
print(x.shape)

# plot the results
plt.plot(x, y, 'bo', x, output, 'ro')
plt.show()