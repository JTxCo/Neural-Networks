import tempfile
import os
from silence_tensorflow import silence_tensorflow
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import zipfile
import tensorflow_model_optimization as tfmot
from tensorflow import keras

################## STARTER CODE GIVE TO STUDENTS
def convert_model_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converted_model = converter.convert()


    _, tflite_file = tempfile.mkstemp('.tflite')

    with open(tflite_file, 'wb') as f:
        f.write(converted_model)

    return tflite_file


def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)


# Evaluate a TFLITE Model
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for i, test_image in enumerate(test_images):
    if i % 1000 == 0:
      print('Evaluated on {n} results so far.'.format(n=i))
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  print('\n')
  # Compare prediction results with ground truth labels to calculate accuracy.
  prediction_digits = np.array(prediction_digits)
  accuracy = (prediction_digits == test_labels).mean()
  return accuracy



# Step 1: Build baseline MNIST model

batch_size = 32
val_size = 0.1
epochs = 2

mnnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(200, activation=tf.nn.relu),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(
  train_images,
  train_labels,
  epochs=4,
  validation_split=0.1,
)

_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)

model.save('mnist_model.h5', include_optimizer=True)


# Prune the model


# short hand for the pruning API
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude



# prune on the TRAINING data
num_images = train_images.shape[0] * 0.9
print(num_images)
print(train_images.shape[0])


# tell the prune fine tuning how many steps to take
# u[dates steps is related to the number of images and the batch size

end_step = np.ceil(num_images / batch_size).astype(np.uint32) * epochs


# set up pruning parameters
# from the start how many connections do we remove from the trained model
# final sparse is the final percentage of connections to remove at the final model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                            final_sparsity=0.90,
                                                            begin_step=0,
                                                            end_step=end_step)
}


# finish setting up how fine tuning is working
# compile and fit the pruning model

# input params aer going ot be my starting model and the pruning parameters

model_pruned = prune_low_magnitude(model, **pruning_params)

# compile the new prunable model
model_pruned.compile(optimizer='adam', 
                     loss=keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                     metrics=['accuracy'])


callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
]

_,prune_model_start_accuracy = model_pruned.evaluate(test_images, test_labels, verbose=0)
model_pruned.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=val_size, callbacks=callbacks)

_, pruned_model_accuracy_end = model_pruned.evaluate(test_images, test_labels, verbose=0)

print("baseline accuracry: ", baseline_model_accuracy)
print("prusned start acc: ", prune_model_start_accuracy)
print("pruned end acc: ", pruned_model_accuracy_end)

# save model and compare size
# stripping off all the supporting infastructure that did the pruning

model_pruned = tfmot.sparsity.keras.strip_pruning(model_pruned)
model_pruned.save('pruned_mnist_model.h5', include_optimizer=True)

print("orig model size: ",  get_gzipped_model_size('mnist_model.h5'))
print("pruned model size: ", get_gzipped_model_size('pruned_mnist_model.h5'))



# Learnining qunatization with TFLITE

quantize_model = tfmot.quantization.keras.quantize_model

# set up quantized model
model_q = quantize_model(model)

model_q.compile(optimizer='adam', 
                     loss=keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                     metrics=['accuracy'])

model_q.summary()

# fit my quantized model 
model_q.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=val_size)

_, quantized_model_accuracy = model_q.evaluate(test_images, test_labels, verbose=0)


print("baseline accuracry: ", baseline_model_accuracy)
print("pruned end acc: ", pruned_model_accuracy_end)
print("quantized acc: ", quantized_model_accuracy)



# compare size of every H5 file

print("orig model size: ",  get_gzipped_model_size('mnist_model.h5'))
print("pruned model size: ", get_gzipped_model_size('pruned_mnist_model.h5'))
print("quantized model size: ", get_gzipped_model_size('quantized_mnist_model.h5'))



b_tflite = convert_model_tflite(model)
p_tflite = convert_model_tflite(model_pruned)
q_tflite = convert_model_tflite(model_q)


print("TFLITE Baseline model: ", get_gzipped_model_size(b_tflite))
print("TFLITE Pruned model: ", get_gzipped_model_size(p_tflite))
print("TFLITE Quantized model: ", get_gzipped_model_size(q_tflite))
