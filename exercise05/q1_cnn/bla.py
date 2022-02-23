import tensorflow as tf

print("SEX")

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.filter(lambda x: x < 3)
print(list(dataset.as_numpy_iterator()))