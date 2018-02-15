from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import argparse
import numpy as np
import tensorflow as tf

from plot_utils import plot

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    
    # Input Layer
    input_layer = tf.reshape(features['images'], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    train_op = None
    
#    # If adversarial labels were passed,
#    # use them instead of the actual ones to compute the loss
    if "adv_labels" in features: 
        labels = features["adv_labels"]
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        # If adversarial labels were passed, compute the gradients
        if 'adv_labels' in features: 
            gradients = tf.gradients(loss, [input_layer])[0]
            predictions['gradients'] = tf.reshape(gradients, [-1, 784])
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)    
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
      
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main():
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--steps',      type=int, default=1, help='number of training steps')
    parser.add_argument('-b', '--batch_size', type=int, default=64,   help='training batch size')
    args = parser.parse_args()
    
    steps      = args.steps
    batch_size = args.batch_size
    
    # Load training data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  
    ################ PART 1 - CLASSIFICATION #########################
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="mnist_convnet_model/")
  
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=steps)
  
    #### PART 2 - GENERATION OF ADVERSARIAL IMAGES ####################
    # Use test data
    images = mnist.test.images
    labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    # Obtain the indices of the images containing 2
    indices = np.where(labels == 2)[0][:10]
    
    # Obtain all the images containing 2
    images = images[indices]
    labels = labels[indices]
    
    # Create adversarial labels and set them to 6
    adv_labels = np.zeros_like(labels) + 6
    
    adv_pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": images,
           "adv_labels": adv_labels},
        num_epochs=1,
        shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=adv_pred_input_fn)
    pred_results = list(pred_results)
    
    classes   = [pred['classes'] for pred in pred_results]
    gradients = np.asarray([pred['gradients'] for pred in pred_results])
    
    adv_images = images + gradients
    
    plot(images, gradients, adv_images, classes)
  
if __name__ == "__main__":
    main()