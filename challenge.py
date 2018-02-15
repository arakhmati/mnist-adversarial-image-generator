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

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        # If adversarial labels were passed, compute the gradients
        if 'adv_labels' in features: 
            loss = tf.losses.sparse_softmax_cross_entropy(labels=features["adv_labels"], logits=logits)
            gradients = tf.gradients(loss, [input_layer])[0]
            predictions['gradients'] = tf.reshape(gradients, (-1, 784))
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

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
    
if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train',        action='store_true',          help='train the network')
    parser.add_argument('-e', '--evaluate',     action='store_true',          help='run evaluation')
    parser.add_argument('-s', '--steps',        type=int,    default=20000,   help='number of training steps')
    parser.add_argument('-b', '--batch_size',   type=int,    default=64,      help='training batch size')
    parser.add_argument('-o', '--old_label',    type=int,    default=2,       help='label to replace')
    parser.add_argument('-n', '--new_label',    type=int,    default=6,       help='adversarial label')
    parser.add_argument('-r', '--n_to_replace', type=int,    default=10,      help='number of images to replace')
    parser.add_argument('-l', '--adv_lr',       type=float,  default=0.001,   help='adversarial learning rate')
    args = parser.parse_args()
    
    train      = args.train
    evaluate   = args.evaluate
    steps      = args.steps
    batch_size = args.batch_size
    
    old_label    = args.old_label
    new_label    = args.new_label
    n_to_replace = args.n_to_replace
    adv_lr       = args.adv_lr
    
    
    # Load training data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    
        # Initialize the estimator
    mnist_classifier = tf.estimator.Estimator(
                                model_fn=cnn_model_fn,
                                model_dir="mnist_convnet_model/"
                                )
    
    if train:
        ################ PART 1 - CLASSIFICATION #########################
        train_data = mnist.train.images # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        # Create the Estimator
        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": train_data},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True)
        mnist_classifier.train(input_fn=train_input_fn,steps=steps)
    
    #### PART 2 - GENERATION OF ADVERSARIAL IMAGES ####################
    # Use test data
    images = mnist.test.images
    labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    # Evaluate the accuracy to make sure that the model is trained well
    if evaluate:
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": images},
            y=labels,
            batch_size=batch_size,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print('{}'.format(eval_results))
    
    # Obtain the indices of the images containing old label
    indices = np.where(labels == old_label)[0][:n_to_replace]
    
    # Obtain all the images containing old label
    images = images[indices]
    labels = labels[indices]
    
    # Create adversarial labels and set them to the new label
    adv_images = np.copy(images)
    adv_labels = np.zeros_like(labels) + new_label
    
    while True:
        adv_pred_results = mnist_classifier.predict(
                input_fn=tf.estimator.inputs.numpy_input_fn(
                    x={
                        "images":     adv_images,
                        "adv_labels": adv_labels
                    },
                    num_epochs=1,
                    shuffle=False)
                )
        # Convert generator to list to reuse it later
        adv_pred_results = list(adv_pred_results)
        
        
        # Obtain class labels
        classes   = [pred['classes'] for pred in adv_pred_results]
    
        # Reshape gradients and update the adversarial image
        gradients = np.asarray([pred['gradients'] for pred in adv_pred_results])
        adv_images -= adv_lr * np.sign(gradients)
#        plot(images, gradients, adv_images, classes)
    
        # Print adversarial probabilities to see the progress
        adv_probabilities = []
        for pred in adv_pred_results:
            probabilities = pred['probabilities'] * 100
            adv_probabilities.append(probabilities[new_label])
        print('Adversarial Probabilities = {}'.format(adv_probabilities))
        
        # If all images predict the adversarial label, break out of the loop
        if (np.asarray(classes) == new_label).all():
            break
    
    # Infer the classes of adversarial images
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": adv_images},
        num_epochs=1,
        shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=pred_input_fn)
    classes   = [pred['classes'] for pred in pred_results]
    
    # Compute the deltas
    deltas = adv_images - images
    
    # Plot and save the figure
    fig = plot(images, deltas, adv_images, classes)
    fig.savefig('challenge.png')