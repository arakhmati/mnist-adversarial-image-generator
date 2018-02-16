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
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_lambda = features["reg_lambda"]
            loss = loss + reg_lambda * sum(reg_losses)
            
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
              labels=labels,
              predictions=predictions["classes"]
          )
      }
      
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
def get_array_from_results(key, results):
    """Utility function for extracting an array from the results of the predictor."""
    return np.asarray([result[key] for result in results])   
    
def generate_adversarial_images(mnist_classifier,
                              images, labels,
                              old_label, new_label,
                              n_to_modify, reg_lambda,
                              one_pixel=False):
    """Function for generating adversarial images.
    
    Parameters
    ----------
    mnist_classifier : tf.estimator.Estimator,
        Estimator object used to train neural network and make predictions.
    images : numpy array
        Original images.
    labels : numpy array
        Original labels.
    old_label : int
        Label to 'attack'.
    new_label : int
        Adversarial label.
    n_to_modify : int
        Number of images to modify.
    reg_lambda : float
        Regularization lambda.
    one_pixel : bool
        Use a single pixel to create an adversarial image.
        
    Returns
    ----------
    images : numpy array,
        Original images used to generate adversarial inputs.
    adv_images : numpy array,
        Generated adversarial images.
    """
    
     # Obtain all images containing old label
    indices = np.where(labels == old_label)[0]
    images = images[indices]
    labels = labels[indices]
    
    # Classify obtained images
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": images},
        num_epochs=1,
        shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=pred_input_fn)
    pred_results = list(pred_results)
    
    classes       = get_array_from_results("classes", pred_results)
    probabilities = get_array_from_results("probabilities", pred_results)
    
    # Get rid of the misclassified images
    indices       = np.where(classes == old_label)[0]
    images        = images[indices]
    labels        = labels[indices]
    probabilities = probabilities[indices]
    
    if one_pixel:
        # Get rid of the images that have a low probability of adversarial label
        indices = (probabilities[:, new_label] > 0.35)
        images  = images[indices]
        labels  = labels[indices]
    
    # Use only 'n_to_modify' images
    images = images[:n_to_modify]
    labels = labels[:n_to_modify]
    
    # Create adversarial labels and set them to the new label
    adv_images = np.copy(images)
    adv_labels = np.zeros_like(labels) + new_label
    
    # reg_lambda has to be an array with shape [n_to_modify, ...]
    reg_lambda =  np.zeros_like(labels, dtype=np.float32) + reg_lambda
    
    max_grad_indices = None
    
    # Infinite while loop that terminates once all of the images predict new_label
    while True:
        adv_pred_results = mnist_classifier.predict(
                input_fn=tf.estimator.inputs.numpy_input_fn(
                    x={
                        "images":     adv_images,
                        "adv_labels": adv_labels,
                        "reg_lambda": reg_lambda
                    },
                    num_epochs=1,
                    shuffle=False)
                )
        # Convert generator to list to reuse it multiple times
        adv_pred_results = list(adv_pred_results)
        
        # Extract class labels from the results
        classes   = get_array_from_results("classes", adv_pred_results)
        # Extract gradients from the results
        gradients = get_array_from_results("gradients", adv_pred_results)
        
        if one_pixel:
            # For each image. find one pixel with the biggest absolute gradient
            if max_grad_indices is None:
                max_grad_indices = np.argmax(np.abs(gradients), axis=1)
                
            # Set all of the gradients to zero except the maximum absolute one
            for gradient, max_grad_index in zip(gradients, max_grad_indices):
                max_grad = gradient[max_grad_index]
                gradient[:] = 0
                gradient[max_grad_index] = max_grad
        
        # Determine  whether to update an adversarial image further or not
        update = (classes == old_label).reshape(-1, 1)
        
        # Perform an update
        adv_images -= np.multiply(adv_lr*np.sign(gradients), update)
    
        # Print probabilities of adversarial label to see the progress
        print('Adversarial Probabilities = ', end='')
        for pred in adv_pred_results:
            probability_new = pred['probabilities'] * 100
            print('{:10f}'.format(probability_new[new_label]), end='')
        print()
        
        # If all images predict the adversarial label, break out of the loop
        if (classes == new_label).all():
            break
        
    return images, adv_images

def predict(mnist_classifier, images):
    """Utility function for predicting the classes of the images."""
     # Infer the classes of adversarial images
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": images},
        num_epochs=1,
        shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=pred_input_fn)
    classes = get_array_from_results("classes", pred_results)
    return classes
    
if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train',        action='store_true',          help='train the network')
    parser.add_argument('-e', '--evaluate',     action='store_true',          help='run evaluation')
    parser.add_argument('-s', '--steps',        type=int,    default=20000,   help='number of training steps')
    parser.add_argument('-b', '--batch_size',   type=int,    default=64,      help='training batch size')
    parser.add_argument('-o', '--old_label',    type=int,    default=2,       help='label to replace')
    parser.add_argument('-n', '--new_label',    type=int,    default=6,       help='adversarial label')
    parser.add_argument('-m', '--n_to_modify',  type=int,    default=10,      help='number of images to modify')
    parser.add_argument('-l', '--adv_lr',       type=float,  default=0.01,    help='adversarial learning rate')
    parser.add_argument('-r', '--reg_lambda',   type=float,  default=0.0001,  help='regularization lambda')
    parser.add_argument('-p', '--one_pixel',    action='store_true',          help='modify only a single pixel')
    args = parser.parse_args()
    
    # Variables used for training and evaluation 
    train      = args.train
    evaluate   = args.evaluate
    steps      = args.steps
    batch_size = args.batch_size
    
    # Variables used for generation of adversarial images
    old_label    = args.old_label
    new_label    = args.new_label
    n_to_modify  = args.n_to_modify
    adv_lr       = args.adv_lr
    reg_lambda   = args.reg_lambda
    one_pixel    = args.one_pixel
    
    # Load training data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    
        # Initialize the estimator
    mnist_classifier = tf.estimator.Estimator(
                                model_fn=cnn_model_fn,
                                model_dir="mnist_convnet_model/"
                                )
    
    # Train the classifier
    if train:
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
        
    # Obtain test data
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
        accuracy = eval_results["accuracy"]
        if eval_results["accuracy"] < 0.95:
            raise ValueError("Accuracy is {:5}% which is low".format(accuracy*100))
    
    # Generate adversarial images
    images, adv_images = generate_adversarial_images(
                              mnist_classifier,
                              images, labels,
                              old_label, new_label,
                              n_to_modify, reg_lambda,
                              one_pixel)
    
    # Infer the classes
    classes     = predict(mnist_classifier, images)
    adv_classes = predict(mnist_classifier, adv_images)
   
    # Compute the deltas
    deltas = adv_images - images
    
    # Plot and save the figure
    fig = plot(images, deltas, adv_images, classes, adv_classes)
    fig.savefig('challenge.png' if not one_pixel else 'bonus.png')