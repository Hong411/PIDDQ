# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:36:17 2025

@author: Bo
"""
import utils
import gpflow
from gpflow.utilities import set_trainable
import tensorflow as tf
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tensorflow.keras import layers, models
from typing import List, Optional, Tuple, Union
import os
from utils.config import PATH_CONFIG

#from sklearn.gaussian_process.kernels import ConstantKernel as C

"""
Gaussian Process Regression operations using both TensorFlow/GPflow and scikit-learn
"""

class GaussianProcessOperations:
    """
    Handles Gaussian Process Regression operations with both TensorFlow/GPflow and scikit-learn
    """
    
    @staticmethod
    def train_tf(
        X_list: Union[List[List[float]], np.ndarray],
        Y_list: Union[List[List[float]], np.ndarray],
        model_name: Optional[str] = None,
        length_scale: float = 1.0,
        variance: float = 1.0,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        noise_variance: float = 1e-3
    ) -> gpflow.models.GPR:
        """
        Train GPflow GPR model with GPU acceleration
        
        Args:
            X_list: Input features (2D array-like)
            Y_list: Target values (2D array-like)
            model_name: Optional name to save the model
            length_scale: Initial length scale for kernel
            variance: Initial variance for kernel
            learning_rate: Learning rate for Adam optimizer
            iterations: Number of training iterations
            noise_variance: Fixed noise variance
            
        Returns:
            Trained GPflow GPR model
        """
        # Convert to tensors
        X_train = tf.convert_to_tensor(X_list, dtype=tf.float64)
        y_train = tf.convert_to_tensor(Y_list, dtype=tf.float64)
        
        # Create and configure model
        kernel = gpflow.kernels.SquaredExponential(
            lengthscales=length_scale, 
            variance=variance
        )
        gpr = gpflow.models.GPR(
            data=(X_train, y_train), 
            kernel=kernel,
            noise_variance=noise_variance
        )
        set_trainable(gpr.likelihood.variance, False)
        
        # Optimization
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        
        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                loss = gpr.training_loss()
            gradients = tape.gradient(loss, gpr.trainable_variables)
            optimizer.apply_gradients(zip(gradients, gpr.trainable_variables))
            return loss
        
        # Training loop
        for _ in range(iterations):
            optimization_step()
        
        if model_name:
            utils.save_model(gpr, model_name)
            
        return gpr
    
    @staticmethod
    def predict_tf(
        model: gpflow.models.GPR,
        X_test: Union[List[List[float]], np.ndarray],
        return_variance: bool = False
    ) -> Union[List[float], Tuple[List[float], List[float]]]:
        """
        Make predictions using trained GPflow GPR model
        
        Args:
            model: Trained GPflow model
            X_test: Input features for prediction
            return_variance: Whether to return predictive variance
            
        Returns:
            Predictive mean (and variance if requested)
        """
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float64)
        mean, variance = model.predict_f(X_test)
        
        if return_variance:
            return mean.numpy().tolist(), variance.numpy().tolist()
        return mean.numpy().tolist()
    
    @staticmethod
    def train_sklearn(
        X_list: Union[List[List[float]], np.ndarray],
        Y_list: Union[List[List[float]], np.ndarray],
        model_name: Optional[str] = None,
        length_scale: float = 1.0,
        alpha: float = 1e-3,
        n_restarts: int = 50
    ) -> GaussianProcessRegressor:
        """
        Train scikit-learn GPR model
        
        Args:
            X_list: Input features (2D array-like)
            Y_list: Target values (2D array-like)
            model_name: Optional name to save the model
            length_scale: Initial length scale for kernel
            alpha: Value added to diagonal of kernel matrix
            n_restarts: Number of restarts for optimizer
            
        Returns:
            Trained scikit-learn GPR model
        """
        X_train = np.array(X_list)
        y_train = np.array(Y_list)
        
        kernel = 1.0 * RBF(length_scale=length_scale)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts
        )
        gpr.fit(X_train, y_train)
        
        if model_name:
            utils.save_model(gpr, model_name)
            
        return gpr
    
    @staticmethod
    def predict_sklearn(
        model: GaussianProcessRegressor,
        X_test: Union[List[List[float]], np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions using trained scikit-learn GPR model
        
        Args:
            model: Trained scikit-learn model
            X_test: Input features for prediction
            
        Returns:
            Predictive mean
        """
        return model.predict(X_test)
    
    @staticmethod
    def save_model(model: Union[gpflow.models.GPR, GaussianProcessRegressor], name: str):
        """Unified model saving"""
        utils.save_model(model, name)
    
    @staticmethod
    def load_model(name: str) -> Union[gpflow.models.GPR, GaussianProcessRegressor]:
        """Unified model loading"""
        return utils.load_model(name)
    
    @staticmethod
    def create_tf_kernel(kernel_type: str = 'RBF', **kwargs):
        """Factory for GPflow kernels"""
        kernels = {
            'RBF': gpflow.kernels.SquaredExponential,
            'Matern12': gpflow.kernels.Matern12,
            'Matern32': gpflow.kernels.Matern32,
            'Matern52': gpflow.kernels.Matern52
        }
        return kernels[kernel_type](**kwargs)

# Create singleton instance for easy access
gpr_operations = GaussianProcessOperations()

"""
Neural network model operations using TensorFlow/Keras
Includes training and prediction functions with GPU support
"""

class NeuralNetworkOperations:
    """
    Handles neural network training and prediction operations with TensorFlow
    """

    @staticmethod
    def train(
        X_list: Union[List[List[float]], np.ndarray],
        Y_list: Union[List[List[float]], np.ndarray],
        model_name: Optional[str] = None,
        epochs: int = 1000,
        batch_size: int = 32,
        verbose: int = 1
    ) -> models.Sequential:
        """
        Train a neural network model with GPU acceleration
        
        Args:
            X_list: Input features (2D array-like)
            Y_list: Target values (2D array-like)
            model_name: Optional name to save the model
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode (0 = silent, 1 = progress bar)
            
        Returns:
            Trained Keras Sequential model
        """
        # Convert data to tensors
        X_train = tf.convert_to_tensor(X_list, dtype=tf.float32)
        y_train = tf.convert_to_tensor(Y_list, dtype=tf.float32)
        
        # Define model architecture
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(X_train.shape[1], activation='linear')
        ])
        
        # Compile and train
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, 
                 batch_size=batch_size, verbose=verbose)
        
        # Save if name provided
        if model_name:
            utils.save_model_nn(model, model_name)
            
        return model
    
    @staticmethod
    def predict(
        model: models.Sequential,
        X_test: Union[List[List[float]], np.ndarray]
    ) -> List[List[float]]:
        """
        Make predictions using a trained neural network
        
        Args:
            model: Trained Keras model
            X_test: Input features for prediction
            
        Returns:
            List of predictions
        """
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        predictions = model.predict(X_test)
        return predictions.tolist()
    
    

# Create singleton instance for easy access
nn_operations = NeuralNetworkOperations()