import os
import numpy as np
import tensorflow as tf
from test import build_model

def export_weights(output_path='weights_tf.npy'):
    print("Building Keras model...")
    model = build_model()
    print("Loading weights from persfective.h5...")
    model.load_weights('persfective.h5')
    
    weights_dict = {}
    print("Extracting weights from layers...")
    for layer in model.layers:
        for weight in layer.weights:
            # Create a unique name using layer name and weight name
            clean_w_name = weight.name.replace(':0', '')
            unique_name = f"{layer.name}/{clean_w_name}"
            weights_dict[unique_name] = weight.numpy()
            # print(f"    - {unique_name} | {weight.shape}")
    
    print(f"Exporting {len(weights_dict)} unique weights to {output_path}...")
    np.save(output_path, weights_dict)
    print("Export complete.")

if __name__ == "__main__":
    export_weights()
