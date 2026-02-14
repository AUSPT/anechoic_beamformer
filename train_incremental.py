import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import librosa

# =========================
# SETUP
# =========================
DATA_DIR = 'temp_training_data'
INPUT_DIR = os.path.join(DATA_DIR, 'input_gsc')
TARGET_DIR = os.path.join(DATA_DIR, 'ground_truth')
MODEL_FILE = 'gsc_unet_model.keras'

# Args for controlling epochs from MATLAB
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2)
args = parser.parse_args()

SAMPLE_RATE = 16000
FFT_LENGTH = 1024
HOP_LENGTH = 256
BATCH_SIZE = 16

# =========================
# DATA PIPELINE
# =========================
def get_dataset():
    # Find files
    x_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.wav")))
    y_files = sorted(glob.glob(os.path.join(TARGET_DIR, "*.wav")))
    
    if not x_files:
        print("Python: No files found. Exiting.")
        exit(0)


    # Helper to process audio
    def process_path(x_path, y_path):
        # Load
        x, _ = librosa.load(x_path.decode(), sr=SAMPLE_RATE)
        y, _ = librosa.load(y_path.decode(), sr=SAMPLE_RATE)
        
        
        # Match lengths
        m = min(len(x), len(y))
        x, y = x[:m], y[:m]
        
        # STFT
        S_x = librosa.stft(x, n_fft=FFT_LENGTH, hop_length=HOP_LENGTH)
        S_y = librosa.stft(y, n_fft=FFT_LENGTH, hop_length=HOP_LENGTH)
        
        # Stack Real/Imag: (Freq, Time, 2)
        X_stack = np.stack([np.real(S_x), np.imag(S_x)], axis=-1)
        Y_stack = np.stack([np.real(S_y), np.imag(S_y)], axis=-1)
        
        # Transpose to (Time, Freq, 2)
        X_fin = np.transpose(X_stack, (1, 0, 2)).astype(np.float32)
        Y_fin = np.transpose(Y_stack, (1, 0, 2)).astype(np.float32)
        
        return X_fin, Y_fin

    # TF Wrapper
    def tf_process(x, y):
        x, y = tf.numpy_function(process_path, [x, y], [tf.float32, tf.float32])
        x.set_shape([None, 513, 2])
        y.set_shape([None, 513, 2])
        return x, y

    ds = tf.data.Dataset.from_tensor_slices((x_files, y_files))
    ds = ds.map(tf_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=([None, 513, 2], [None, 513, 2]))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# =========================
# MODEL
# =========================
def conv_block(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def build_model():
    # Input: (Time, Freq, 2) -> Real, Imag of GSC Output
    inputs = layers.Input(shape=(None, 513, 2))
    
    # Slice to 512 bins for easy downsampling
    x = layers.Lambda(lambda z: z[:, :, :512, :])(inputs)
    
    # --- ENCODER ---
    c1 = layers.Conv2D(32, 3, padding='same', activation='linear')(x)
    c1 = layers.LeakyReLU(0.2)(c1)
    p1 = layers.MaxPooling2D((1, 2))(c1)
    
    c2 = layers.Conv2D(64, 3, padding='same', activation='linear')(p1)
    c2 = layers.LeakyReLU(0.2)(c2)
    p2 = layers.MaxPooling2D((1, 2))(c2)
    
    c3 = layers.Conv2D(128, 3, padding='same', activation='linear')(p2)
    c3 = layers.LeakyReLU(0.2)(c3)
    p3 = layers.MaxPooling2D((1, 2))(c3)
    
    # --- BOTTLENECK ---
    b = layers.Conv2D(256, 3, padding='same', activation='linear')(p3)
    b = layers.LeakyReLU(0.2)(b)
    
    # --- DECODER ---
    u3 = layers.Conv2DTranspose(128, 3, strides=(1, 2), padding='same')(b)
    u3 = layers.Concatenate()([u3, c3])
    u3 = layers.Conv2D(128, 3, padding='same', activation='relu')(u3)
    
    u2 = layers.Conv2DTranspose(64, 3, strides=(1, 2), padding='same')(u3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)

    u1 = layers.Conv2DTranspose(32, 3, strides=(1, 2), padding='same')(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = layers.Conv2D(32, 3, padding='same', activation='relu')(u1)
    
    # Output: 2 Channels (Real, Imag)
    out = layers.Conv2D(2, 1, activation='linear')(u1)
    
    # Pad back to 513 bins
    out = layers.Lambda(lambda z: tf.pad(z, [[0,0], [0,0], [0,1], [0,0]]))(out)
    
    model = models.Model(inputs, out)

  # Custom Loss: Weighted Magnitude + Complex Error
    def complex_loss(y_true, y_pred):
        # y: (Batch, Time, Freq, 2)
        real_true = y_true[:, :, :, 0]
        imag_true = y_true[:, :, :, 1]
        real_pred = y_pred[:, :, :, 0]
        imag_pred = y_pred[:, :, :, 1]
        
        # 1. Complex MAE
        loss_complex = tf.abs(real_true - real_pred) + tf.abs(imag_true - imag_pred)
        
        # 2. Magnitude MAE (Helps convergence)
        mag_true = tf.sqrt(real_true**2 + imag_true**2 + 1e-8)
        mag_pred = tf.sqrt(real_pred**2 + imag_pred**2 + 1e-8)
        loss_mag = tf.abs(mag_true - mag_pred)
        
        return tf.reduce_mean(loss_complex + loss_mag)

    model.compile(optimizer='adam', loss=complex_loss)
    return model


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    dataset = get_dataset()
    model = build_model()
    if os.path.exists(MODEL_FILE):
        print(f"Python: Resuming training from {MODEL_FILE}")
        # We need custom_objects to load the custom loss function
        model.load_weights(MODEL_FILE)
    else:
        print("Python: Creating new model")
        model = build_model()
    model.summary()  
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'train_beamform_clean_anehcoic.keras', save_best_only=True
        
    )    
    model.fit(
        dataset,
        steps_per_epoch=50,
        epochs=100,
        callbacks=[checkpoint]
    )
    
    model.save(MODEL_FILE)
    print("Python: Model saved.")