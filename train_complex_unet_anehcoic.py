import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import pyroomacoustics as pra
import librosa
import soundfile as sf
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
SAMPLE_RATE = 16000
FRAME_LENGTH = 512
FRAME_STEP = 256
FFT_LENGTH = 512
BATCH_SIZE = 16 # U-Nets use more memory, reducing batch size slightly
EPOCHS = 100
STEPS_PER_EPOCH = 500

ROOM_DIM = [4.9, 4.9, 4.9]
RT60_TARGET = 0.5
MIC_R = np.array([[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]]).T

TARGET_FOLDER = 'target'
INTERFERENCE_FOLDER = 'interference'

# ==========================================
# 2. COMPLEX DATA GENERATION
# ==========================================

def load_random_file(folder_path, duration=3.0, _cache={}):
    if folder_path not in _cache:
        print(f"Indexing {folder_path}...")
        files = glob.glob(os.path.join(folder_path, "**", "*.wav"), recursive=True) + \
                glob.glob(os.path.join(folder_path, "**", "*.flac"), recursive=True)
        if not files: raise ValueError(f"No files in {folder_path}")
        _cache[folder_path] = files

    filepath = random.choice(_cache[folder_path])
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
    
    target_samples = int(duration * SAMPLE_RATE)
    while len(audio) < target_samples:
        audio = np.concatenate([audio, audio])
    
    start = random.randint(0, len(audio) - target_samples)
    return audio[start:start+target_samples]

def simulate_room_anechoic(target, interference):    
    room = pra.AnechoicRoom(fs=16000)
    room.add_microphone_array(MIC_R)
    
    # Normalise for SIR of 0db 
    rms_target = np.sqrt(np.mean(target**2)) + 1e-8    
    rms_interference = np.sqrt(np.mean(interference**2)) + 1e-8    
    scaler = rms_target / rms_interference
    interference = interference * scaler

    # Sources
    room.add_source(position=[2.45, 3.45, 1.5], signal=target) 
    room.add_source(position=[3.22, 3.06, 1.5], signal=interference)

    room.simulate()
    mix = room.mic_array.signals
    mix = mix / (np.max(np.abs(mix)) + 1e-8)

    # Add White interference
    sig_power = np.mean(mix ** 2)
    interference_power = sig_power / (10 ** (5 / 10))
    white_interference = np.random.normal(0, np.sqrt(interference_power), mix.shape)
    mix_noisy = mix + white_interference

    # Target: target signal aligned (delay ~3ms)
    delay = int(1.0 / 343 * SAMPLE_RATE)
    target_padded = np.zeros(mix.shape[1])
    target_padded[:len(target)] = target
    target_aligned = np.roll(target_padded, delay)

    return mix_noisy.T, target_aligned

def extract_complex_features(mix, target):
    # STFT parameters
    stft_kwargs = {'n_fft': FFT_LENGTH, 'hop_length': FRAME_STEP, 'win_length': FRAME_LENGTH}
    
    # STFT inputs (2 Mics)
    stft_1 = librosa.stft(mix[:, 0], **stft_kwargs)
    stft_2 = librosa.stft(mix[:, 1], **stft_kwargs)
    
    # STFT target
    stft_target = librosa.stft(target, **stft_kwargs)

    # Feature Stacking: [Real1, Imag1, Real2, Imag2]
    # We normalize using a power-law compression (signed) to make training stable
    # Formula: X^0.5 * sign(X) (Approximation)
    # Actually, simpler approach for U-Net: Just standard Real/Imag scaling.
    
    # Stack: (Freq, Time, 4) -> Transpose to (Time, Freq, 4)
    # We treat Frequency as the "Image Height" or "Feature Dimension"
    # For Conv1D/2D, we usually want (Time, Freq, Channels)
    
    X_real1 = np.real(stft_1)
    X_imag1 = np.imag(stft_1)
    X_real2 = np.real(stft_2)
    X_imag2 = np.imag(stft_2)
    
    inputs = np.stack([X_real1, X_imag1, X_real2, X_imag2], axis=-1)
    inputs = np.transpose(inputs, (1, 0, 2)) # (Time, Freq, 4)

    # Targets: [Real, Imag] of target speech
    Y_real = np.real(stft_target)
    Y_imag = np.imag(stft_target)
    
    targets = np.stack([Y_real, Y_imag], axis=-1)
    targets = np.transpose(targets, (1, 0, 2)) # (Time, Freq, 2)

    return inputs.astype(np.float32), targets.astype(np.float32)

def data_generator():
    while True:
        try:
            c = load_random_file(TARGET_FOLDER)
            n = load_random_file(INTERFERENCE_FOLDER)
            m, t = siumulate_room_anechoic(c, n)
            x, y = extract_complex_features(m, t)
            min_len = min(x.shape[0], y.shape[0])
            yield x[:min_len], y[:min_len]
        except Exception: continue

def get_dataset():
    freq_bins = FFT_LENGTH // 2 + 1
    ds = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, freq_bins, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, freq_bins, 2), dtype=tf.float32)
        )
    )
    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=([None, freq_bins, 4], [None, freq_bins, 2]))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ==========================================
# 3. U-NET MODEL ARCHITECTURE
# ==========================================

def conv_block(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def build_complex_unet():
    # Input: (Time, Freq, 4)
    # To use Conv2D, we treat it as an image. 
    # But usually audio U-Nets treat Frequency as the dimension to compress.
    # Let's treat Input as (Batch, Time, Freq, 4)
    
    freq_bins = FFT_LENGTH // 2 + 1 # 257
    
    # We need to pad Frequency to a power of 2 (256) for U-Net pooling to work targetly
    # Or just crop 257 -> 256. 256 is easier.
    inputs = layers.Input(shape=(None, freq_bins, 4))
    
    # Slice to 256 bins
    x_sliced = layers.Lambda(lambda z: z[:, :, :256, :])(inputs)
    
    # --- ENCODER ---
    # Downsample Frequency axis only
    c1 = conv_block(x_sliced, 16, (3, 3))
    p1 = conv_block(c1, 16, (3, 3), strides=(1, 2)) # Freq / 2 -> 128
    
    c2 = conv_block(p1, 32, (3, 3))
    p2 = conv_block(c2, 32, (3, 3), strides=(1, 2)) # Freq / 4 -> 64
    
    c3 = conv_block(p2, 64, (3, 3))
    p3 = conv_block(c3, 64, (3, 3), strides=(1, 2)) # Freq / 8 -> 32
    
    c4 = conv_block(p3, 128, (3, 3))
    p4 = conv_block(c4, 128, (3, 3), strides=(1, 2)) # Freq / 16 -> 16
    
    # --- BOTTLENECK ---
    b = conv_block(p4, 256, (3, 3))
    
    # --- DECODER (Upsampling) ---
    u4 = layers.Conv2DTranspose(128, (3, 3), strides=(1, 2), padding='same')(b)
    u4 = layers.Concatenate()([u4, c4])
    u4 = layers.BatchNormalization()(u4)
    u4 = layers.LeakyReLU(0.2)(u4)
    
    u3 = layers.Conv2DTranspose(64, (3, 3), strides=(1, 2), padding='same')(u4)
    u3 = layers.Concatenate()([u3, c3])
    u3 = layers.BatchNormalization()(u3)
    u3 = layers.LeakyReLU(0.2)(u3)
    
    u2 = layers.Conv2DTranspose(32, (3, 3), strides=(1, 2), padding='same')(u3)
    u2 = layers.Concatenate()([u2, c2])
    u2 = layers.BatchNormalization()(u2)
    u2 = layers.LeakyReLU(0.2)(u2)
    
    u1 = layers.Conv2DTranspose(16, (3, 3), strides=(1, 2), padding='same')(u2)
    u1 = layers.Concatenate()([u1, c1])
    u1 = layers.BatchNormalization()(u1)
    u1 = layers.LeakyReLU(0.2)(u1)
    
    # Output Layer
    # Predict Real and Imag (2 channels)
    outputs_256 = layers.Conv2D(2, (1, 1), activation='linear')(u1)
    
    # Pad back to 257 (Add a zero column)
    outputs = layers.Lambda(lambda z: tf.pad(z, [[0,0], [0,0], [0,1], [0,0]]))(outputs_256)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
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

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(TARGET_FOLDER):
        print("Please ensure 'target' and 'interference' folders exist.")
        exit()

    print("Initializing Complex U-Net...")
    dataset = get_dataset()
    model = build_complex_unet()
    model.summary()
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'train_complex_unet_anehcoic.py', save_best_only=True
        
    )    
    model.fit(
        dataset,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    print("Training Complete.")
