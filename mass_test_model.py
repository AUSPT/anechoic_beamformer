import os
import time
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import pyroomacoustics as pra
from pesq import pesq
from pystoi import stoi

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILENAME = 'beamformer_anechoic.h5'
TARGET_FOLDER = 'target'
INTERFERENCE_FOLDER = 'interference'

# Number of testing rounds
TEST_ROUNDS = 500 

SAMPLE_RATE = 16000
FRAME_LENGTH = 512
FRAME_STEP = 256
FFT_LENGTH = 512
BATCH_SIZE = 16 


ROOM_DIM = [4.9, 4.9, 4.9]
RT60_TARGET = 0.5
MIC_R = np.array([[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]]).T

# ==========================================
# 2. U-NET MODEL ARCHITECTURE (Inference Mode)
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
    
    # We Frequency to be a power of 2 (256) for U-Net pooling to work cleanly
    # We can crop 257 -> 256. 256 is easier as bin 257 is usually discarded for the purposes of preventing aliasing.
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
        
    return model

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def load_random_file(folder_path, duration=5.0, _cache={}):
    if folder_path not in _cache:
        # Only print indexing once
        if len(_cache) == 0:
            print(f"Indexing folders...")
        files = glob.glob(os.path.join(folder_path, "**", "*.wav"), recursive=True) + \
                glob.glob(os.path.join(folder_path, "**", "*.flac"), recursive=True)
        if not files: raise ValueError(f"No files in {folder_path}")
        _cache[folder_path] = files

    filepath = random.choice(_cache[folder_path])
    try:
        audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
        target_samples = int(duration * SAMPLE_RATE)
        while len(audio) < target_samples:
            audio = np.concatenate([audio, audio])
        start = random.randint(0, len(audio) - target_samples)
        return audio[start:start+target_samples], os.path.basename(filepath)
    except: return load_random_file(folder_path, duration, _cache)

def create_simulation(target, interference):
    room = pra.AnechoicRoom(fs=16000)    
    room.add_microphone_array(MIC_R)
    # Normalise for SIR of 0db 
    rms_target = np.sqrt(np.mean(target**2)) + 1e-8    
    rms_interference = np.sqrt(np.mean(interference**2)) + 1e-8    
    scaler = rms_target / rms_interference
    interference = interference * scaler
    room.add_source(position=[2.45, 3.45, 1.5], signal=target) 
    room.add_source(position=[3.22, 3.06, 1.5], signal=interference)

    room.simulate()
    mix = room.mic_array.signals
    mix = mix / (np.max(np.abs(mix)) + 1e-8)
    
    # 5dB SNR
    sig_power = np.mean(mix ** 2)
    noise_power = sig_power / (10 ** (5 / 10))
    white_noise = np.random.normal(0, np.sqrt(noise_power), mix.shape)
    final_mix = mix + white_noise

    delay_samples = int(1.0 / 343 * SAMPLE_RATE)
    ref_aligned = np.zeros(mix.shape[1])
    ref_aligned[:len(target)] = target
    ref_aligned = np.roll(ref_aligned, delay_samples)
    
    return final_mix, ref_aligned

def compute_snr(clean, estimate):
    min_len = min(len(clean), len(estimate))
    clean = clean[:min_len]
    estimate = estimate[:min_len]
    noise_residual = estimate - clean
    s_power = np.sum(clean ** 2)
    n_power = np.sum(noise_residual ** 2) + 1e-10
    snr = 10 * np.log10(s_power / n_power)
    return snr

def calculate_metrics(clean_ref, processed_sig, mix_sig_mic1):
    min_len = min(len(clean_ref), len(processed_sig), len(mix_sig_mic1))
    clean = clean_ref[:min_len]
    est = processed_sig[:min_len]
    mix = mix_sig_mic1[:min_len]

    # STOI & PESQ
    stoi_clean = stoi(clean, est, SAMPLE_RATE, extended=False)
    stoi_mix = stoi(clean, mix, SAMPLE_RATE, extended=False)
    try:
        pesq_clean = pesq(SAMPLE_RATE, clean, est, 'wb')
        pesq_mix = pesq(SAMPLE_RATE, clean, mix, 'wb')
    except: 
        # Fallback if PESQ fails on silent frames
        pesq_clean = 0.0
        pesq_mix = 0.0

    # SNR Calculation
    snr_input = compute_snr(clean, mix)
    snr_output = compute_snr(clean, est)

    return {
        "STOI_Input": stoi_mix, "STOI_Output": stoi_clean, 
        "PESQ_Input": pesq_mix, "PESQ_Output": pesq_clean,
        "SNR_Input": snr_input, "SNR_Output": snr_output
    }

# ==========================================
# 4. INFERENCE
# ==========================================
def run_ai_beamforming_unet(model, mix_audio):
    stft_1 = librosa.stft(mix_audio[0], n_fft=FFT_LENGTH, hop_length=FRAME_STEP, win_length=FRAME_LENGTH)
    stft_2 = librosa.stft(mix_audio[1], n_fft=FFT_LENGTH, hop_length=FRAME_STEP, win_length=FRAME_LENGTH)

    features = np.stack([np.real(stft_1), np.imag(stft_1), np.real(stft_2), np.imag(stft_2)], axis=-1)
    features = np.transpose(features, (1, 0, 2))
    input_tensor = np.expand_dims(features, axis=0)

    prediction = model.predict(input_tensor, verbose=0)
    
    pred_real = prediction[0, :, :, 0].T
    pred_imag = prediction[0, :, :, 1].T
    
    complex_stft = pred_real + 1j * pred_imag
    enhanced_audio = librosa.istft(complex_stft, hop_length=FRAME_STEP, win_length=FRAME_LENGTH)
    
    return enhanced_audio

def main():
    random.seed(time.time())
    print("------------------------------------------------")
    print(f"   AI EVALUATION: {MODEL_FILENAME}              ")
    print(f"   ROUNDS: {TEST_ROUNDS}                        ")
    print("------------------------------------------------")
    
    model = build_complex_unet()
    
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: {MODEL_FILENAME} not found.")
        return
        
    print(f"Loading weights from {MODEL_FILENAME}...")
    try:
        model.load_weights(MODEL_FILENAME)
        print("Weights Loaded. Starting Tests...")
    except Exception as e:
        print(f"Load Error: {e}")
        return
    
    # Metrics Accumulators
    history = {
        "stoi_in": [], "stoi_out": [],
        "pesq_in": [], "pesq_out": [],
        "snr_in": [], "snr_out": []
    }
    
    start_total_time = time.time()

    for i in range(TEST_ROUNDS):
        # Load and Simulate
        target_sig, _ = load_random_file(TARGET_FOLDER)
        inter_sig, _ = load_random_file(INTERFERENCE_FOLDER)
        mix_matrix, clean_reference = create_simulation(target_sig, inter_sig)
        
        # Inference
        output_audio = run_ai_beamforming_unet(model, mix_matrix)
        
        # Calculate Metrics
        scores = calculate_metrics(clean_reference, output_audio, mix_matrix[0])
        
        # Store Metrics
        history["stoi_in"].append(scores["STOI_Input"])
        history["stoi_out"].append(scores["STOI_Output"])
        history["pesq_in"].append(scores["PESQ_Input"])
        history["pesq_out"].append(scores["PESQ_Output"])
        history["snr_in"].append(scores["SNR_Input"])
        history["snr_out"].append(scores["SNR_Output"])
        
        # Simple progress bar
        if (i + 1) % 10 == 0 or (i + 1) == TEST_ROUNDS:
            print(f"Processed {i + 1}/{TEST_ROUNDS} rounds...")

    total_time = time.time() - start_total_time
    
    # ==========================================
    # FINAL REPORT CARD
    # ==========================================
    avg_stoi_in = np.mean(history["stoi_in"])
    avg_stoi_out = np.mean(history["stoi_out"])
    
    avg_pesq_in = np.mean(history["pesq_in"])
    avg_pesq_out = np.mean(history["pesq_out"])
    
    avg_snr_in = np.mean(history["snr_in"])
    avg_snr_out = np.mean(history["snr_out"])
    
    print("\n" + "="*50)
    print(f" Average metrics over({TEST_ROUNDS} rounds)")
    print("="*50)
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Avg time per round:     {total_time/TEST_ROUNDS:.3f} seconds")
    print("-" * 50)
    print(f"{'METRIC':<10} | {'AVG INPUT':<10} | {'AVG OUTPUT':<10} | {'IMPROVEMENT':<10}")
    print("-" * 50)
    
    print(f"{'STOI':<10} | {avg_stoi_in:.4f}     | {avg_stoi_out:.4f}     | {avg_stoi_out - avg_stoi_in:+.4f}")
    print(f"{'PESQ':<10} | {avg_pesq_in:.4f}     | {avg_pesq_out:.4f}     | {avg_pesq_out - avg_pesq_in:+.4f}")
    print(f"{'OSINR(dB)':<10} | {avg_snr_in:.4f}     | {avg_snr_out:.4f}     | {avg_snr_out - avg_snr_in:+.4f}")
    print("="*50)

if __name__ == "__main__":
    main()