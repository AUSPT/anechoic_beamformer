import os
import time
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import librosa.display
import soundfile as sf
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILENAME = 'beamformer_anechoic.h5'
TARGET_FOLDER = 'target'
INTERFERENCE_FOLDER = 'interference'

SAMPLE_RATE = 16000
FRAME_LENGTH = 512
FRAME_STEP = 256 #Model-dependent - earlier models trained with 256
FFT_LENGTH = 512

ROOM_DIM = [4.9, 4.9, 4.9]
RT60_TARGET = 0.5
MIC_R = np.array([[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]]).T

# ==========================================
# 2. MODEL ARCHITECTURE (V1 - Conv2DTranspose)
# ==========================================
def conv_block(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def build_complex_unet():
    # Input: (Time, Freq, 4)
    # To use Conv2D, we treat it as an image. 
    # But audio U-Nets treat Frequency as the dimension to compress.
    
    freq_bins = FFT_LENGTH // 2 + 1 # 257
    
    # We need to pad Frequency to a power of 2 (256) for U-Net pooling to work cleanly
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
       
    return model

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def load_random_file(folder_path, duration=5.0, _cache={}):
    if folder_path not in _cache:
        print(f"Indexing {folder_path}...")
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

def calculate_sisnr(ref, est):
    """Calculates Scale-Invariant Signal-to-Noise Ratio (SI-SNR)"""
    eps = 1e-8
    
    # Normalization
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    
    # Project estimate onto reference (finding the scaling factor)
    ref_energy = np.sum(ref ** 2) + eps
    scaling_factor = np.dot(est, ref) / ref_energy
    
    # The part of the output that is actually the target signal
    target_projection = scaling_factor * ref
    
    # The part of the output that is noise/error
    noise_residual = est - target_projection
    
    # Calculate Ratios
    target_pow = np.sum(target_projection ** 2) + eps
    noise_pow = np.sum(noise_residual ** 2) + eps
    
    return 10 * np.log10(target_pow / noise_pow)

def calculate_metrics(clean_ref, processed_sig, mix_sig_mic1):
    min_len = min(len(clean_ref), len(processed_sig), len(mix_sig_mic1))
    clean = clean_ref[:min_len]
    est = processed_sig[:min_len]
    mix = mix_sig_mic1[:min_len]

    # STOI
    stoi_clean = stoi(clean, est, SAMPLE_RATE, extended=False)
    stoi_mix = stoi(clean, mix, SAMPLE_RATE, extended=False)
    
    # PESQ
    try:
        pesq_clean = pesq(SAMPLE_RATE, clean, est, 'wb')
        pesq_mix = pesq(SAMPLE_RATE, clean, mix, 'wb')
    except: 
        pesq_clean = 0.0
        pesq_mix = 0.0

    # SNR (SI-SNR)
    snr_clean = calculate_sisnr(clean, est)   # OSINR
    snr_mix = calculate_sisnr(clean, mix)     # Input SNR

    return {
        "STOI_Input": stoi_mix, "STOI_Output": stoi_clean, 
        "PESQ_Input": pesq_mix, "PESQ_Output": pesq_clean,
        "SNR_Input": snr_mix,   "SNR_Output": snr_clean
    }
# ==========================================
# 4. SPECTROGRAM
# ==========================================
def save_comparative_spectrogram(ref, mix, out, filename):
    plt.figure(figsize=(10, 10))
    
    # Convert to STFT
    D_ref = librosa.stft(ref, n_fft=512, hop_length=160)
    D_mix = librosa.stft(mix, n_fft=512, hop_length=160)
    D_out = librosa.stft(out, n_fft=512, hop_length=160)
    
    # DB
    S_ref = librosa.amplitude_to_db(np.abs(D_ref), ref=np.max)
    S_mix = librosa.amplitude_to_db(np.abs(D_mix), ref=np.max)
    S_out = librosa.amplitude_to_db(np.abs(D_out), ref=np.max)

    # Plot
    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_ref, sr=SAMPLE_RATE, hop_length=160, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Target (Clean)')
    plt.xlabel('')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_mix, sr=SAMPLE_RATE, hop_length=160, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Input Mixture (Target + Interference @ RT50 = 0.5s + WGN) (Noisy)')
    plt.xlabel('')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_out, sr=SAMPLE_RATE, hop_length=160, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Deep Learning Output (Reconstructed)')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ==========================================
# 5. INFERENCE (COMPLEX U-NET)
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
    print("------------------------------------------------")
    
    model = build_complex_unet()
    
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: {MODEL_FILENAME} not found.")
        return
        
    print(f"Loading weights from {MODEL_FILENAME}...")
    try:
        model.load_weights(MODEL_FILENAME)
        print("Weights Loaded.")
    except Exception as e:
        print(f"Load Error: {e}")
        return
    
    iteration = 1
    
    while True:
        print(f"\n[Test Case #{iteration}]")
        target_sig, t_name = load_random_file(TARGET_FOLDER)
        inter_sig, i_name = load_random_file(INTERFERENCE_FOLDER)
        
        print(f"Target: {t_name}")
        print(f"Noise:  {i_name}")
        
        # mix_matrix comes out as (Time, Channels), e.g., (80000, 2)
        mix_matrix, clean_reference = simulate_room_anechoic(target_sig, inter_sig)
        
        start_t = time.time()
        
        # FIX 1: Transpose here so the AI gets (Channels, Time)
        # The AI needs to separate mix_matrix[0] (Mic 1) and mix_matrix[1] (Mic 2)
        output_audio = run_ai_beamforming_unet(model, mix_matrix.T) 
        
        proc_time = time.time() - start_t
        
        f_in = f"eval_{iteration:02d}_INPUT.flac" # Changed to lowercase .flac for safety
        f_out = f"eval_{iteration:02d}_OUTPUT.flac"
        f_ref = f"eval_{iteration:02d}_REF.flac"
        f_img = f"eval_{iteration:02d}_SPECTROGRAM.png"
        
        # FIX 2: Do NOT transpose here. 
        # Soundfile expects (Samples, Channels). mix_matrix is already (Samples, Channels).
        sf.write(f_in, mix_matrix, SAMPLE_RATE)
        
        sf.write(f_out, output_audio, SAMPLE_RATE)
        sf.write(f_ref, clean_reference, SAMPLE_RATE)
        
        print("Generating Spectrograms...")
        # Note: mix_matrix[0] here refers to the first sample (length 2), which is wrong for plotting.
        # We need the first channel.
        # FIX 3: Slice the column for the spectrogram (All time samples, Channel 0)
        save_comparative_spectrogram(clean_reference, mix_matrix[:, 0], output_audio, f_img)
        # FIX: pass separate channels for AI (Channels, Time)
        output_audio = run_ai_beamforming_unet(model, mix_matrix.T) 
        print("Calculating Metrics...")
        # FIX: Pass 1D array (All time, Channel 0)
        scores = calculate_metrics(clean_reference, output_audio, mix_matrix[:, 0])
        
        print("\n" + "="*40)
        print(f" REPORT CARD (Case #{iteration})")
        print("="*40)
        print(f"{'METRIC':<10} | {'INPUT':<10} | {'OUTPUT':<10} | {'CHANGE':<10}")
        print("-" * 46)
        
        stoi_imp = scores['STOI_Output'] - scores['STOI_Input']
        print(f"{'STOI':<10} | {scores['STOI_Input']:.3f}      | {scores['STOI_Output']:.3f}      | {stoi_imp:+.3f}")
        
        pesq_imp = scores['PESQ_Output'] - scores['PESQ_Input']
        print(f"{'PESQ':<10} | {scores['PESQ_Input']:.3f}      | {scores['PESQ_Output']:.3f}      | {pesq_imp:+.3f}")

        # NEW: OSINR
        snr_imp = scores['SNR_Output'] - scores['SNR_Input']
        print(f"{'OSINR (dB)':<10} | {scores['SNR_Input']:.3f}      | {scores['SNR_Output']:.3f}      | {snr_imp:+.3f}")
        
        print("-" * 46)
        print(f"Spectrogram: {f_img}")
        
        choice = input("\nPress [ENTER] to continue, or 'q' to quit: ")
        if choice.lower() == 'q':
            break
        iteration += 1

if __name__ == "__main__":
    main()
