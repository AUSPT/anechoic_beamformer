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
MODEL_FILENAME = 'gsc_unet_model.keras' # Ensure this matches your saved model name
TARGET_FOLDER = 'target'
INTERFERENCE_FOLDER = 'interference'

SAMPLE_RATE = 16000
# UPDATED TO MATCH TRAINING SCRIPT
FFT_LENGTH = 1024  
FRAME_STEP = 256   
FRAME_LENGTH = 1024 

ROOM_DIM = [4.9, 4.9, 4.9]
MIC_R = np.array([[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]]).T

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
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
    
    # Normalise for SIR
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

    # Target: target signal aligned
    delay = int(1.0 / 343 * SAMPLE_RATE)
    target_padded = np.zeros(mix.shape[1])
    target_padded[:len(target)] = target
    target_aligned = np.roll(target_padded, delay)

    return mix_noisy.T, target_aligned

def calculate_sisnr(ref, est):
    eps = 1e-8
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = np.sum(ref ** 2) + eps
    scaling_factor = np.dot(est, ref) / ref_energy
    target_projection = scaling_factor * ref
    noise_residual = est - target_projection
    return 10 * np.log10((np.sum(target_projection ** 2) + eps) / (np.sum(noise_residual ** 2) + eps))

def calculate_metrics(clean_ref, processed_sig, mix_sig_mic1):
    min_len = min(len(clean_ref), len(processed_sig), len(mix_sig_mic1))
    clean = clean_ref[:min_len]
    est = processed_sig[:min_len]
    mix = mix_sig_mic1[:min_len]

    stoi_clean = stoi(clean, est, SAMPLE_RATE, extended=False)
    stoi_mix = stoi(clean, mix, SAMPLE_RATE, extended=False)
    
    try:
        pesq_clean = pesq(SAMPLE_RATE, clean, est, 'wb')
        pesq_mix = pesq(SAMPLE_RATE, clean, mix, 'wb')
    except: 
        pesq_clean = 0.0
        pesq_mix = 0.0

    snr_clean = calculate_sisnr(clean, est)
    snr_mix = calculate_sisnr(clean, mix)

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
    
    # UPDATED n_fft to 1024 for visualization consistency
    D_ref = librosa.stft(ref, n_fft=1024, hop_length=256)
    D_mix = librosa.stft(mix, n_fft=1024, hop_length=256)
    D_out = librosa.stft(out, n_fft=1024, hop_length=256)
    
    S_ref = librosa.amplitude_to_db(np.abs(D_ref), ref=np.max)
    S_mix = librosa.amplitude_to_db(np.abs(D_mix), ref=np.max)
    S_out = librosa.amplitude_to_db(np.abs(D_out), ref=np.max)

    plt.subplot(3, 1, 1)
    librosa.display.specshow(S_ref, sr=SAMPLE_RATE, hop_length=256, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Target (Clean)')

    plt.subplot(3, 1, 2)
    librosa.display.specshow(S_mix, sr=SAMPLE_RATE, hop_length=256, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Input (Noisy Mic 1)')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(S_out, sr=SAMPLE_RATE, hop_length=256, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Deep Learning Output')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ==========================================
# 5. INFERENCE (UPDATED)
# ==========================================
def run_ai_beamforming_unet(model, single_channel_audio):
    """
    Adapts the audio to match the training script:
    1. STFT with n_fft=1024 (513 bins)
    2. Stack Real/Imag -> (Freq, Time, 2)
    3. Transpose -> (Time, Freq, 2)
    """
    
    # STFT
    stft = librosa.stft(single_channel_audio, n_fft=FFT_LENGTH, hop_length=FRAME_STEP)
    
    # Stack Real/Imag -> (513, Time, 2)
    features = np.stack([np.real(stft), np.imag(stft)], axis=-1)
    
    # Transpose to (Time, 513, 2)
    features = np.transpose(features, (1, 0, 2))
    
    # Add Batch Dimension -> (1, Time, 513, 2)
    input_tensor = np.expand_dims(features, axis=0)

    # Predict
    prediction = model.predict(input_tensor, verbose=0)
    
    # Unpack -> (Time, 513, 2)
    pred_real = prediction[0, :, :, 0]
    pred_imag = prediction[0, :, :, 1]
    
    # Transpose back to (513, Time) for ISTFT
    pred_real = pred_real.T
    pred_imag = pred_imag.T
    
    complex_stft = pred_real + 1j * pred_imag
    enhanced_audio = librosa.istft(complex_stft, hop_length=FRAME_STEP, n_fft=FFT_LENGTH)
    
    return enhanced_audio

def main():
    random.seed(time.time())
    print("------------------------------------------------")
    print(f"   AI EVALUATION: {MODEL_FILENAME}              ")
    print("------------------------------------------------")
    
    # Create the model with the logic defined in training
    model = build_model()
    
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: {MODEL_FILENAME} not found.")
        # If model file is missing, we return, but if you want to test the loop with uninitialized weights, comment out return
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
        
        # mix_matrix is (Samples, 2)
        mix_matrix, clean_reference = simulate_room_anechoic(target_sig, inter_sig)
        
        start_t = time.time()
        
        # === CRITICAL CHANGE FOR HYBRID MODEL ===
        # Your model expects a SINGLE audio input (the GSC output).
        # Since we don't have the GSC MATLAB code running here, we will feed
        # Microphone 1 as the "Noisy Input". 
        # If you have a Python equivalent of your GSC, apply it to mix_matrix here first!
        input_signal = mix_matrix[:, 0] 
        
        output_audio = run_ai_beamforming_unet(model, input_signal) 
        
        proc_time = time.time() - start_t
        
        f_in = f"eval_{iteration:02d}_INPUT.flac" 
        f_out = f"eval_{iteration:02d}_OUTPUT.flac"
        f_ref = f"eval_{iteration:02d}_REF.flac"
        f_img = f"eval_{iteration:02d}_SPECTROGRAM.png"
        
        sf.write(f_in, input_signal, SAMPLE_RATE)
        sf.write(f_out, output_audio, SAMPLE_RATE)
        sf.write(f_ref, clean_reference, SAMPLE_RATE)
        
        print("Generating Spectrograms...")
        save_comparative_spectrogram(clean_reference, input_signal, output_audio, f_img)
        
        print("Calculating Metrics...")
        scores = calculate_metrics(clean_reference, output_audio, input_signal)
        
        print("\n" + "="*40)
        print(f" REPORT CARD (Case #{iteration})")
        print("="*40)
        print(f"{'METRIC':<10} | {'INPUT':<10} | {'OUTPUT':<10} | {'CHANGE':<10}")
        print("-" * 46)
        
        stoi_imp = scores['STOI_Output'] - scores['STOI_Input']
        print(f"{'STOI':<10} | {scores['STOI_Input']:.3f}      | {scores['STOI_Output']:.3f}      | {stoi_imp:+.3f}")
        
        pesq_imp = scores['PESQ_Output'] - scores['PESQ_Input']
        print(f"{'PESQ':<10} | {scores['PESQ_Input']:.3f}      | {scores['PESQ_Output']:.3f}      | {pesq_imp:+.3f}")

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