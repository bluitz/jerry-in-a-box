"""
YIN pitch detection algorithm implementation based on RPiPitch.
This is a simplified version focused on guitar note detection.
"""
import numpy as np
from numpy.fft import rfft
from scipy.signal import fftconvolve

class YINPitchDetector:
    def __init__(self, sample_rate=44100, buffer_size=1024, threshold=0.1):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.yin_buffer = np.zeros(buffer_size // 2)
        self.yin_prob = 0.0
        self.pitch = 0.0
        self.confidence = 0.0

    def get_pitch(self, audio_buffer):
        """Estimate the pitch of the audio signal using the YIN algorithm."""
        # Ensure the buffer is the correct size
        if len(audio_buffer) != self.buffer_size:
            return 0.0, 0.0
        
        # Convert to mono if stereo
        if len(audio_buffer.shape) > 1:
            audio_buffer = np.mean(audio_buffer, axis=1)
        
        # Step 1: Calculate the difference function
        self.yin_buffer = self._difference(audio_buffer)
        
        # Step 2: Calculate the cumulative mean normalized difference
        self.yin_buffer = self._cumulative_mean_normalized_difference(self.yin_buffer)
        
        # Step 3: Absolute threshold
        tau = self._absolute_threshold(self.yin_buffer)
        
        # If no pitch found, return 0
        if tau == -1:
            self.pitch = 0.0
            self.confidence = 0.0
        else:
            # Step 4: Parabolic interpolation for better accuracy
            if tau > 0 and tau < len(self.yin_buffer) - 1:
                s0, s1, s2 = self.yin_buffer[tau-1], self.yin_buffer[tau], self.yin_buffer[tau+1]
                adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0))
                tau += adjustment
            
            # Convert to frequency
            self.pitch = self.sample_rate / tau if tau != 0 else 0.0
            
            # Calculate confidence (inverse of the minimum value of the difference function)
            self.confidence = 1.0 - np.min(self.yin_buffer[1:])
        
        return self.pitch, self.confidence
    
    def _difference(self, audio_buffer):
        """Calculate the difference function."""
        # Autocorrelation using FFT
        fft_size = self.buffer_size
        audio_fft = rfft(audio_buffer, n=fft_size)
        audio_fft_conj = np.conj(audio_fft)
        
        # Multiply FFT by its complex conjugate
        product = audio_fft * audio_fft_conj
        
        # Inverse FFT to get autocorrelation
        autocorr = np.fft.irfft(product)
        
        # Calculate difference function
        diff = np.zeros(self.buffer_size // 2)
        diff[0] = 1.0
        
        for tau in range(1, len(diff)):
            diff[tau] = autocorr[0] - 2 * autocorr[tau] + autocorr[0]
            
        return diff
    
    @staticmethod
    def _cumulative_mean_normalized_difference(diff):
        """Calculate the cumulative mean normalized difference."""
        running_sum = 0.0
        diff[0] = 1.0
        
        for tau in range(1, len(diff)):
            running_sum += diff[tau]
            if running_sum == 0:
                diff[tau] = 1.0
            else:
                diff[tau] *= tau / running_sum
                
        return diff
    
    def _absolute_threshold(self, yin_buffer):
        """Find the first dip below the threshold."""
        tau = 0
        
        # Skip the first few values to avoid local minima
        for tau in range(2, len(yin_buffer) - 1):
            if yin_buffer[tau] < self.threshold:
                # Find the local minimum
                while (tau + 1 < len(yin_buffer) - 1 and 
                       yin_buffer[tau + 1] < yin_buffer[tau]):
                    tau += 1
                return tau
                
        # If no dip found below threshold, find the global minimum
        return np.argmin(yin_buffer[2:]) + 2
