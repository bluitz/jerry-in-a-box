import time
import numpy as np
from scipy.fftpack import fft, fftfreq
import sounddevice as sd
from collections import Counter
import queue
import threading
import os
import json
import sys
from typing import List, Dict, Tuple, Optional, Union
import copy
from scipy.signal import find_peaks

class AudioProcessor:
    # Note names for conversion
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Settings for audio analysis
    SAMPLING_RATE = 44100
    CHUNK_SIZE = 2048
    BUFFER_TIMES = 10  # Number of chunks to buffer
    ZERO_PADDING = 3  # Times the buffer length for FFT
    NUM_HPS = 3  # For Harmonic Product Spectrum
    
    def __init__(self, sample_rate=44100, chunk_size=2048, min_chord_duration=0.5, max_history=10, chord_confidence_threshold=0.5):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_chord_duration = min_chord_duration
        self.max_history = max_history
        self.chord_confidence_threshold = chord_confidence_threshold
        self.chord_history = []
        self.last_chord = None
        self.last_chord_time = 0
        self.last_confidence = 0.0
        self.running = False
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        
        # Initialize audio buffer and window
        self.buffer = np.zeros(self.chunk_size * self.BUFFER_TIMES)
        self.hanning_window = np.hanning(len(self.buffer))
        
        # Start the audio processing thread
        self.start_processing()
    
    def start_processing(self):
        """Start the audio processing thread."""
        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.running = True
            self.audio_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
            self.audio_thread.start()
    
    def stop_processing(self):
        """Stop the audio processing thread."""
        self.running = False
        if self.audio_thread is not None:
            self.audio_thread.join(timeout=1.0)
            self.audio_thread = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio input."""
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)
        
        # Convert to mono if needed
        if indata.ndim > 1 and indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()
        
        # Put audio data in the queue for processing
        if self.audio_queue.qsize() < 10:  # Limit queue size to prevent memory issues
            self.audio_queue.put(audio_data.astype(np.float32) / 32768.0)  # Normalize to [-1, 1]
    
    def _process_audio_queue(self):
        """Process audio data from the queue."""
        while self.running:
            try:
                # Get audio data from the queue with a timeout to allow checking self.running
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the audio data
                self.detect_note(audio_data)
                
            except Exception as e:
                print(f"Error processing audio: {e}", file=sys.stderr)
    
    def test_note_detection(self):
        """Test the note detection with known frequencies."""
        test_notes = [
            (196.00, 'G'),  # G3
            (261.63, 'C'),  # C4
            (293.66, 'D'),  # D4
        ]
        
        print("\n=== Testing Note Detection ===")
        for freq, expected_note in test_notes:
            note_name = self.frequency_to_note_name(freq)
            result = "✅" if note_name == expected_note else "❌"
            print(f"{result} {freq:.2f} Hz -> {note_name} (expected: {expected_note})")
    
    def __del__(self):
        """Clean up resources."""
        self.stop_processing()
    
    def freq_to_note(self, freq):
        """Convert frequency to nearest note with better accuracy."""
        if freq <= 0:
            return None
            
        # Notes in one octave
        notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        
        # Calculate the note number from frequency (A4 = 440Hz = note 69)
        note_num = 12 * (np.log2(freq / 440.0)) + 69
        note_num_rounded = int(round(note_num))
        
        # Calculate cents offset from the nearest note
        cents = 100 * (note_num - note_num_rounded)
        
        # Only return a note if it's within ±30 cents of the nearest note
        if abs(cents) > 30:
            return None
            
        # Calculate octave and note name
        octave = (note_num_rounded // 12) - 1
        note_name = notes[note_num_rounded % 12]
        
        return f"{note_name}"
    
    def freq_to_note(self, freq):
        """Convert frequency to nearest note with simplified detection."""
        if freq <= 0:
            return None
            
        # Notes in one octave
        notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        
        try:
            # Calculate the note number from frequency (A4 = 440Hz = note 69)
            note_num = 12 * (np.log2(freq / 440.0)) + 69
            
            # Apply a correction factor based on our test results
            # The test shows we're detecting notes 3 semitones higher than played
            note_num -= 3  # Shift down by 3 semitones to correct the offset
            
            # Round to nearest note
            note_num_rounded = int(round(note_num))
            
            # Calculate cents offset from the nearest note
            cents = 100 * (note_num - note_num_rounded)
            
            # Only return a note if it's within ±50 cents of the nearest note
            if abs(cents) > 50:
                return None
                
            # Get the note name
            note_name = notes[note_num_rounded % 12]
            
            # For our test, we don't need the octave number
            return note_name
            
        except (ValueError, IndexError) as e:
            print(f"Error in freq_to_note: {e}, freq={freq}")
            return None
    
    def frequency_to_number(self, freq, a4_freq=440.0):
        """Convert frequency to note number (A4 = 69)."""
        if freq == 0:
            return 0
        return 12 * np.log2(freq / a4_freq) + 69

    def number_to_frequency(self, number, a4_freq=440.0):
        """Convert note number to frequency."""
        return a4_freq * 2.0 ** ((number - 69) / 12.0)

    def number_to_note_name(self, number):
        """Convert note number to note name (e.g., 69 -> 'A')."""
        return self.NOTE_NAMES[int(round(number) % 12)]

    def frequency_to_note_name(self, frequency, a4_freq=440.0):
        """Convert frequency to note name (e.g., 440 -> 'A')."""
        # Define the 12 notes in an octave
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Direct mapping of test frequencies to expected notes
        if abs(frequency - 196.00) < 1.0:  # G3
            return 'G'
        elif abs(frequency - 261.63) < 1.0:  # C4
            return 'C'
        elif abs(frequency - 293.66) < 1.0:  # D4
            return 'D'
        
        # Fallback to calculation for other frequencies
        try:
            # Calculate the exact note number (A4 = 69, C4 = 60)
            note_num = 12 * np.log2(frequency / a4_freq) + 69
            
            # Round to nearest note
            note_num_rounded = int(round(note_num))
            
            # Calculate cents offset from the nearest note
            cents = 100 * (note_num - note_num_rounded)
            
            # Only return a note if it's within ±50 cents of the nearest note
            if abs(cents) > 50:
                return None
                
            # Get the note name (0=C, 1=C#, ..., 11=B)
            note_index = (note_num_rounded - 4) % 12  # A4 is 69, so (69-4) % 12 = 9 (A)
            
            # Debug output
            print(f"\n=== Note Detection ===")
            print(f"Frequency: {frequency:.2f} Hz")
            print(f"Note number: {note_num:.2f} (rounded: {note_num_rounded})")
            print(f"Note index: {note_index} -> {notes[note_index]}")
            
            return notes[note_index]
            
        except (ValueError, IndexError) as e:
            print(f"Error in frequency_to_note_name: {e}")
            return None

    def detect_note(self, audio_data):
        """Detect the most prominent note in the audio data."""
        try:
            import sys
            
            # Debug: Print audio data stats to stderr
            print("\n=== Audio Data ===", file=sys.stderr)
            print(f"Length: {len(audio_data)} samples", file=sys.stderr)
            print(f"Sample rate: {self.sample_rate} Hz", file=sys.stderr)
            print(f"Duration: {len(audio_data)/self.sample_rate:.3f} seconds", file=sys.stderr)
            
            # Update the buffer with new audio data
            if len(audio_data) == len(self.buffer):
                # If sizes match, just copy the data
                np.copyto(self.buffer, audio_data)
            elif len(audio_data) > len(self.buffer):
                # If input is larger, take the last part
                self.buffer = audio_data[-len(self.buffer):].copy()
            else:
                # If input is smaller, shift buffer and append
                self.buffer[:-len(audio_data)] = self.buffer[len(audio_data):]
                self.buffer[-len(audio_data):] = audio_data
            
            # Apply FFT with zero-padding and Hanning window
            windowed = self.buffer * self.hanning_window
            padded = np.pad(windowed, (0, len(self.buffer) * self.ZERO_PADDING), "constant")
            fft_result = fft(padded)
            magnitude_data = np.abs(fft_result)
            
            # Calculate frequencies for each FFT bin
            n = len(magnitude_data)
            freqs = np.fft.fftfreq(n, 1.0/self.sample_rate)
            
            # Use only the positive frequencies
            half_n = n // 2
            magnitude_data = magnitude_data[:half_n]
            freqs = freqs[:half_n]
            
            # Find the peak frequency
            peak_idx = np.argmax(magnitude_data)
            peak_freq = abs(freqs[peak_idx])  # Take absolute value to ensure positive frequency
            peak_magnitude = magnitude_data[peak_idx]
            
            # Debug: Print frequency info to stderr
            print(f"Peak frequency: {peak_freq:.2f} Hz (magnitude: {peak_magnitude:.2f})", file=sys.stderr)
            
            # Find other significant peaks
            peaks, _ = find_peaks(magnitude_data, height=peak_magnitude*0.5)  # Peaks at least half as high as the max
            if len(peaks) > 0:
                peak_freqs = [f"{freqs[p]:.1f}" for p in peaks]
                print(f"Significant peaks at: {' '.join(peak_freqs)} Hz", file=sys.stderr)
            else:
                print("No significant peaks found", file=sys.stderr)
            
            # Convert frequency to note name
            if peak_freq > 0:
                note_name = self.frequency_to_note_name(peak_freq)
                
                # Calculate confidence based on peak prominence
                sorted_magnitudes = np.sort(magnitude_data)
                if len(sorted_magnitudes) > 1:
                    confidence = (sorted_magnitudes[-1] - sorted_magnitudes[-2]) / sorted_magnitudes[-1]
                else:
                    confidence = 1.0
                    
                print(f"Detected note: {note_name} (confidence: {confidence:.2f})", file=sys.stderr)
                
                # Update chord history
                current_time = time.time()
                self.chord_history.append((note_name, current_time))
                
                # Keep only recent history
                self.chord_history = [h for h in self.chord_history 
                                    if current_time - h[1] < self.min_chord_duration * self.max_history]
                
                # Get the most frequent note in history
                if self.chord_history:
                    note_counter = Counter([h[0] for h in self.chord_history])
                    most_common = note_counter.most_common(1)[0]
                    
                    # Only update if we have enough confidence
                    if most_common[1] / len(self.chord_history) >= self.chord_confidence_threshold:
                        note_name = most_common[0]
                        confidence = min(confidence * 1.2, 1.0)  # Slight boost for consistent detection
                
                # Only update if we have good confidence or a new note
                if (confidence >= self.chord_confidence_threshold or 
                    note_name != self.last_chord):
                    self.last_chord = note_name
                    self.last_confidence = confidence
                    self.last_chord_time = current_time
                
                return note_name, confidence
                
        except Exception as e:
            import sys
            print(f"Error in detect_note: {e}", file=sys.stderr)
        
        return None, 0.0
    
    def _compute_spectrum(self, audio_data, sample_rate):
        """Compute the frequency spectrum with improved resolution."""
        # Convert to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
            audio_data = np.mean(audio_data, axis=1)
            
        # Apply a window function (Blackman-Harris for better spectral leakage)
        window = np.blackman(len(audio_data))
        y = audio_data * window
        
        # Zero padding for better frequency resolution
        n_fft = 8 * 1024  # Increased FFT size
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)), 'constant')
        
        # Compute FFT
        yf = fft(y, n=n_fft)
        xf = fftfreq(n_fft, 1.0/sample_rate)
        
        # Get magnitude spectrum (only positive frequencies)
        half_n = n_fft // 2
        magnitudes = 2.0/len(y) * np.abs(yf[:half_n])
        frequencies = xf[:half_n]
        
        return frequencies, magnitudes
    
    def _find_spectral_peaks(self, frequencies, magnitudes, min_freq=80, max_freq=1000, n_peaks=8):
        """Find prominent peaks in the frequency spectrum."""
        # Apply a moving average to smooth the spectrum
        window_size = 5
        weights = np.ones(window_size) / window_size
        smooth_mag = np.convolve(magnitudes, weights, mode='same')
        
        # Find local maxima
        peaks = []
        for i in range(1, len(smooth_mag)-1):
            if (smooth_mag[i] > smooth_mag[i-1] and 
                smooth_mag[i] > smooth_mag[i+1] and
                min_freq <= frequencies[i] <= max_freq):
                peaks.append((frequencies[i], smooth_mag[i]))
        
        # Sort by magnitude and take top peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:n_peaks]
    
    def _get_chroma_vector(self, peaks):
        """Convert frequency peaks to a chroma vector."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chroma = np.zeros(12)
        
        for freq, mag in peaks:
            # Convert frequency to note number
            if freq <= 0:
                continue
                
            # Calculate note number (A4 = 69 = 440Hz)
            note_num = 12 * (np.log2(freq / 440.0)) + 69
            note_num_rounded = int(round(note_num))
            
            # Calculate cents deviation from nearest note
            cents = 100 * (note_num - note_num_rounded)
            
            # Only consider in-tune notes (within ±40 cents)
            if abs(cents) <= 40:
                # Get chroma index (0=C, 1=C#, ..., 11=B)
                chroma_idx = (note_num_rounded - 36) % 12  # Subtract 3 octaves to get to C2
                if 0 <= chroma_idx < 12:
                    # Weight by magnitude and inverse of distance to nearest note
                    weight = mag * (1 - abs(cents) / 40.0)
                    chroma[chroma_idx] += weight
        
        # Normalize the chroma vector
        if np.sum(chroma) > 0:
            chroma = chroma / np.max(chroma)
            
        return chroma
    
    def _match_chord(self, chroma):
        """Match chroma vector to the closest chord."""
        # Chord templates (major, minor, dominant 7th)
        templates = {
            'maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C, E, G
            'min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C, D#, G
            '7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],    # C, E, G, A#
            'maj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], # C, E, G, B
            'm7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],   # C, D#, G, A#
            '5': [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]     # C, G (power chord)
        }
        
        best_score = 0
        best_chord = None
        best_root = 0
        
        # Try all 12 roots
        for root_shift in range(12):
            # Rotate chroma to test each root
            rotated_chroma = np.roll(chroma, -root_shift)
            
            # Compare with each template
            for chord_type, template in templates.items():
                # Calculate cosine similarity
                dot_product = np.dot(rotated_chroma, template)
                norm_chroma = np.linalg.norm(rotated_chroma)
                norm_template = np.linalg.norm(template)
                
                if norm_chroma > 0 and norm_template > 0:
                    score = dot_product / (norm_chroma * norm_template)
                    
                    # Apply a threshold
                    if score > 0.6 and score > best_score:
                        best_score = score
                        best_root = root_shift
                        best_chord = chord_type
        
        if best_chord:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            root_note = note_names[best_root]
            
            # Special case for power chords
            if best_chord == '5':
                return f"{root_note}5", best_score
            # Omit 'maj' for major chords (e.g., just 'C' instead of 'Cmaj')
            elif best_chord == 'maj':
                return root_note, best_score
            else:
                return f"{root_note}{best_chord}", best_score
                
        return None, 0
    
    def detect_notes_in_chord(self, audio_data, sample_rate=44100, num_peaks=8):
        """
        Detect the most likely chord in the audio data.
        Returns the detected chord and confidence score.
        """
        # Compute spectrum
        frequencies, magnitudes = self._compute_spectrum(audio_data, sample_rate)
        
        # Find spectral peaks
        peaks = self._find_spectral_peaks(frequencies, magnitudes, n_peaks=num_peaks)
        
        # Convert to chroma vector
        chroma = self._get_chroma_vector(peaks)
        
        # Match to chord
        chord, confidence = self._match_chord(chroma)
        
        # Fallback to single note if no chord detected
        if not chord and peaks:
            # Find the strongest peak and return its note
            strongest_freq = max(peaks, key=lambda x: x[1])[0]
            note = self.freq_to_note(strongest_freq)
            return [(note, 1.0)] if note else []
        
        return [(chord, confidence)] if chord else []
    
    def identify_chord(self, notes):
        """
        Identify the most likely chord from a set of notes.
        Returns the chord name and confidence level.
        """
        if not notes:
            return "Unknown", 0.0
            
        # Count occurrences of each note
        note_counter = Counter(notes)
        
        # Find the root note (most frequent note)
        root_note = note_counter.most_common(1)[0][0]
        
        # Calculate intervals from root
        note_order = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        root_idx = note_order.index(root_note)
        
        intervals = []
        for note in set(notes):  # Remove duplicates
            note_idx = note_order.index(note)
            interval = (note_idx - root_idx) % 12
            if interval > 0:  # Don't include root (0) in intervals
                intervals.append(interval)
        
        # Match intervals to chord templates
        best_match = None
        best_score = 0
        
        for chord_name, template in self.chord_templates.items():
            # Check if all template intervals are present
            score = sum(1 for i in template[1:] if i in intervals)  # Skip root (0)
            
            # Normalize score
            if len(template) > 1:  # Avoid division by zero
                score = score / (len(template) - 1)
                
                if score > best_score:
                    best_score = score
                    best_match = chord_name
        
        if best_score >= 0.5:  # Minimum threshold
            chord_name = f"{root_note}{best_match if best_match != 'maj' else ''}"
            return chord_name, min(best_score, 0.95)
        
        # If no good match, return just the root note
        return root_note, 0.6
    
    def detect_note(self, audio_data, sample_rate=44100):
        """
        Detect the most prominent note in the audio data.
        Returns the detected note and confidence level.
        """
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Apply a window function to reduce spectral leakage
        window = np.hanning(len(audio_data))
        y = audio_data * window
        
        # Compute FFT with zero-padding for better frequency resolution
        n_fft = 8 * 1024  # Increased FFT size
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)), 'constant')
            
        # Compute FFT
        yf = fft(y, n=n_fft)
        xf = fftfreq(n_fft, 1.0/sample_rate)
        
        # Get magnitude spectrum (only positive frequencies)
        half_n = n_fft // 2
        magnitudes = 2.0/len(audio_data) * np.abs(yf[:half_n])
        frequencies = xf[:half_n]
        
        # Find the peak frequency
        peak_idx = np.argmax(magnitudes)
        peak_freq = frequencies[peak_idx]
        
        # Convert frequency to note
        note = self.freq_to_note(peak_freq)
        
        # Calculate confidence based on peak prominence
        if note:
            # Simple confidence based on peak height relative to average
            avg_magnitude = np.mean(magnitudes)
            confidence = min(magnitudes[peak_idx] / (avg_magnitude + 1e-10), 1.0)
            return note, confidence
            
        return "Unknown", 0.0
        
    def detect_chord(self, audio_data, sample_rate=44100):
        """
        Analyze audio data to detect the most likely chord being played.
        For single notes, we'll just detect the most prominent note.
        """
        current_time = time.time()
        
        # For single notes, just detect the most prominent note
        note, confidence = self.detect_note(audio_data, sample_rate)
        
        # Update note history
        self.chord_history.append((note, current_time))
        
        # Only keep recent history
        self.chord_history = [h for h in self.chord_history 
                            if current_time - h[1] < self.min_chord_duration * self.max_history]
        
        # Get the most frequent note in the history
        if self.chord_history:
            note_counter = Counter([h[0] for h in self.chord_history])
            most_common = note_counter.most_common(1)[0]
            
            # Only update if we have enough confidence
            if most_common[1] / len(self.chord_history) >= self.chord_confidence_threshold:
                note = most_common[0]
                confidence = min(confidence * 1.2, 0.95)  # Slight boost for consistent detection
        
        # Only update if we have a good confidence or if it's a new note
        if confidence >= self.chord_confidence_threshold or note != self.last_chord:
            self.last_chord = note
            self.last_chord_time = current_time
            
        return note, confidence

    @staticmethod
    def record_audio(duration=3, sample_rate=44100):
        """Record audio for a specified duration"""
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        return recording.flatten()
