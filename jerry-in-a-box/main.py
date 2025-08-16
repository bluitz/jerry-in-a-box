import argparse
import time
import logging
import threading
import queue
import sys
import select
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import pyttsx3
import speech_recognition as sr
from typing import List, Optional
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from jerry_in_a_box.audio_processor import AudioProcessor
from jerry_in_a_box.song_database import SongDatabase

# Initialize text-to-speech engine
try:
    tts_engine = pyttsx3.init()
    # Set properties (optional)
    tts_engine.setProperty('rate', 150)  # Speed of speech
    tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
except Exception as e:
    print(f"Warning: Could not initialize text-to-speech: {e}")
    tts_engine = None

# Check if OpenAI API key is set
if not os.getenv('OPENAI_API_KEY'):
    print("Warning: OPENAI_API_KEY environment variable not set. ChatGPT functionality will be disabled.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jerry_in_a_box.log')
    ]
)
logger = logging.getLogger(__name__)

class JerryInABox:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.song_db = SongDatabase()
        self.current_progression = []
        self.last_chord_time = 0
        self.chord_timeout = 5  # seconds of silence before resetting progression
        self.running = False
        self.keyboard_queue = queue.Queue()
        self.keyboard_thread = None
        self.keyboard_mapping = {
            'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G',
            '1': 'A#', '2': 'C#', '3': 'D#', '4': 'F#', '5': 'G#',
            '?': '?'  # Add question mark for asking questions
        }
        self.conversation_history = []  # Store conversation history for context
        self.last_chords = []  # Store last 5 chords
        self.max_chord_history = 5  # Number of chords to keep in history
        self.last_processed_progression = None  # Store the last processed progression

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self._adjust_for_ambient_noise()

    def _adjust_for_ambient_noise(self):
        """Adjust the recognizer for ambient noise"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio processing"""
        if status:
            logger.warning(f"Audio status: {status}")

        # Process the audio data to detect chords
        current_time = time.time()

        # Only process audio if we have enough data
        if len(indata) > 0:
            try:
                # Detect the chord from the audio
                chord, confidence = self.audio_processor.detect_chord(indata[:, 0])

                # Only consider chords with sufficient confidence
                if confidence > 0.7:  # Adjust threshold as needed
                    self._add_chord_to_progression(chord)
                    self._process_progression()
                    self._display_current_state()

            except Exception as e:
                logger.error(f"Error processing audio: {e}")

    def _display_current_state(self):
        """Display the current state of chord detection and matches"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

        print("Jerry in a Box - Chord Recognition")
        print("=" * 80)

        # Show current chord and last 5 chords
        print(f"Current chord: {self.current_progression[-1] if self.current_progression else 'None'}")
        print(f"Last {self.max_chord_history} chords: {' -> '.join(self.last_chords) if self.last_chords else 'None'}")
        print(f"Full progression: {' -> '.join(self.current_progression) if self.current_progression else 'None'}")
        print("\nListening for chords... (Press Ctrl+C to quit)\n")

        # Show recent matches if we have any
        if hasattr(self, 'last_matches') and self.last_matches:
            print("\nPossible Matches:")
            print("-" * 80)

            for i, (song, score, next_chords) in enumerate(self.last_matches[:5], 1):
                percentage = min(int(score * 100), 100)
                print(f"{i}. {song.title} - {song.artist} ({percentage}% match)")
                print(f"   Next likely chords: {' -> '.join(next_chords) if next_chords else 'End of progression'}")
                print(f"   Source: {song.source}")
                print("-" * 80)

    def _process_progression(self):
        """Process the current chord progression and find matching songs"""
        if not self.last_chords or len(self.last_chords) < 2:
            return

        # Make a copy to avoid modifying the original
        progression_to_match = self.last_chords.copy()

        try:
            # Find similar progressions in the database
            matches = self.song_db.find_similar_progressions(progression_to_match)

            if matches:
                print("\n🎵 Possible Song Matches:")
                print("-" * 60)
                for i, (song, score, next_chords) in enumerate(matches[:3], 1):
                    percentage = min(int(score * 100), 100)
                    # Create a visual representation of the match confidence
                    bar_length = 20
                    filled_length = int(bar_length * score)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)

                    print(f"\n🎸 {song.title} - {song.artist}")
                    print(f"   🎯 Match: {bar} {percentage}%")
                    if next_chords:
                        print(f"   ⏭️  Next: {' → '.join(next_chords[:3])}...")
                    else:
                        print("   ⏹️  End of progression")
                    print(f"   📚 Source: {song.source}")
            else:
                print("\n🔍 No matching songs found.")
                print("   Try adding more chords or check your input.")

        except Exception as e:
            print(f"\n❌ Error finding matches: {e}")

        print("\n" + "-" * 60)
        print("🎹 Add more chords: A-G (basic) | 1-5 (sharps) | C (clear) | Q (quit)")
        print("-" * 60)

    def _clear_screen(self):
        """Clear the terminal screen"""
       # print("\033[H\033[J", end="")  # ANSI escape codes to clear screen

    def _print_header(self):
        """Print the application header"""
        print("=" * 60)
        print("Jerry in a Box - Chord Recognition System".center(60))
        print("=" * 60)
        print("\nKeyboard input enabled. Press keys to input chords:")
        print("  A-G: Basic chords")
        print("  1-5: Sharps (1=A#, 2=C#, 3=D#, 4=F#, 5=G#)")
        print("    ?: Ask a question (speak when prompted)")
        print("    C: Clear progression")
        print("    Q: Quit\n")
        print("-" * 60)

    def _keyboard_listener(self):
        """Listen for keyboard input in a separate thread"""
        self._clear_screen()
        self._print_header()

        while self.running:
            # Check if there's input ready (non-blocking)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()
                if key in self.keyboard_mapping or key in ['q', 'c']:
                    self.keyboard_queue.put(key)
                else:
                    print(f"\nUnknown key: {key}. Use A-G, 1-5, C, or Q")

    def _ask_chatgpt(self, question):
        """Ask a question to ChatGPT and get a response"""
        if not os.getenv('OPENAI_API_KEY'):
            print("\n❌ Error: OPENAI_API_KEY environment variable not set.")
            return

        print("\n🤔 Thinking...")

        try:
            # Add user question to conversation history
            self.conversation_history.append({"role": "user", "content": question})

            # Call OpenAI API
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful music theory assistant. Keep your answers concise and focused on music theory, especially related to guitar, chords, and music composition."}
            ] + self.conversation_history[-6:],  # Keep last 3 exchanges for context
            max_tokens=300,
            temperature=0.7)

            # Get the response text
            answer = response.choices[0].message.content.strip()

            # Add assistant's response to conversation history
            self.conversation_history.append({"role": "assistant", "content": answer})

            # Print the response
            print(f"\n🎵 {answer}")

            # Speak the response
            if tts_engine is not None:
                try:
                    # Split long text into sentences to avoid TTS issues
                    import re
                    sentences = re.split(r'(?<=[.!?]) +', answer)
                    for sentence in sentences:
                        tts_engine.say(sentence)
                        tts_engine.runAndWait()
                except Exception as e:
                    print(f"\n⚠️ Could not speak the response: {e}")

            return answer

        except Exception as e:
            error_msg = f"\n❌ Error communicating with ChatGPT: {str(e)}"
            print(error_msg)
            return error_msg

    def _process_keyboard_input(self):
        """Process keyboard input from the queue"""
        try:
            while not self.keyboard_queue.empty():
                key = self.keyboard_queue.get_nowait()

                if key == 'q':  # Quit
                    self.running = False
                    return
                elif key == 'c':  # Clear progression
                    self.current_progression = []
                    self.last_chords = []
                    self._clear_screen()
                    self._print_header()
                    print("\n✅ Progression cleared\n")
                    print("Start a new progression by entering chords (A-G, 1-5 for sharps)")
                elif key == '?':  # Ask a question
                    self._clear_screen()
                    self._print_header()
                    print("\n🎤 Please ask your question (speak now)...")

                    try:
                        with self.microphone as source:
                            print("Listening... (speak now)")
                            try:
                                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                                question = self.recognizer.recognize_google(audio)
                                print(f"\n❓ Your question: {question}")
                                self._ask_chatgpt(question)
                            except sr.WaitTimeoutError:
                                print("\n⏱️  No speech detected. Please try again.")
                            except sr.UnknownValueError:
                                print("\n❌ Could not understand audio. Please try again.")
                            except sr.RequestError as e:
                                print(f"\n❌ Could not request results; {e}")
                            except Exception as e:
                                print(f"\n❌ Unexpected error: {e}")
                    except Exception as e:
                        print(f"\n❌ Could not access microphone: {e}")

                    print("\nPress any key to continue...")

                elif key in self.keyboard_mapping:
                    chord = self.keyboard_mapping[key]
                    self._add_chord_to_progression(chord)
                    # The _add_chord_to_progression will handle the display
        except queue.Empty:
            pass

    def _add_chord_to_progression(self, chord):
        """Add a chord to the current progression and update display"""
        current_time = time.time()

        # If it's been a while since the last chord, reset the progression
        if current_time - self.last_chord_time > self.chord_timeout:
            self.current_progression = []
            self.last_chords = []

        # Add the chord if it's different from the last one
        if not self.current_progression or self.current_progression[-1] != chord:
            self.current_progression.append(chord)
            self.last_chords.append(chord)

            # Keep only the last N chords
            if len(self.last_chords) > self.max_chord_history:
                self.last_chords = self.last_chords[-self.max_chord_history:]

            self.last_chord_time = current_time

            # Don't clear the screen to help with debugging
            print("\n" + "="*60)
            print("🎸 Current Progression:")
            print(f"   {' → '.join(self.last_chords)}\n")

            # Process the progression if we have enough chords
            if len(self.last_chords) >= 2:
                self._process_progression()
            else:
                print("Add more chords to find matching songs...")
                print("\n" + "-" * 60)

    def start(self):
        """Start the application"""
        self.running = True

        # Start keyboard input in a separate thread
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        # Start audio processing
        try:
            print("Starting Jerry in a Box...")
            print("Press 'q' to quit, 'c' to clear progression, '?' to ask a question")
            self.audio_processor.start_stream(self.audio_callback)

            # Main loop
            while self.running:
                self._process_keyboard_input()
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping Jerry in a Box...")
            self.running = False

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.running = False

        finally:
            self.stop()

    def stop(self):
        """Stop the application"""
        self.running = False
        self.audio_processor.stop_stream()
        print("\nThanks for using Jerry in a Box!")

def main():
    parser = argparse.ArgumentParser(description='Jerry in a Box - Guitar Chord Recognition System')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        app = JerryInABox()
        app.start()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
