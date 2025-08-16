import speech_recognition as sr
import webbrowser
import time
from dataclasses import dataclass
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

@dataclass
class VoiceCommand:
    """Represents a voice command with its handler function"""
    phrase: str
    handler: Callable[[], None]

class VoiceCommandProcessor:
    def __init__(self, song_db):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.song_db = song_db
        self.commands = [
            VoiceCommand(
                phrase="find a song",
                handler=self._handle_find_song
            ),
            # Add more commands here as needed
        ]
        self.is_listening = False
        self._adjust_for_ambient_noise()

    def _adjust_for_ambient_noise(self):
        """Adjust the recognizer for ambient noise"""
        with self.microphone as source:
            logger.info("Adjusting for ambient noise. Please wait...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            logger.info("Ambient noise adjustment complete.")

    def _is_command_match(self, text, command_phrase):
        """Check if the recognized text matches a command phrase"""
        # Split both the recognized text and command phrase into words
        text_words = set(text.lower().split())
        command_words = set(command_phrase.lower().split())
        
        # Check if all words in the command phrase are in the recognized text
        return command_words.issubset(text_words)
    
    def start_listening(self):
        """Start listening for voice commands"""
        self.is_listening = True
        print("\nVoice command processor started. Say 'find a song' to search for a song.")
        
        while self.is_listening:
            try:
                with self.microphone as source:
                    print("\nListening for command...")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=4)
                
                try:
                    # Recognize speech using Google's speech recognition
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"\nRecognized: {text}")
                    
                    # Check for matching commands
                    command_matched = False
                    for command in self.commands:
                        if self._is_command_match(text, command.phrase):
                            print(f"Executing command: {command.phrase}")
                            command.handler()
                            command_matched = True
                            break
                    
                    if not command_matched:
                        print("Command not recognized. Try saying 'find a song'.")
                    
                except sr.UnknownValueError:
                    print("Could not understand audio. Please try again.")
                except sr.RequestError as e:
                    print(f"Error with the speech recognition service; {e}")
                
            except KeyboardInterrupt:
                logger.info("Stopping voice command processor...")
                self.is_listening = False
            except Exception as e:
                logger.error(f"Error in voice command processing: {e}")
                time.sleep(1)  # Prevent tight loop on errors

    def stop_listening(self):
        """Stop listening for voice commands"""
        self.is_listening = False

    def _handle_find_song(self):
        """Handle the 'find a song' command"""
        print("\nWhat song would you like to find? (e.g., 'Row Jimmy')")
        
        try:
            with self.microphone as source:
                print("Listening for song name (speak now)...")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=8)
                
                try:
                    song_name = self.recognizer.recognize_google(audio)
                    print(f"\nSearching for: {song_name}")
                    
                    # Search for the song in our database
                    matching_songs = self.song_db.search_songs(song_name)
                    
                    if matching_songs:
                        print("\nFound these matching songs:")
                        for i, song in enumerate(matching_songs, 1):
                            print(f"{i}. {song.title} - {song.artist}")
                            print(f"   Progression: {' -> '.join(song.progression[:8])}...\n")
                    else:
                        print(f"\nNo matching songs found for '{song_name}' in the database.")
                        print("Would you like to search online? (say 'yes' or 'no')")
                        
                        # Get user's response
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                        response = self.recognizer.recognize_google(audio).lower()
                        
                        if 'yes' in response:
                            self._search_online(song_name)
                        
                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand the song name.")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
        
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def _search_online(self, song_name: str):
        """Search for song chords online"""
        print(f"\nSearching online for '{song_name}' chords...")
        
        # Try popular chord websites
        search_queries = [
            f"{song_name} chords ultimate guitar",
            f"{song_name} chords e-chords",
            f"{song_name} chords chordify"
        ]
        
        for query in search_queries:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            print(f"Trying: {url}")
            webbrowser.open(url)
            
            # Ask if the user found what they were looking for
            print("\nDid you find the song? (say 'yes' or 'no')")
            
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    response = self.recognizer.recognize_google(audio).lower()
                    
                    if 'yes' in response:
                        print("Great! Would you like to add this song to the database? (say 'yes' or 'no')")
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                        response = self.recognizer.recognize_google(audio).lower()
                        
                        if 'yes' in response:
                            self._add_song_to_database(song_name)
                        return
                    
            except (sr.UnknownValueError, sr.RequestError):
                print("Let me try another source...")
                continue
    
    def _add_song_to_database(self, song_name: str):
        """Guide the user through adding a song to the database"""
        print("\nLet's add this song to the database.")
        
        try:
            # Get artist name
            print("Who is the artist?")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                artist = self.recognizer.recognize_google(audio)
            
            # Get chord progression
            print("\nPlease play the chord progression or say it (e.g., 'G D Em C'):")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=15)
                progression_text = self.recognizer.recognize_google(audio)
            
            # Parse the progression
            progression = self.song_db.parse_progression_from_text(progression_text)
            
            # Add to database
            self.song_db.add_song(
                title=song_name,
                artist=artist,
                progression=progression
            )
            
            print(f"\nSuccessfully added '{song_name}' to the database!")
            print(f"Progression: {' -> '.join(progression)}")
            
        except Exception as e:
            print(f"Error adding song to database: {e}")
            print("You can add the song manually later by editing the songs.json file.")
