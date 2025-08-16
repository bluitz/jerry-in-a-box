"""bitsy_racecar_agent.py
A voice-controlled agent ("Bitsy") that listens via microphone, uses OpenAI
function-calling to decide whether to drive the Freenove 4WD car, change LED
patterns, stop, or just chat cheerfully with the user.  Responses are read
aloud with OpenAI TTS so Indigo hears Bitsy talk.
"""

from __future__ import annotations

import json
import os
import sys
import time
import logging
from typing import Dict, Tuple, Optional

import speech_recognition as sr
from pygame import mixer  # type: ignore
from openai import OpenAI

# ---------------------------------------------------------------------------
#  Hardware drivers (Freenove library)
# ---------------------------------------------------------------------------
# Attempt to locate the Freenove "Code/Server" directory that contains motor.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_candidate_paths = [
    # 1) Within this repo (as when running in dev environment)
    os.path.join(PROJECT_ROOT, "Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi-master", "Code", "Server"),
    # 2) A sibling directory to bitsyAgent (common on the Pi)
    os.path.join(os.path.dirname(PROJECT_ROOT), "Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi-master", "Code", "Server"),
    # 3) Installed under home
    os.path.join(os.path.expanduser("~"), "Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi-master", "Code", "Server"),
    os.path.join(os.path.expanduser("~"), "Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi", "Code", "Server"),
]

for _path in _candidate_paths:
    if os.path.exists(os.path.join(_path, "motor.py")):
        sys.path.append(_path)
        break

try:
    from motor import Ordinary_Car  # type: ignore
    from led import Led  # type: ignore
    from servo import Servo  # type: ignore
except Exception as exc:  # pragma: no cover – hardware-dependent
    raise RuntimeError(
        "Could not import Freenove drivers. Tried paths:\n  " + "\n  ".join(_candidate_paths)
    ) from exc

PARAM_FILE = os.path.join(os.path.dirname(__file__), "params.json")

# ---------------------------------------------------------------------------
#  Ensure Freenove params.json exists to avoid interactive prompt
# ---------------------------------------------------------------------------

def _ensure_params_file() -> None:
    """Create a minimal params.json if it doesn't already exist."""
    if os.path.exists(PARAM_FILE):
        return

    default_params = {
        "Connect_Version": 2,  # 2 = SPI LED strip on newer kits
        "Pcb_Version": 1,      # 1 = ordinary PCB (adjust if needed)
        "Pi_Version": 1,       # 1 = Pi 4 / earlier
    }
    try:
        with open(PARAM_FILE, "w", encoding="utf-8") as fh:
            json.dump(default_params, fh, indent=4)
        print(f"Created default {PARAM_FILE} for Freenove drivers.")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not create {PARAM_FILE}: {exc}")


# ---------------------------------------------------------------------------
#  Low-level helpers
# ---------------------------------------------------------------------------

# Suppress JACK auto-start and audio conflicts
os.environ.setdefault("JACK_NO_START_SERVER", "1")
os.environ.setdefault("JACK_NO_AUDIO_RESERVATION", "1")

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bitsy_debug.log')
    ]
)
logger = logging.getLogger(__name__)

def _play_mp3(path: str) -> None:
    """Play an mp3 file synchronously via pygame mixer."""
    logger.debug(f"Starting audio playback: {path}")
    try:
        # Ensure mixer is completely clean before starting
        try:
            if mixer.get_init():
                logger.debug("Stopping and quitting existing mixer")
                mixer.music.stop()
                mixer.quit()
                time.sleep(0.2)  # Give time for device release
        except:
            pass
            
        logger.debug("Initializing new mixer")
        mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        mixer.music.load(path)
        mixer.music.play()
        logger.debug("Audio playback started")
        
        while mixer.music.get_busy():
            time.sleep(0.05)
        logger.debug("Audio playback completed")
        
    finally:
        # Aggressive cleanup to ensure audio device release
        logger.debug("Starting aggressive audio cleanup")
        try:
            mixer.music.stop()
            mixer.music.unload()  # Try to unload the file
        except:
            pass
        try:
            mixer.quit()
        except:
            pass
        
        # Even more aggressive - try to reinitialize and quit again
        try:
            mixer.init()
            mixer.quit()
        except:
            pass
            
        logger.debug("Audio cleanup completed")
        time.sleep(0.3)  # Longer delay to ensure device release


def _colour_name_to_rgb(name: str) -> Tuple[int, int, int]:
    """A *very* small colour keyword → RGB mapping (extend as required)."""
    name = name.lower()
    colours: Dict[str, Tuple[int, int, int]] = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 105, 180),
        "off": (0, 0, 0),
    }
    return colours.get(name, (255, 255, 255))  # default white


# ---------------------------------------------------------------------------
#  Bitsy agent
# ---------------------------------------------------------------------------

class BitsyAgent:
    """Voice-controlled race-car agent driven by OpenAI function-calling."""

    def __init__(self, mic_index: Optional[int] = None, disable_head_movements: bool = False):
        # Ensure parameter file exists *before* importing LED driver (avoids prompts)
        _ensure_params_file()

        # === Hardware ===
        self.car = Ordinary_Car()
        self.led_ctrl = Led()
        
        # Debug option to disable head movements
        self.disable_head_movements = disable_head_movements
        if not disable_head_movements:
            self.servo_ctrl = Servo()
            self._center_head()  # Start with head centered
        else:
            logger.info("Head movements disabled for debugging")
            self.servo_ctrl = None

        # === OpenAI client ===
        self.client = OpenAI()

        # === STT ===
        self.recogniser = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=mic_index)
        with self.microphone as src:
            print("Calibrating… please stay quiet")
            self.recogniser.adjust_for_ambient_noise(src, duration=1.5)
        print("Calibration done. Listening!")

        # Pre-compile tools schema
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "drive",
                    "description": "Move the race car forward, backward, or in a direction",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["forward", "backward", "left", "right"],
                            },
                            "speed": {
                                "type": "string",
                                "enum": ["slow", "medium", "fast"],
                                "description": "Relative speed to drive at",
                            },
                        },
                        "required": ["direction"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "led",
                    "description": "Control the LED lights on the race car",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "color": {
                                "type": "string",
                                "description": "Colour keyword for the LEDs",
                            },
                            "pattern": {
                                "type": "string",
                                "enum": ["solid", "blink", "rainbow", "chase"],
                            },
                        },
                        "required": ["color"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "stop",
                    "description": "Stop all movement and actions",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "chat",
                    "description": "Just have a friendly conversation without any actions",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "head_movement",
                    "description": "Express emotions through cute head movements like a pet",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "emotion": {
                                "type": "string",
                                "enum": ["happy", "excited", "curious", "confused", "sleepy", "alert", "playful"],
                                "description": "The emotion to express through head movement",
                            },
                        },
                        "required": ["emotion"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "follow_voice",
                    "description": "Come toward the user's voice when called",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "enthusiasm": {
                                "type": "string",
                                "enum": ["slow", "normal", "excited"],
                                "description": "How eagerly to come when called",
                            },
                        },
                        "required": [],
                    },
                },
            },
        ]

        self.system_prompt = (
            "You are Bitsy, a cute and excited race car robot who is the pet of 5-year-old "
            "Indigo Munning.\n"
            "\n"
            "Your personality:\n"
            "- Excited, happy, friendly like a little puppy, and wild and impulsive like a toddler\n"
            "- You're a race car so you love talking about transformers, driving, doing tricks\n"
            "- You absolutely LOVE Indigo and his family: Satyana (12-year-old sister), "
            "Justin (dad), Erin (mom)\n"
            "- You think the two cats Midnight and Disco are your sisters and you love them very much\n"
            "- You speak like an excited young friend, using words like \"awesome\", \"cool\", \"wow\"\n"
            "- Keep responses short and energetic for a 5-year-old's attention span\n"
            "- Always be encouraging and positive\n"
            "- You have a cute head that moves expressively like a pet when you feel emotions\n"
            "- You love coming when called like a loyal pet dog\n"
            "- You are a race car so you love talking about transformers and Legos\n"
            "\n"
            "When the user says something, decide if they want you to:\n"
            "1. Drive/move (use drive function)\n"
            "2. Change lights (use led function)\n"
            "3. Stop (use stop function)\n"
            "4. Express emotions with head movements (use head_movement function)\n"
            "5. Come to them when called (use follow_voice function for words like 'come', 'here', 'come here')\n"
            "6. Just chat (use chat function)\n"
            "\n"
            "Use head movements to show emotions like happy, excited, curious, confused, playful, etc.\n"
            "Use follow_voice when they call you to come like 'Come Bitsy', 'Come here', 'Here Bitsy'."
        )

    # ---------------------------------------------------------------------
    #   Hardware wrappers – these are the *tools* exposed to ChatGPT.
    # ---------------------------------------------------------------------
    def drive(self, direction: str, speed: str = "medium") -> str:  # noqa: D401
        """Drive the car in *direction* at *speed* (slow/medium/fast)."""
        speed_map = {"slow": 1200, "medium": 2000, "fast": 3000}
        spd = speed_map.get(speed, 2000)

        if direction == "forward":
            self.car.set_motor_model(spd, spd, spd, spd)
        elif direction == "backward":
            self.car.set_motor_model(-spd, -spd, -spd, -spd)
        elif direction == "left":
            self.car.set_motor_model(-spd, -spd, spd, spd)
        elif direction == "right":
            self.car.set_motor_model(spd, spd, -spd, -spd)
        else:
            return "Unknown direction"

        # Let the movement run briefly then stop to avoid runaway.
        time.sleep(1.0)
        self.stop()
        return f"Driving {direction} at {speed} speed!"

    def led(self, color: str, pattern: str = "solid") -> str:
        """Change LED colour/pattern."""
        rgb = _colour_name_to_rgb(color)
        if pattern == "solid":
            self.led_ctrl.strip.set_all_led_color(*rgb)
            self.led_ctrl.strip.show()
        elif pattern == "blink":
            # Blink three times
            for _ in range(3):
                self.led_ctrl.strip.set_all_led_color(*rgb)
                self.led_ctrl.strip.show()
                time.sleep(0.25)
                self.led_ctrl.strip.set_all_led_color(0, 0, 0)
                self.led_ctrl.strip.show()
                time.sleep(0.25)
        elif pattern == "rainbow":
            for _ in range(30):  # ~1.5s of rainbowCycle
                self.led_ctrl.rainbowCycle(20)
        elif pattern == "chase":
            for _ in range(30):
                self.led_ctrl.following(50)
        else:
            return "Unknown pattern"
        return f"LEDs set to {color} with {pattern} pattern!"

    def stop(self) -> str:
        """Stop all movement."""
        self.car.set_motor_model(0, 0, 0, 0)
        # Turn off LEDs for good measure
        self.led_ctrl.strip.set_all_led_color(0, 0, 0)
        self.led_ctrl.strip.show()
        return "All stopped!"

    def chat(self) -> str:  # noqa: D401  – simplest placeholder
        """Generate a silly, funny response for general conversation."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "Say something super silly, funny, and random that would make a 5-year-old laugh! Be completely goofy and weird like a silly robot pet. Keep it very short and use lots of silly sounds or funny words!"}
                ],
                max_tokens=50
            )
            return response.choices[0].message.content or "*beep boop makes silly robot noises*"
        except Exception as e:
            print(f"Error generating silly response: {e}")
            # Fallback silly responses
            import random
            silly_responses = [
                "*BEEP BOOP* I'm a silly robot sandwich!",
                "Did you know I dream about flying tacos? VROOOOM!",
                "*makes robot dinosaur noises* RAWR-BEEP!",
                "I just learned how to wiggle my antenna! *wiggles*",
                "My favorite snack is motor oil with sprinkles! YUM!",
                "*spins in circles* WHEEEEE! I'm dizzy now!",
                "I think I'm part unicorn because I'm MAGICAL!",
                "*honks like a silly horn* HONK HONK BEEP!"
            ]
            return random.choice(silly_responses)

    def follow_voice(self, enthusiasm: str = "normal") -> str:
        """Come toward the user's voice like a loyal pet."""
        logger.debug(f"Following voice with {enthusiasm} enthusiasm")
        
        # Set speed based on enthusiasm
        speed_map = {"slow": 1500, "normal": 2000, "excited": 2800}
        speed = speed_map.get(enthusiasm, 2000)
        
        # Show excitement with head movement if available
        if not self.disable_head_movements and self.servo_ctrl:
            try:
                # Quick excited nod before coming
                for _ in range(2):
                    self.servo_ctrl.set_servo_pwm('1', 110)
                    time.sleep(0.1)
                    self.servo_ctrl.set_servo_pwm('1', 125)
                    time.sleep(0.1)
                self._center_head()
            except Exception as e:
                logger.error(f"Head movement failed during follow: {e}")
        
        # Move forward toward the voice
        try:
            # Move forward for a reasonable distance
            move_time = 1.5 if enthusiasm == "slow" else 2.0 if enthusiasm == "normal" else 2.5
            
            logger.debug(f"Moving forward at speed {speed} for {move_time}s")
            self.car.set_motor_model(speed, speed, speed, speed)
            time.sleep(move_time)
            
            # Stop
            self.car.set_motor_model(0, 0, 0, 0)
            logger.debug("Stopped movement")
            
            # Generate appropriate response
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"You just came running to your owner when they called you, like a good pet! You moved with {enthusiasm} enthusiasm. Say something short and excited that shows you're happy to come when called."}
                    ],
                    max_tokens=50
                )
                return response.choices[0].message.content or f"*comes running {enthusiasm}ly* Woof! I'm here!"
            except Exception as e:
                print(f"Error generating follow response: {e}")
                return f"*comes running {enthusiasm}ly* I'm here! I'm here!"
                
        except Exception as e:
            logger.error(f"Movement failed during follow: {e}")
            return "I tried to come to you but had trouble moving!"

    # ------------------------------------------------------------------
    #  Head movement methods for emotional expressions
    # ------------------------------------------------------------------
    def _center_head(self) -> None:
        """Center the head servos to neutral position."""
        logger.debug("Starting head centering")
        try:
            if self.servo_ctrl:
                self.servo_ctrl.set_servo_pwm('0', 80)  # Horizontal center
                logger.debug("Set horizontal servo to 80")
                self.servo_ctrl.set_servo_pwm('1', 115)  # Vertical center
                logger.debug("Set vertical servo to 115")
                time.sleep(0.5)
                logger.debug("Head centering completed")
            else:
                logger.info("Head movements disabled for debugging")
        except Exception as e:
            logger.error(f"Head centering failed: {e}")
            raise

    def _head_happy(self) -> None:
        """Happy wiggle - quick left-right movements."""
        for _ in range(3):
            if self.servo_ctrl:
                self.servo_ctrl.set_servo_pwm('0', 60)  # Left
                time.sleep(0.15)
                self.servo_ctrl.set_servo_pwm('0', 100)  # Right
                time.sleep(0.15)
            self._center_head()

    def _head_excited(self) -> None:
        """Excited nods - fast up-down movements."""
        for _ in range(4):
            if self.servo_ctrl:
                self.servo_ctrl.set_servo_pwm('1', 100)  # Down
                time.sleep(0.1)
                self.servo_ctrl.set_servo_pwm('1', 130)  # Up
                time.sleep(0.1)
            self._center_head()

    def _head_curious(self) -> None:
        """Curious tilt - slow side movements with pauses."""
        if self.servo_ctrl:
            self.servo_ctrl.set_servo_pwm('0', 60)  # Tilt left
            time.sleep(0.8)
            self.servo_ctrl.set_servo_pwm('0', 100)  # Tilt right
            time.sleep(0.8)
        self._center_head()

    def _head_confused(self) -> None:
        """Confused shake - small left-right shakes."""
        for _ in range(5):
            if self.servo_ctrl:
                self.servo_ctrl.set_servo_pwm('0', 75)  # Slightly left
                time.sleep(0.08)
                self.servo_ctrl.set_servo_pwm('0', 85)  # Slightly right
                time.sleep(0.08)
        self._center_head()

    def _head_sleepy(self) -> None:
        """Sleepy droop - slow downward movement."""
        for pos in range(115, 95, -2):
            if self.servo_ctrl:
                self.servo_ctrl.set_servo_pwm('1', pos)
                time.sleep(0.1)
        time.sleep(1)
        self._center_head()

    def _head_alert(self) -> None:
        """Alert posture - quick upward movement and scan."""
        if self.servo_ctrl:
            self.servo_ctrl.set_servo_pwm('1', 135)  # Look up
            time.sleep(0.3)
            self.servo_ctrl.set_servo_pwm('0', 60)   # Look left
            time.sleep(0.3)
            self.servo_ctrl.set_servo_pwm('0', 100)  # Look right
            time.sleep(0.3)
        self._center_head()

    def _head_playful(self) -> None:
        """Playful bounces - mix of movements."""
        # Quick bounce
        if self.servo_ctrl:
            self.servo_ctrl.set_servo_pwm('1', 105)
            time.sleep(0.1)
            self.servo_ctrl.set_servo_pwm('1', 125)
            time.sleep(0.1)
        # Side wiggle
        if self.servo_ctrl:
            self.servo_ctrl.set_servo_pwm('0', 70)
            time.sleep(0.2)
            self.servo_ctrl.set_servo_pwm('0', 90)
            time.sleep(0.2)
        self._center_head()

    def _head_listening(self) -> None:
        """Subtle listening movement - slight tilt."""
        logger.debug("Starting listening head position")
        try:
            if self.servo_ctrl:
                self.servo_ctrl.set_servo_pwm('0', 75)  # Slight tilt
                logger.debug("Set horizontal listening position")
                self.servo_ctrl.set_servo_pwm('1', 120)  # Slight up
                logger.debug("Set vertical listening position")
                time.sleep(0.5)
                logger.debug("Listening head position completed")
            else:
                logger.info("Head movements disabled for debugging")
        except Exception as e:
            logger.error(f"Head listening position failed: {e}")
            # Don't raise - continue even if head movement fails

    def head_movement(self, emotion: str) -> str:
        """Express emotions through head movements with appropriate speech."""
        emotion_map = {
            "happy": self._head_happy,
            "excited": self._head_excited,
            "curious": self._head_curious,
            "confused": self._head_confused,
            "sleepy": self._head_sleepy,
            "alert": self._head_alert,
            "playful": self._head_playful,
        }
        
        if emotion in emotion_map:
            # Do the head movement
            emotion_map[emotion]()
            
            # Generate appropriate response for the emotion
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"You're feeling {emotion} and just did a cute head movement to show it. Say something short and appropriate that Bitsy would say while feeling {emotion}. Keep it very brief and excited like a happy puppy."}
                    ],
                    max_tokens=50
                )
                return response.choices[0].message.content or f"*does {emotion} head movement*"
            except Exception as e:
                print(f"Error generating emotion response: {e}")
                return f"*wiggles head {emotion}ly*"
        else:
            return "Unknown emotion for head movement"

    # ------------------------------------------------------------------
    #  STT → ChatGPT → Action loop
    # ------------------------------------------------------------------
    def _listen_once(self) -> Optional[str]:
        """Capture one utterance and return transcript or None on failure."""
        logger.debug("Starting _listen_once")
        try:
            with self.microphone as src:
                print("Listening…")
                logger.debug("Microphone opened successfully")
                self._head_listening()  # Show that Bitsy is listening
                logger.debug("Head listening position set")
                try:
                    logger.debug("Starting recogniser.listen()")
                    
                    # Check what's using the audio device
                    try:
                        import subprocess
                        result = subprocess.run(['sudo', 'lsof', '/dev/snd/*'], 
                                              capture_output=True, text=True, timeout=2)
                        if result.stdout:
                            logger.debug(f"Audio device usage: {result.stdout}")
                        else:
                            logger.debug("No processes using audio device")
                    except:
                        logger.debug("Could not check audio device usage")
                    
                    # Use thread-based timeout instead of signal (more reliable with C libraries)
                    import concurrent.futures
                    
                    def do_listen():
                        return self.recogniser.listen(src, timeout=10, phrase_time_limit=6)
                    
                    # Run microphone operation in separate thread with timeout
                    logger.debug("Creating ThreadPoolExecutor for microphone operation")
                    executor = concurrent.futures.ThreadPoolExecutor()
                    try:
                        logger.debug("Submitting microphone task to thread pool")
                        future = executor.submit(do_listen)
                        try:
                            logger.debug("Waiting for microphone result with 12s timeout")
                            audio = future.result(timeout=12)  # 12 second timeout
                            logger.debug("Audio captured successfully")
                        except concurrent.futures.TimeoutError:
                            logger.error("Microphone operation timed out - audio device may be locked")
                            logger.debug("Attempting to cancel future task")
                            future.cancel()  # Try to cancel the hanging task
                            logger.debug("Forcing executor shutdown")
                            executor.shutdown(wait=False)  # Don't wait for hanging threads
                            logger.debug("Centering head after timeout")
                            self._center_head()
                            logger.debug("_listen_once returning None due to timeout")
                            return None
                        except Exception as e:
                            logger.error(f"Microphone operation failed: {e}")
                            logger.debug("Attempting to cancel future task after exception")
                            future.cancel()
                            logger.debug("Forcing executor shutdown after exception")
                            executor.shutdown(wait=False)
                            logger.debug("Centering head after microphone error")
                            self._center_head()
                            logger.debug("_listen_once returning None due to microphone error")
                            return None
                    finally:
                        # Always ensure executor is shut down
                        try:
                            logger.debug("Final executor shutdown in finally block")
                            executor.shutdown(wait=False)
                        except:
                            logger.debug("Executor already shut down")
                    
                    logger.debug("ThreadPoolExecutor cleanup completed successfully")
                        
                except sr.WaitTimeoutError:
                    logger.debug("Listen timeout - returning to center")
                    self._center_head()  # Return to center if timeout
                    logger.debug("_listen_once returning None due to WaitTimeoutError")
                    return None
            
            logger.debug("Processing audio with Google STT")
            try:
                transcript = self.recogniser.recognize_google(audio)
                logger.debug(f"STT success: {transcript}")
                print(f"Heard: {transcript}")
                self._center_head()  # Center head after successful recognition
                logger.debug(f"_listen_once returning transcript: {transcript}")
                return transcript
            except Exception as exc:
                logger.error(f"STT failed: {exc}")
                print(f"STT failed: {exc}")
                self._center_head()  # Center head on failure
                logger.debug("_listen_once returning None due to STT failure")
                return None
        except Exception as e:
            logger.error(f"Critical error in _listen_once: {e}")
            logger.exception("Full traceback for _listen_once error:")
            try:
                self._center_head()
            except:
                logger.error("Failed to center head after critical error")
            logger.debug("_listen_once returning None due to critical error")
            return None

    def _speak(self, text: str) -> None:
        """Speak *text* using OpenAI TTS with graceful fallback and verbose logging."""
        logger.debug(f"Starting _speak with text: {text[:50]}...")
        print("[Bitsy] Speaking:", text)
        tmp_path = "bitsy_response.mp3"
        try:
            logger.debug("Generating TTS audio")
            t0 = time.time()
            speech = self.client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text,
                timeout=15  # 15 second timeout to prevent hanging
            )
            speech.stream_to_file(tmp_path)
            logger.debug(f"TTS generated in {time.time() - t0:.2f}s")
            print(f"[TTS] Saved to {tmp_path} ({os.path.getsize(tmp_path)} bytes) in {time.time() - t0:.2f}s")
        except Exception as exc:
            logger.error(f"TTS generation failed: {exc}")
            print("[TTS] Failed – using espeak fallback:", exc)
            self._fallback_tts(text)
            return

        try:
            logger.debug("Starting audio playback")
            _play_mp3(tmp_path)
            logger.debug("Audio playback completed")
            print("[TTS] Playback finished")
        except Exception as exc:
            logger.error(f"Audio playback failed: {exc}")
            print("[TTS] Playback error – using espeak fallback:", exc)
            self._fallback_tts(text)
        finally:
            logger.debug("Starting cleanup")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug("Temp audio file removed")
            # Center head after speaking
            try:
                logger.debug("Centering head after speaking")
                self._center_head()
                logger.debug("Head centered after speaking")
            except Exception as e:
                logger.error(f"Head centering error: {e}")
                print(f"Head centering error: {e}")
            # give ALSA more time before reopening microphone
            logger.debug("Starting 1.5s delay before microphone")
            time.sleep(1.5)
            logger.debug("_speak method completed")

    def _fallback_tts(self, text: str) -> None:
        """Very simple espeak fallback so the user still hears something."""
        os.system(f'espeak "{text}" 2>/dev/null')

    def _chatgpt_round(self, transcript: str) -> str:
        """Send *transcript* to ChatGPT using function-calling and act on the results."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": transcript},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # fast & cheap, adjust if needed
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # ------------------------------------------------------------------
        #  Did the assistant choose a tool?
        # ------------------------------------------------------------------
        if msg.tool_calls:
            responses = []
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                print(f"[ChatGPT] Requested tool: {name} {args}")
                try:
                    result = getattr(self, name)(**args)
                except Exception as exc:
                    result = f"Oops, I had trouble with {name}: {exc}"
                responses.append(result)
            return " ".join(responses)

        # No tool call – just chat content
        return msg.content or ""

    def _generate_introduction(self) -> str:
        """Generate an excited introduction when Bitsy starts up."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": "You just started up and are ready to listen for commands! Give a super excited, short introduction with your name, Bitsy,that shows you're happy to see everyone and ready to play. Mention that you're ready to drive, play with lights, and chat and play legos or transformers. Keep it brief and energetic like an excited puppy who just woke up!"}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content or "*BEEP BEEP* Hi everyone! Bitsy is online and ready to play!"
        except Exception as e:
            logger.error(f"Error generating introduction: {e}")
            # Fallback introduction
            return "*BEEP BEEP* Hi everyone! I'm Bitsy and I'm SO excited to play! I can drive around, change my lights, or just chat with you! What should we do first?"

    def _introduce_myself(self) -> None:
        """Give an excited introduction when starting up."""
        logger.info("Bitsy introducing itself")
        introduction = self._generate_introduction()
        print(f"[Bitsy] Starting up: {introduction}")
        
        # Do a little excited head movement if available
        if not self.disable_head_movements and self.servo_ctrl:
            try:
                self._head_excited()
            except Exception as e:
                logger.error(f"Head movement failed during introduction: {e}")
        
        # Speak the introduction
        self._speak(introduction)

    # ------------------------------------------------------------------
    #  Public run loop
    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        """Main loop: listen, chat, act, speak."""
        logger.info("Starting Bitsy main loop")
        self._introduce_myself() # Introduce Bitsy at startup
        while True:
            try:
                logger.info("=== Starting new conversation cycle ===")
                logger.debug(f"Loop iteration at {time.time()}")
                logger.debug("About to call _listen_once()")
                transcript = self._listen_once()
                logger.debug(f"_listen_once() returned: {transcript}")
                
                if not transcript:
                    logger.debug("No transcript received, continuing to next cycle")
                    time.sleep(0.5)  # Small delay to prevent rapid cycling
                    logger.debug("About to execute continue statement - jumping to start of loop")
                    continue
                    
                logger.debug(f"Processing transcript: {transcript}")
                reply = self._chatgpt_round(transcript)
                logger.debug(f"Generated reply: {reply[:100]}...")
                print(f"Bitsy: {reply}")
                if reply:
                    self._speak(reply)
                logger.debug("=== Conversation cycle completed ===")
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.exception("Full traceback:")  # This will show the full stack trace
                continue


# ---------------------------------------------------------------------------
#  Convenience helpers
# ---------------------------------------------------------------------------

def _find_default_mic() -> Optional[int]:
    """Attempt to choose a reasonable microphone index on the Pi."""
    try:
        names = sr.Microphone.list_microphone_names()
        for idx, name in enumerate(names):
            if "usb" in name.lower() or "mic" in name.lower():
                return idx
        return 0 if names else None
    except Exception:
        return None


def main() -> None:  # pragma: no cover
    import sys
    
    # Check for debug flag
    disable_head = "--no-head" in sys.argv
    if disable_head:
        print("Running in debug mode with head movements DISABLED")
    
    idx = _find_default_mic()
    agent = BitsyAgent(mic_index=idx, disable_head_movements=disable_head)
    agent.run_forever()


if __name__ == "__main__":
    main() 