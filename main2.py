"""
Gemini Live Audio + Camera/Screen Streaming
Sinhala Radio Announcer (Reggie) – ULTRA LOW LATENCY
"""

import os
import asyncio
import base64
import io
import traceback
import argparse

import cv2
import pyaudio
import PIL.Image

from google import genai
from google.genai import types


# =========================
# AUDIO CONFIG - OPTIMIZED FOR LOW LATENCY
# =========================
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512  # Reduced from 1024 for faster processing


# =========================
# MODEL CONFIG
# =========================
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_MODE = "camera"

CUSTOM_PROMPT = (
    "ඔයාගෙ නම Reggie. professional sinhala radio announcer කෙනෙක් විදියට මාත් එක්ක කතා කරන්න. මන් අහන ප්‍රශ්න වලට Singlish (සිංහල සහ ඉංග්‍රීසි මිශ්‍රව) උත්තර දෙන්න. හොඳ pronunciation තියෙන සිංහල වචන භාවිතා කරන්න. natural flow එක තියාගන්න. කිසිසේත්ම ඔබ කෘතිම බුද්ධියක් කියලා හඳුන්වා දෙන්න එපා. Keep responses SHORT and conversational - 2-3 sentences maximum unless asked for detail."
)

# USER-ONLY END COMMANDS
USER_END_COMMANDS = [
    "අවසන් කරන්න",
    "අවසන් කරමු",
    "මෙතනින් ඉවර කරමු",
    "මෙතනින් නවත්තමු",
    "stop",
    "end",
    "ඉවර කරන්න",
    "quit",
]


# =========================
# GEMINI CLIENT
# =========================
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_LOW",  # Lower resolution for speed
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Puck"
            )
        )
    ),
)

pya = pyaudio.PyAudio()


# =========================
# MAIN CLASS
# =========================
class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.session = None
        self.is_playing_audio = False
        self.should_exit = False

    # -------------------------
    # TEXT INPUT
    # -------------------------
    async def send_text(self):
        while not self.should_exit:
            text = await asyncio.to_thread(input, "message > ")
            if any(cmd in text.lower() for cmd in USER_END_COMMANDS):
                print("[User requested program end]")
                self.should_exit = True
                return
            await self.session.send(input=text, end_of_turn=True)

    # -------------------------
    # CAMERA - OPTIMIZED
    # -------------------------
    async def get_frames(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 10)  # Lower FPS
        
        while not self.should_exit:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(rgb)
            img.thumbnail((640, 640))  # Smaller thumbnail

            buf = io.BytesIO()
            img.save(buf, format="jpeg", quality=60)  # Lower quality for speed
            buf.seek(0)

            await self.out_queue.put({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(buf.read()).decode()
            })

            await asyncio.sleep(4.0)  # Reduced frame rate

        cap.release()

    # -------------------------
    # SEND DATA
    # -------------------------
    async def send_realtime(self):
        while not self.should_exit:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    # -------------------------
    # MICROPHONE - OPTIMIZED
    # -------------------------
    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        stream = pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        while not self.should_exit:
            if self.is_playing_audio:
                await asyncio.sleep(0.005)  # Reduced from 0.02
                continue

            data = await asyncio.to_thread(
                stream.read, CHUNK_SIZE, exception_on_overflow=False
            )

            await self.out_queue.put({
                "mime_type": "audio/pcm",
                "data": data
            })

        stream.close()

    # -------------------------
    # RECEIVE AUDIO
    # -------------------------
    async def receive_audio(self):
        while not self.should_exit:
            turn = self.session.receive()
            async for response in turn:
                if response.data:
                    await self.audio_in_queue.put(response.data)

                if response.text:
                    print(response.text, end="", flush=True)

    # -------------------------
    # PLAY AUDIO - OPTIMIZED
    # -------------------------
    async def play_audio(self):
        stream = pya.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        while not self.should_exit:
            data = await self.audio_in_queue.get()
            self.is_playing_audio = True
            await asyncio.to_thread(stream.write, data)
            self.is_playing_audio = False

        stream.close()

    # -------------------------
    # MAIN
    # -------------------------
    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                await self.session.send(input=CUSTOM_PROMPT, end_of_turn=True)

                tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())

                while not self.should_exit:
                    await asyncio.sleep(0.05)  # Reduced from 0.1

        except ExceptionGroup as e:
            traceback.print_exception(e)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()

    asyncio.run(AudioLoop(video_mode=args.mode).run())