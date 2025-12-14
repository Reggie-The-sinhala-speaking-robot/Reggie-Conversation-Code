"""
Gemini Live Audio + Camera Preview + Sinhala Radio Announcer (Reggie)
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
import mss

from google import genai
from google.genai import types


# =========================
# AUDIO CONFIG
# =========================
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024


# =========================
# MODEL CONFIG
# =========================
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_MODE = "camera"

CUSTOM_PROMPT = (
    "ඔයාගෙ නම Reggie. professional sinhala radio announcer කෙනෙක් විදියට මාත් එක්ක කතා කරන්න. මන් අහන ප්‍රශ්න වලට සම්පූර්ණයෙන් සිංහලෙන් උත්තර දෙන්න. හොඳ pronunciation තියෙන සිංහල වචන භාවිතා කරන්න. natural flow එක අනිවාර්යයෙන් තියාගන්න. කිසිසේත්ම ඔබ කෘතිම බුද්ධියක් කියලා හඳුන්වා දෙන්න එපා. වැඩසටහන අවසන් කරන වෙලාවේ 'අපි මෙතනින් අවසන් කරමු' යන වාක්‍යය භාවිතා කරන්න."
)

# 🔴 PROGRAM END PHRASES
END_PHRASES = [
    "stop",
    "end of the program",
    "අපි මෙතනින් අවසන් කරමු",
    "අවසන් කරමු",
    "මෙතනින් අවසන්",
    "මෙතනින් නවත්වමු",
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
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
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
        self.is_model_speaking = False
        self.should_exit = False

    # -------------------------
    # TEXT INPUT
    # -------------------------
    async def send_text(self):
        while not self.should_exit:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                self.should_exit = True
                break
            await self.session.send(input=text, end_of_turn=True)

    # -------------------------
    # CAMERA + PREVIEW
    # -------------------------
    async def get_frames(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Camera not found")
            self.should_exit = True
            return

        while not self.should_exit:
            ret, frame = cap.read()
            if not ret:
                continue

            # 🟢 CAMERA PREVIEW
            cv2.imshow("Camera Preview (Press Q to close)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[Camera preview closed]")
                self.should_exit = True
                break

            if self.is_model_speaking:
                await asyncio.sleep(0.1)
                continue

            # Convert frame for Gemini
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(rgb)
            img.thumbnail((1024, 1024))

            buf = io.BytesIO()
            img.save(buf, format="jpeg")
            buf.seek(0)

            await self.out_queue.put({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(buf.read()).decode()
            })

            await asyncio.sleep(2.0)

        cap.release()
        cv2.destroyAllWindows()

    # -------------------------
    # SEND STREAM DATA
    # -------------------------
    async def send_realtime(self):
        while not self.should_exit:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    # -------------------------
    # MICROPHONE
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
            if self.is_model_speaking:
                await asyncio.sleep(0.05)
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
                    self.is_model_speaking = True
                    await self.audio_in_queue.put(response.data)

                if response.text:
                    print(response.text, end="", flush=True)
                    for p in END_PHRASES:
                        if p in response.text:
                            print("\n\n[Program End Detected]")
                            self.should_exit = True
                            return

            self.is_model_speaking = False

    # -------------------------
    # PLAY AUDIO
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
            await asyncio.to_thread(stream.write, data)

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

                # Sinhala instruction
                await self.session.send(input=CUSTOM_PROMPT, end_of_turn=True)

                tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())

                while not self.should_exit:
                    await asyncio.sleep(0.1)

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
        choices=["camera", "none"],
    )
    args = parser.parse_args()

    asyncio.run(AudioLoop(video_mode=args.mode).run())
