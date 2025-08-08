import pyaudio
import numpy as np
import asyncio


from TimeBomb import ResettableTimer

from STT_Model import load_stt_model
stt_inference = load_stt_model()

from Speech_Detector import load_speech_detect_model
speech_detect = load_speech_detect_model()


# --- PyAudio Setup ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("\n🎤 Listening... Press Ctrl+C to stop.")

# Global waveform accumulator
waveform_np = np.array([], dtype=np.float32)

async def transcribe_and_reset():
    """Called when timeout happens - transcribe accumulated audio"""
    global waveform_np
    if len(waveform_np) > 0:
        txt = stt_inference(waveform_np)
        print(f"Transcription: {txt}")
        waveform_np = np.array([], dtype=np.float32)  # Reset the buffer
    else:
        print("\n⏰ Timeout - no audio to transcribe")

async def main():
    global waveform_np
    
    # Create timer that triggers transcription after 3 seconds of no speech
    timer = ResettableTimer(0.9, transcribe_and_reset)
    
    try:
        while True:
            # Read audio chunk in a thread to not block event loop
            loop = asyncio.get_event_loop()
            audio_chunk_bytes = await loop.run_in_executor(
                None, lambda: stream.read(CHUNK, exception_on_overflow=False)
            )
            
            # Convert to numpy
            audio_chunk_np_i16 = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
            audio_chunk_np_f32 = audio_chunk_np_i16.astype(np.float32) / 32768.0
            
            # Accumulate waveform
            waveform_np = np.concatenate([waveform_np, audio_chunk_np_f32])
            
            # VAD detection
            is_speech = speech_detect(audio_chunk_np_f32)
            if is_speech:
                print("SPEECH DETECTED!", end='\r')
                timer.reset()
            else:
                print("Silence...                                    ", end='\r')
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        timer.cancel()
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Stream closed.")

if __name__ == "__main__":
    asyncio.run(main())