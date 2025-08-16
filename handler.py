import runpod
import os
import requests
from pyannote.audio import Pipeline

# This part runs only once when the worker starts
# It loads the model into memory for speed
hf_token = os.environ.get('HUGGING_FACE_TOKEN')
if not hf_token:
    raise ValueError("HUGGING_FACE_TOKEN environment variable not set.")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token
)

# This is the function that handles incoming requests
def handler(job):
    job_input = job.get('input', {})
    audio_url = job_input.get('audio_url')

    if not audio_url:
        return {"error": "Please provide 'audio_url' in the input."}

    # Download the audio file from the URL provided by n8n
    try:
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        temp_audio_file = '/tmp/audio.wav'
        with open(temp_audio_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download audio file: {str(e)}"}

    # Run the diarization model on the file
    diarization = pipeline(temp_audio_file)

    # Format the result into a clean list
    diarization_result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_result.append({
            'speaker': speaker,
            'start': round(turn.start, 2),
            'stop': round(turn.end, 2)
        })

    # Clean up the temporary file and return the result
    os.remove(temp_audio_file)
    return diarization_result

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
