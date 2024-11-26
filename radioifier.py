import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import audio2numpy as a2n
import subprocess


def low_pass_filter(data, cutoff_frequency, sample_rate, order=3):
    """Apply a low-pass filter to the audio data."""
    b, a = butter(order, cutoff_frequency / (sample_rate / 2), btype='low')
    return lfilter(b, a, data)


def high_pass_filter(data, cutoff_frequency, sample_rate, order=3):
    """Apply a high-pass filter to the audio data."""
    b, a = butter(order, cutoff_frequency / (sample_rate / 2), btype='high')
    return lfilter(b, a, data)


def apply_filters(audio_data, rate, low_cut, high_cut):
    """
    Apply low-pass and high-pass filters to the audio data.
    Handles stereo or mono audio.
    """
    if audio_data.ndim == 1:  # Mono audio
        audio_data = low_pass_filter(audio_data, high_cut, rate)
        audio_data = high_pass_filter(audio_data, low_cut, rate)
    elif audio_data.ndim == 2:  # Stereo audio
        audio_data = np.array([
            low_pass_filter(channel, high_cut, rate)
            for channel in audio_data.T
        ]).T
        audio_data = np.array([
            high_pass_filter(channel, low_cut, rate)
            for channel in audio_data.T
        ]).T
    else:
        raise ValueError("Unsupported audio format.")
    return audio_data


if __name__ == "__main__":
    parser = ArgumentParser(description="Apply low-pass and high-pass filters to audio files.")
    parser.add_argument("--input_file", type=str, required=True, help="Input file pattern (e.g., '*.wav').")
    parser.add_argument("--cutoff_low", type=int, default=500, help="Low cutoff frequency (Hz).")
    parser.add_argument("--cutoff_high", type=int, default=5000, help="High cutoff frequency (Hz).")
    parser.add_argument("--skip_mp3", type=int, default=False, help="skips converting to mp3 file.")
    args = parser.parse_args()

    input_file = args.input_file
    low_cutoff = args.cutoff_low
    high_cutoff = args.cutoff_high
    skip_mp3_conversion = args.skip_mp3
    count = 0
    finished_files = []

    # Gather input files
    files = glob(input_file)
    if not files:
        print("No files found. Please check the input pattern.")
        exit(1)

    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for file in files:
        try:
            print(f"Processing: {file}")
            filename_noext = Path(file).stem
            output_path = os.path.join(output_dir, f"{filename_noext}_filtered.wav")

            # Load audio file
            audio_data, sr = a2n.open_audio(file)
            if sr is None or audio_data is None:
                print(f"Skipping {file}: Unable to read audio data.")
                continue

            # Apply filters
            audio_data_filtered = apply_filters(audio_data, sr, low_cutoff, high_cutoff)
            #audio_data_filtered_resampled = resample(audio_data_filtered, 164000)
            # Write output file
            wavfile.write(output_path, sr, audio_data_filtered.astype(np.float32))
            print(f"Written filtered audio to: {output_path}")
            finished_files.append(output_path)
            count += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")
    print(f"Finished processing {count} files.")

    print(f"Converting {count} files into mp3 files.")
    for file in finished_files:
      filename_noext = Path(file).stem
      output_path = os.path.join(output_dir, f"{filename_noext}.mp3")

      print(f"converting {file} to {output_path}")
      result = subprocess.run(f"ffmpeg -y -i \"{file}\" \"{output_path}\"", capture_output=True, text=True)
      if result.returncode != 0:
        print(" === Error converting. details below")
        print("result.stderr")
   
    print(f"Conversion of {count} files complete.")