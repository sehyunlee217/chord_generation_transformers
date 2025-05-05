# Transformer-based Guitar Chord Generator 🎸 

Generate genre-aware guitar chord progressions using a transformer model — trained from scratch on over 600,000 songs!

Built for fun, learning, and giving myself an excuse to practice guitar a bit more.

<img width="1551" alt="Screenshot 2025-05-05 at 00 16 40" src="https://github.com/user-attachments/assets/4f580a66-8e65-481d-9086-e28a23a1a44a" />

## Summary

A transformer-based model trained on the [CHORDONOMICON dataset]([https://arxiv.org/abs/2410.22046](https://huggingface.co/datasets/ailsntua/Chordonomicon)) dataset — a massive collection of 666,000 songs and their chord progressions — to generate realistic, genre-aware, and playable chord sequences.

Whether you’re a guitarist, music producer, or machine learning enthusiast, this lets you:
- Generate chords in 12 genres
- Customize number of chords, randomness (temperature), and style
- Hear the generated sequence via MIDI 
- Play along with your guitar 

## Demo: 
- [Listen to generated chords being played](https://drive.google.com/file/d/1Rx7A4NfZUcxuh6L-_Dt9X3Nrx_gJvYwW/view?usp=sharing)
- [See it being generated through MIDI in Garage Band](https://drive.google.com/file/d/1n0jxeBTrrH2ndUn4KyohJAHzYf4LBNij/view?usp=sharing)
- [Hear me play guitar using the same generated chords]

## Dataset

CHORDONOMICON:
Kantarelis et al., 2024 – “A Dataset of 666,000 Songs and their Chord Progressions”
[arXiv:2410.22046]

	•	600k+ songs across 12 genres
	•	4000+ unique chords
	•	Cleaned & processed using PySpark for scalable preprocessing

## How It Works

### Model Architecture
- Transformer model inspired by Attention Is All You Need
- Trained from scratch (for learning) using PyTorch
- 4 layers of encoder/decoders, added genre embedding for input, d_model=256, ffn_size=256. 
- Supports genre conditioning, masking, and decoding
- Blog post explaining line-by-line of the model coming soon at [my website](https://www.shyun.dev/)

### Training
- GPU: Trained on RunPod with a single RTX 4090
- See ```genre_train_runpod.ipynb``` for full training pipeline on runpod

## Supported Genres
```
{
  'alternative': 0, 'country': 1, 'electronic': 2, 'jazz': 3, 
  'metal': 4, 'pop': 5, 'pop rock': 6, 'punk': 7, 'rap': 8, 
  'reggae': 9, 'rock': 10, 'soul': 11
}
```

## Features

 - Generate realistic chord progressions from scratch
 - Genre conditioning for stylistic accuracy
 - Adjustable parameters: sequence length, temperature, starting chords
 - Play audio output via  MIDI (IAC Driver on macOS)
 - Bonus: Watch me play the generated chords on my guitar!

## Requirements
- Python 3.8+
- PyTorch
- PySpark (Can use pandas) 
- mido (for MIDI)

## Example Usage: 
```
  python chord_generator.py \
  --initial "G D A Emin" \
  --genre 10 \
  --num_chords 20 \
  --play \
  --style rock
```

