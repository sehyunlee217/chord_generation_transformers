import torch
import pickle
import argparse
import time
import mido
from genre_aware_model import build_transformer

def load_resources():
    # Load chord mappings
    with open("chord_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    
     # Load processed sequences
    with open("processed_sequences.pkl", "rb") as f:
        data = pickle.load(f)
    
    chord_to_id = mappings["chord_to_id"]
    id_to_chord = mappings["id_to_chord"]
    
    # get genres for sequences
    genres_for_sequences = data["genres_for_sequences"]
    unique_genres = list(set(genres_for_sequences))
        # fixed genre mapping for consistency
    genre_to_id = {
        'alternative': 0,
        'country': 1,
        'electronic': 2,
        'jazz': 3,
        'metal': 4,
        'pop': 5,
        'pop rock': 6,
        'punk': 7,
        'rap': 8,
        'reggae': 9,
        'rock': 10,
        'soul': 11
    }
    id_to_genre = {i: genre for genre, i in genre_to_id.items()}

    print(genre_to_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model / same param as trainig
    model = build_transformer(
        src_vocab_size=len(chord_to_id),
        tgt_vocab_size=len(chord_to_id),
        src_seq_len=200,
        tgt_seq_len=200,
        genre_len=len(genre_to_id),
        d_model=256,
        ffn_size=256,
        dropout=0.2
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load("chord_final_model.pt", map_location=device))
    model.eval() 
    
    return model, chord_to_id, id_to_chord, genre_to_id, id_to_genre, device

def generate_chords(model, initial_sequence, genre, device, num_chords=5, temperature=1.1, top_k=10, chord_to_id=None, id_to_chord=None):
    """Generate chord progression based on initial sequence and genre"""

    # Convert the initial sequence to tokens
    initial_sequence_ids = [chord_to_id[chord] for chord in initial_sequence]
    src_input = torch.tensor(initial_sequence_ids).unsqueeze(0).to(device)
    
    # Convert genre to tensor
    genre_tensor = torch.tensor([genre]).to(device)
    
    # Create source mask
    src_mask = (src_input != chord_to_id["PAD"]).unsqueeze(1).unsqueeze(2).to(device)
    
    # Encode the input sequence
    enc_out = model.encode(src_input, genre_tensor, src_mask=src_mask)
    
    # Initialize target input
    tgt_input = src_input.clone()
    
    generated_sequence = []
    
    # Generate chords one by one
    for _ in range(num_chords):
        # Create a target mask (lower triangular)
        tgt_mask = torch.tril(torch.ones((tgt_input.size(1), tgt_input.size(1)))).unsqueeze(0).unsqueeze(0).to(device)
        
        # Decode
        dec_out = model.decode(tgt_input, enc_out, genre_tensor, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Project to vocabulary
        logits = model.project(dec_out)
        
        # Apply temperature scaling
        logits = logits[:, -1, :] / temperature
        
        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Top-K sampling
        top_k_values, top_k_indices = torch.topk(probs, top_k, dim=-1)
        
        # Sample from the top-k most likely chords
        next_chord_idx = torch.multinomial(top_k_values, 1).item()
        next_chord = top_k_indices[0, next_chord_idx].item()
        
        # Add to generated sequence
        generated_sequence.append(next_chord)
        
        # Update target input with new token
        tgt_input = torch.cat([tgt_input, torch.tensor([[next_chord]], device=device)], dim=1)
    
    # Convert IDs back to chord names
    return [id_to_chord[chord_id] for chord_id in generated_sequence]

def get_chord_notes(chord_name):
    """Dynamically generate MIDI notes for any chord name"""
    # Base MIDI notes for root notes (C3-B3)
    root_notes = {
        'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
        'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 
        'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
    }
    
    # Common chord structures as semitone intervals
    chord_structures = {
        '': [0, 4, 7],         # Major
        'm': [0, 3, 7],        # Minor
        '7': [0, 4, 7, 10],    # Dominant 7th
        'maj7': [0, 4, 7, 11], # Major 7th
        'm7': [0, 3, 7, 10],   # Minor 7th
        'dim': [0, 3, 6],      # Diminished
        'dim7': [0, 3, 6, 9],  # Diminished 7th
        'aug': [0, 4, 8],      # Augmented
        'sus2': [0, 2, 7],     # Suspended 2nd
        'sus4': [0, 5, 7],     # Suspended 4th
        '6': [0, 4, 7, 9],     # Major 6th
        'm6': [0, 3, 7, 9],    # Minor 6th
        '9': [0, 4, 7, 10, 14],# Dominant 9th
        'add9': [0, 4, 7, 14], # Add 9
        'madd9': [0, 3, 7, 14] # Minor add 9
    }
    
    # Parse chord name
    if not chord_name:
        return []
        
    # Extract root note
    root = chord_name[0]
    if len(chord_name) > 1 and chord_name[1] in ['#', 'b']:
        root += chord_name[1]
        quality = chord_name[2:]
    else:
        quality = chord_name[1:]
    
    # Check if root note exists
    if root not in root_notes:
        print(f"Unknown root note in '{chord_name}'")
        return []
    
    # Get chord structure or default to major
    structure = chord_structures.get(quality, chord_structures[''])
    
    # Generate MIDI notes
    midi_notes = [root_notes[root] + interval for interval in structure]
    return midi_notes

def play_arpeggiated_chord(outport, notes, duration=0.9, pattern="up"):
    """Play the chord as an arpeggio instead of all at once"""
    note_duration = duration / (len(notes) * 2)  # Allow for note overlap
    
    if pattern == "up":
        play_sequence = notes
    elif pattern == "down":
        play_sequence = notes[::-1]
    elif pattern == "updown":
        play_sequence = notes + notes[1:-1][::-1]  # Up then down without repeating top/bottom
    else:  # "random"
        import random
        play_sequence = random.sample(notes, len(notes))
    
    # Start notes with slight overlap
    active_notes = []
    for note in play_sequence:
        # Play note
        outport.send(mido.Message('note_on', note=note, velocity=70))
        active_notes.append(note)
        time.sleep(note_duration)
        
        # Release oldest note if we have more than 2 playing
        if len(active_notes) > 2:
            oldest = active_notes.pop(0)
            outport.send(mido.Message('note_off', note=oldest, velocity=70))
    
    # Let final notes ring a bit longer
    time.sleep(duration * 0.3)
    
    # Release any remaining notes
    for note in active_notes:
        outport.send(mido.Message('note_off', note=note, velocity=70))

def play_rhythmic_chord(outport, notes, duration=0.9, pattern="basic"):
    """Play the chord with a rhythmic pattern"""
    base_duration = duration / 8  # Divide into 8 time units
    
    patterns = {
        "basic": [1, 0, 0.5, 0, 1, 0, 0.5, 0],  # 1=full, 0.5=half, 0=rest
        "waltz": [1, 0, 0.5, 0.5, 0.5, 0],
        "rock": [1, 0.3, 0.5, 0.3, 0.8, 0.3, 0.5, 0.3],
        "ballad": [1, 0, 0, 0, 0.7, 0, 0, 0],
    }
    
    rhythm = patterns.get(pattern, patterns["basic"])
    
    for strength in rhythm:
        if strength > 0:  # Not a rest
            # Play chord with varying velocity based on beat strength
            velocity = int(60 + (strength * 40))
            for note in notes:
                outport.send(mido.Message('note_on', note=note, velocity=velocity))
            time.sleep(base_duration)
            for note in notes:
                outport.send(mido.Message('note_off', note=note, velocity=velocity))
        else:
            # Rest
            time.sleep(base_duration)

def play_chord_with_bassline(outport, chord_notes, duration=0.9):
    """Play chord with a simple bassline"""
    # Bass note is the root, but an octave lower
    bass_note = chord_notes[0] - 12
    
    # Play bass note
    outport.send(mido.Message('note_on', note=bass_note, velocity=80))
    time.sleep(duration * 0.2)
    
    # Play chord
    for note in chord_notes:
        outport.send(mido.Message('note_on', note=note, velocity=60))
    
    time.sleep(duration * 0.6)
    
    # Release chord
    for note in chord_notes:
        outport.send(mido.Message('note_off', note=note, velocity=60))
        
    time.sleep(duration * 0.2)
    
    # Release bass note
    outport.send(mido.Message('note_off', note=bass_note, velocity=80))


def play_chord_progression(chord_progression, duration=1, style="mixed"):
    """Play the generated chord progression as MIDI with specified style"""
    try:
        import mido
        import time
        import random
        
        # List available MIDI ports
        available_ports = mido.get_output_names()
        if not available_ports:
            print("\nNo MIDI ports found! Using pygame instead.")
        
        print(f"Using MIDI port: {available_ports[0]}")
        outport = mido.open_output(available_ports[0])
        
        # Use specified style instead of random choice
        print(f"Playing with '{style}' style")
        
        # Define a rock power chord function
        def play_rock_power_chord(notes, duration):
            """Play chord with rock-style power chord emphasis"""
            # For rock, emphasize root and fifth, add bass
            if len(notes) >= 3:
                power_chord = [notes[0], notes[2]]  # Root and fifth
                bass_note = notes[0] - 12  # One octave down
            else:
                power_chord = notes
                bass_note = notes[0] - 12
            
            # Play bass note first
            outport.send(mido.Message('note_on', note=bass_note, velocity=90))
            
            # Play power chord with palm muting pattern
            base_duration = duration / 16
            
            # Strong downbeat
            for note in power_chord:
                outport.send(mido.Message('note_on', note=note, velocity=85))
            time.sleep(base_duration * 1.5)
            
            # Palm mute (briefly release then hit again softer)
            for note in power_chord:
                outport.send(mido.Message('note_off', note=note, velocity=85))
            time.sleep(base_duration * 0.5)
            
            # Lighter strums
            for i in range(3):
                vel = 70 if i % 2 == 0 else 60  # Alternate accents
                for note in power_chord:
                    outport.send(mido.Message('note_on', note=note, velocity=vel))
                time.sleep(base_duration)
                for note in power_chord:
                    outport.send(mido.Message('note_off', note=note, velocity=vel))
                time.sleep(base_duration)
            
            # Release bass
            outport.send(mido.Message('note_off', note=bass_note, velocity=90))
        
        for i, chord_name in enumerate(chord_progression):
            notes = get_chord_notes(chord_name)
            
            if not notes:
                print(f"Chord '{chord_name}' not recognized.")
                time.sleep(duration)
                continue
                
            print(f"Playing: {chord_name}")
            
            # Choose playing style based on parameter
            if style == "mixed":
                current_style = random.choice(["normal", "rhythm", "arpeggio", "bassline"])
            elif style == "rock":
                # Special handling for rock style
                play_rock_power_chord(notes, duration)
                continue  # Skip the rest of the loop
            else:
                current_style = style
                
            # Play according to the chosen style
            if current_style == "arpeggio":
                pattern = random.choice(["up", "down", "updown"])
                play_arpeggiated_chord(outport, notes, duration, pattern)
            elif current_style == "rhythm":
                pattern = "rock" if style == "rock" else random.choice(["basic", "waltz", "rock", "ballad"])
                play_rhythmic_chord(outport, notes, duration, pattern)
            elif current_style == "bassline":
                play_chord_with_bassline(outport, notes, duration)
            else:
                # In the "normal" style section
                for note in notes:
                    outport.send(mido.Message('note_on', note=note, velocity=70))
                # Hold 90% of the duration
                time.sleep(duration * 0.9)  
                for note in notes:
                    outport.send(mido.Message('note_off', note=note, velocity=70))
                # Very short gap (10%) between chords
                time.sleep(duration * 0.1)
                
    except ImportError:
        print("mido package not installed.")
    except Exception as e:
        print(f"Error playing MIDI: {e}")


def list_genres():
    _, _, _, genre_to_id, id_to_genre, _ = load_resources()
    print("\nAvailable genres and their IDs:")
    for genre_id, genre_name in sorted(id_to_genre.items()):
        print(f"  {genre_id}: {genre_name}")
    print()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate chord progressions using a pre-trained model")
    parser.add_argument("--initial", type=str, default="G D", help="Initial chord sequence (space-separated)")
    parser.add_argument("--genre", type=int, default=10, help="Genre ID for generation")
    parser.add_argument("--num_chords", type=int, default=20, help="Number of chords to generate")
    parser.add_argument("--temperature", type=float, default=1.1, help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top predictions to sample from")
    parser.add_argument("--play", action="store_true", help="Play the chord progression as MIDI")
    parser.add_argument("--list_genres", action="store_true", help="List all available genres")
    parser.add_argument("--style", type=str, default="mixed", 
                    choices=["mixed", "normal", "rock", "ballad", "arpeggio", "bassline"], 
                    help="Style for chord playback")
    
    args = parser.parse_args()
    
    # Load resources
    model, chord_to_id, id_to_chord, genre_to_id, id_to_genre, device = load_resources()
    
    # Parse initial sequence
    initial_sequence = args.initial.split()
    
    # Generate chord progression
    generated_chords = generate_chords(
        model=model,
        initial_sequence=initial_sequence,
        genre=args.genre,
        device=device,
        num_chords=args.num_chords,
        temperature=args.temperature,
        top_k=args.top_k,
        chord_to_id=chord_to_id,
        id_to_chord=id_to_chord
    )
    if args.list_genres:
        list_genres()
        return
    
    # Print results
    print(f"Genre: {id_to_genre.get(args.genre, 'Unknown')}")
    print("\nGenerated chord progression:")
    print(" -> ".join(initial_sequence + generated_chords))
    
    # Play the progression if requested
    if args.play:
        print("\nPlaying chord progression...")
        play_chord_progression(initial_sequence + generated_chords, style=args.style)
if __name__ == "__main__":
    main()