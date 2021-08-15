from Midi_Parser import MIDI_parser
from transformerXL import Music_transformer,generate
import configuration as config
import numpy as np
import pathlib
import time
import argparse
import os


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('n_songs_gen', type=int,
                            help='Number of files to generate')
    arg_parser.add_argument('-l', '--gen_len', type=int, default=300,
                            help='Length of the generated midis (in midi messages)')
    arg_parser.add_argument('checkpoint_path', type=str,
                            help='Path to the saved weights')
    arg_parser.add_argument('-np', '--npz_dir', type=str, default='npz_files',
                            help='Directory with the npz files')
    arg_parser.add_argument('-o', '--dst_dir', type=str, default='generated_midis',
                            help='Directory where the generated midi files will be stored')

    args = arg_parser.parse_args()

    start_time = time.time()
    print("start generation")
    midi_filenames = [str(i) for i in range(1, args.n_songs_gen + 1)]
    midi_filenames = [f + '.midi' for f in midi_filenames]
    midi_filenames = [os.path.join(args.dst_dir, f) for f in midi_filenames]

    #preperation of npz file
    npz_filenames = list(pathlib.Path(args.npz_dir).rglob('*.npz'))
    filenames_sample = np.random.choice(npz_filenames, args.n_songs_gen, replace=False)

    midi_parser = MIDI_parser.build_from_config(config)
    model, _ = Music_transformer.build_from_config(config=config, checkpoint_path=args.checkpoint_path)
    midi_list, _, _, _ = generate(model=model, model_seq_len=config.seq_len, mem_len=config.mem_len, max_gen_len=args.gen_len,
                                                            parser=midi_parser, filenames_npz=filenames_sample)

    for midi, filename in zip(midi_list, midi_filenames):
        midi.save(filename)

    print ("My program took "+ str(time.time() - start_time)+ " to run")
