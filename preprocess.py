import configuration as config
import argparse
import pathlib
import time
from Midi_Parser import MIDI_parser


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--midi_dir', type=str, default='dataset', help='Directory where the midi files are stored')
    arg_parser.add_argument('-np', '--npz_dir', type=str, default='npz_files',help='Directory where the npz files will be saved')

    args = arg_parser.parse_args()

    if pathlib.Path(args.npz_dir).exists():
        assert pathlib.Path(args.npz_dir).is_dir()
    else:
        pathlib.Path(args.npz_dir).mkdir(parents=True, exist_ok=True)

    midi_path = []
    midi_path = list(map(lambda x: str(x),pathlib.Path(args.midi_dir).rglob(config.midi_type)))

    assert len(midi_path) > 0
    print(f'found {len(midi_path)} midi files ')

    midi_parser = MIDI_parser.build_from_config(config)

    start_time = time.time()

    print('preprocess..')
    midi_parser.preprocess_dataset(src_filenames=midi_path, dst_dir=args.npz_dir, batch_size=config.batch_size)
    print(f'Created dataset with {len(midi_path)} files')
    print("Program took " + str(time.time() - start_time) + " sec to run")

