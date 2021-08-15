from datetime import datetime
import configuration as config
from Midi_Parser import MIDI_parser
from transformerXL import Music_transformer,train_model
import tensorflow as tf
import argparse
import pathlib
import time
tf.config.experimental_run_functions_eagerly(False)  # TODO put in comment and see the effect

def add_To_doc(text):
    with open("documentation.txt", "a") as text_file:
        text_file.write(text + "\n")

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-np', '--npz_dir', type=str, default='npz_files',
                            help='Directory where the npz files are stored')
    arg_parser.add_argument('-c', '--checkpoint_dir', type=str, default='checkpoints',
                            help='Directory where the saved weights *will* be stored')
    arg_parser.add_argument('-w', '--weights_optimizer', type=str,
                            default=None, help='Path to saved model weights and optimizers')
    arg_parser.add_argument('-doc', '--documentation', type=str,default='documentation'
                            , help='document where all running result will store')
    arg_parser.add_argument('-d', '--documentation_save', type=bool, default=False
                            , help='save documentation about runing')
    args = arg_parser.parse_args()

    assert pathlib.Path(args.npz_dir).is_dir()
    if pathlib.Path(args.checkpoint_dir).exists():
        assert pathlib.Path(args.checkpoint_dir).is_dir()
    else:
        pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if not args.weights_optimizer is None:
        assert pathlib.Path(args.weights).is_file()


    midi_parser = MIDI_parser.build_from_config(config)

    print('load NPZ files(dataset)')
    dataset = midi_parser.get_tf_dataset(file_directory=args.npz_dir, batch_size=config.batch_size, n_samples=None)
    batches_per_epoch = tf.data.experimental.cardinality(dataset).numpy()

    #build model
    model, optimizer = Music_transformer.build_from_config(config=config)

    start_time = time.time()

    recommend,text=train_model(model, dataset, optimizer, batches_per_epoch,args.checkpoint_dir)
    print("My program took "+ str(time.time() - start_time)+ " to run")

    documentation=f'========================================================================\n'
    documentation+=f'{str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}\n'
    documentation+=f'Running with {config.n_epochs}  \n'
    documentation+=text
    documentation+=f'\nRunning took {time.time() - start_time}\n'
    documentation+=f'recommendation : {recommend}\n'
    if args.documentation_save:
        add_To_doc(documentation)
        print("documentation added successfully")
    print(f'recommendation : {recommend}')

