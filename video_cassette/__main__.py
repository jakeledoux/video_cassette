import argparse
from video_cassette import enforce_extension, str_to_vector2, Tape

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='video_cassette',
        description='Video Cassette can encode and decode files from video.')

    parser.add_argument('mode', type=str,
                        help='Either encode or decode')
    parser.add_argument('file', type=str,
                        help='Input file.')
    parser.add_argument('--out', type=str,
                        help='Where to store the output file.')
    parser.add_argument('--res', type=str_to_vector2,
                        help='Resolution of video output as (W,H). Must match \
                        aspect ratio of data blocks. Defaults to (1280, 720)')
    parser.add_argument('--blocks', type=str_to_vector2,
                        help='Number of data blocks as (C,R). Defaults to \
                        (640, 360).')
    parser.add_argument('--use-disk', action='store_true',
                        help='Will write frames to disk temporarily instead \
                              of keeping everything in memory. This will \
                              take much longer and use of disk space, but is \
                              recommended for large files so you don\'t run \
                              out of memory during the encoding process.')

    args = parser.parse_args()

    cols, rows = args.blocks or (640, 360)

    if os.path.exists(args.file):
        if args.mode.lower() == 'encode':
            print('Loading file contents into memory...')
            tape = Tape.from_file(args.file)
            print('Done. Beginning video encoding...')
            tape.write_video(enforce_extension(args.out, ('.mp4', '.mkv')),
                             resolution=args.res or (1280, 720),
                             cols=cols, rows=rows, use_disk=args.use_disk)
            print(f"'{args.file}' successfully encoded to '{args.out or enforce_extension(tape.file_name, ('.mp4',))}'.")
        elif args.mode.lower() == 'decode':
            tape = Tape.from_video(args.file, cols=cols, rows=rows)
            tape.write_file()
            print(f"'{args.file}' successfully decoded to '{args.out or tape.file_name}'.")
        else:
            print(f'{args.mode} is not a valid mode (encode, decode)')
    else:
        print(f'Can not find file: {args.file}')

