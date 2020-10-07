''' Video Cassette

    This is a module for storing any file or other binary content in video
    form. This allows essentially infinite storage if the resulting files are
    uploaded to YouTube.

    Inspired by this reddit post: https://www.reddit.com/r/Python/comments/j620fv/i_made_a_program_that_gives_me_infinite_storage/

    (c) Jake Ledoux, 2020
    contactjakeledoux@gmail.com
'''

import math
import numpy as np
import os
from typing import List, Tuple, Union


class Tape(object):
    ''' Tape objects handle the conversion between videos, binary tapes, and
        the original files.
    '''
    HEADER_FILENAME_LEN: int = 32
    HEADER_BYTELEN_LEN: int = 16
    HEADER_LEN: int = HEADER_FILENAME_LEN + HEADER_BYTELEN_LEN
    
    def __repr__(self):
        return f'<Tape \'{self.file_name}\' {self.file_len} bytes>'

    def __init__(self, file_path: str, file_name: str, file_bytes: bytes):
        ''' You really shouldn't initialize this class directly. Instead make
            use of the from_* static methods.

            :param file_path: The directory of the source file.
            :param file_name: The name and extension of the source file.
            :param file_bytes: The binary content of the source file.
        '''
        self.file_path = file_path
        self.file_name = file_name
        self.file_bytes = file_bytes

    def write_tape_file(self, filename: Union[str, None] = None):
        ''' Write Tape header and contents to binary file.

            :param filename: The filename to write to, defaults to the source
                file's name with the extension '.tape'.
        '''
        if filename is None:
            filename = os.path.splitext(self.file_name)[0] + '.tape'
        with open(filename, 'wb') as f:
            f.write(self.header + self.file_bytes)

    def write_file(self, path_override: Union[str, None] = None):
        ''' Write Tape contents to file.

            :param path_override: The path + filename to write to, defaults to
                the original directory and name.
        '''
        # User header path if not overridden
        if path_override is None:
            file_location = os.path.join(self.file_path, self.file_name)
        # Use override path
        else:
            file_location = path_override

        with open(file_location, 'wb') as f:
            f.write(self.file_bytes)

    def write_video(self, filename: Union[str, None] = None,
                    resolution: Tuple[int, int] = (1920, 1080),
                    cols: int = 16, rows: int = 9):
        ''' Writes Tape to MP4 file.

            :param filename: The filename to write to, defaults to the source
                file's name with the extension '.mp4'.
            :param resolution: The resolution of the final video. Must match
                aspect ratio of `cols`, `rows`. Defaults to 1920x1080.
            :param cols: The number of data blocks to have per row,
                defaults to 16.
            :param rows: The number of data blocks to have per column,
                defaults to 9.
        '''
        if filename is None:
            filename = os.path.splitext(self.file_name)[0] + '.mp4'
        write_frames(filename, bytes_to_frames(self.bytes, cols, rows), resolution)

    @property
    def bytes(self) -> bytes:
        ''' Get Tape header and contents in bytes form.
            :returns: Concatenated bytes of header and contents respectively.
        '''
        return self.header + self.file_bytes

    @property
    def file_len(self) -> int:
        ''' Get length of file contents in number of bytes.
            :returns: Number of bytes in the source file.
        '''
        return len(self.file_bytes)

    @property
    def header(self) -> bytes:
        ''' Get header for this tape. This is used to inform the library on how
            to unpack this file from binary form.
            :returns: 48-byte ASCII bytestring header.
        '''
        file_name, extension = os.path.splitext(self.file_name)
        header_file_name = f'{file_name[:32-len(extension)] + extension:>32}'
        header_byte_len = f'{str(self.file_len)[:16]:>16}'
        header = bytes(header_file_name + header_byte_len, encoding='ascii')
        assert len(header) == Tape.HEADER_LEN
        return header

    @property
    def ext(self) -> str:
        ''' Get extension of source file.
            :returns: Extension of source file including the dot.
        '''
        return os.path.splitext(self.file_name)[1]
    
    @staticmethod
    def from_file(file: str) -> 'Tape':
        ''' Creates a Tape object from any file.
            :param file: The path + filename of the file to be loaded.
            :returns: Tape object containing file metadata and contents.
        '''
        file_path, file_name = os.path.split(os.path.abspath(file))
        file_bytes = read_bytes(os.path.join(file_path, file_name))
        return Tape(file_path, file_name, file_bytes)
    
    @staticmethod
    def from_bytes(tape_bytes: bytes) -> 'Tape':
        ''' Creates a Tape object from Tape bytes, including the header.
            :param tape_bytes: The binary content of a Tape in bytes form.
            :returns: Tape object containing file metadata and contents.
        '''
        # Load header and contents from bytes
        file_name = tape_bytes[:Tape.HEADER_FILENAME_LEN].decode('ascii').strip()
        byte_len = int(tape_bytes[Tape.HEADER_FILENAME_LEN:Tape.HEADER_LEN].decode('ascii').strip())
        file_bytes = tape_bytes[Tape.HEADER_LEN:]
        # Make sure contents are of appropriate length
        assert len(file_bytes) >= byte_len
        # Trim padded contents
        file_bytes = file_bytes[:byte_len]

        file_path = os.path.abspath('.')
        return Tape(file_path, file_name, file_bytes)

    @staticmethod
    def from_video(filename: str, cols: int = 16, rows: int = 9) -> 'Tape':
        ''' Creates a Tape object from a video file.
            :param filename: The path + filename of the source video.
            :param cols: The number of data cols in the video, defaults to 16.
            :param rows: The number of data rows in the video, defaults to 9.
            :returns: Tape object recovered from video.
        '''
        return Tape.from_bytes(frames_to_bytes(load_frames(filename, cols, rows)))


def read_bytes(filename: str) -> bytes:
    ''' Read file binary contents of file.
        :param filename: The path + filename of the file to read.
        :returns: Binary contents of file.
    '''
    with open(filename, 'rb') as f:
        file_bytes = f.read()
    return file_bytes


def write_bytes(filename: str, file_bytes: bytes):
    ''' Write binary contents to file.
        :param filename: The path + filename to write to.
        :param file_bytes: The binary data to write.
    '''
    with open(filename, 'wb') as f:
        f.write(file_bytes)
    
    
def bytes_to_bit_list(raw_bytes: bytes) -> List[int]:
    ''' Converts bytes to a list of bits.
        :param raw_bytes: The bytes to convert.
        :returns: List of bits as integers.
    '''
    bit_list = [int(c) 
                for byte in raw_bytes
                for c in f'{byte:08b}']
    return bit_list


def bit_list_to_bytes(bit_list: List[int]) -> bytes:
    ''' Converts a list of bits to a bytes object.
        :param bit_list: List of bits as integers to convert.
        :returns: Resulting bytes object.
    '''
    if len(bit_list) % 8 == 0:
        byte_list = list()
        for i in range(0, len(bit_list), 8):
            bit_slice = bit_list[i:i + 8]
            byte_list.append(int(''.join((str(n) for n in bit_slice)), 2))
        return bytes(byte_list)
    else:
        raise Exception('Odd number of bits.')


def bytes_to_frames(raw_bytes: bytes, cols: int = 16, rows: int = 9) -> np.ndarray:
    ''' Converts bytes to video form.
        :param raw_bytes: bytes object to convert.
        :param cols: Number of data blocks per row, defaults to 16.
        :param row: Number of data blocks per column, defaults to 9.
        :returns: Array of frames as 3D Numpy array. Dimensions will be 
            (nframes, rows, cols).
    '''
    bits = np.array(bytes_to_bit_list(raw_bytes), dtype=np.uint8)
    
    # Pad and reshape to fit into frames of size (cols, rows)
    bits.resize((math.ceil(float(len(bits)) / (cols * rows)), rows, cols))
    return bits


def frames_to_bytes(frames: np.ndarray) -> bytes:
    ''' Converts video frames to bytes.
        :param frames: Array of frames as 3D Numpy Array. Dimensions of 
            (nframes, rows, cols).
        :returns: Bytes converted from frames.
    '''
    linear_bits = list(np.ndarray.flatten(frames))
    return bit_list_to_bytes(linear_bits)


def write_frames(filename: str, frames: np.ndarray,
                 resolution: Tuple[int, int] = (1920, 1080),
                 framerate: int = 30, use_disk: bool = True):
    ''' Upscales frames and writes them to MP4 file.
        :param filename: The path + filename to write video to. (Should end in
            .mp4 extension)
        :param frames: Array of frames as 3D Numpy Array. Dimensions of 
            (nframes, rows, cols).
        :param resolution: The resolution to upscale frames to. Must match
            original frame aspect ratio. Defaults to 1920x1080.
        :param framerate: The framerate of the resulting video in frames per
            second. This has no effect on the packing or upacking processes,
            it only changes the duration of the video when viewed externally.
            It's recommended to set this to something common like 24, 30, or
            60. Anything above 60 frames per second will break YouTube
            compatibility. Defaults to 30 FPS.
        :param use_disk: If true, resized images will be temporarily written to
            disk until final video is created. Otherwise, all images will be
            kept in memory until completion. While keeping all images in memory
            is much faster, this is bad if you're using a high resolution
            and/or lots of frames as you will quickly run out of memory.
    '''
    import cv2
    import ffmpeg
    import shutil
    import tempfile

    width, height = resolution

    # Upscale frames
    new_frames = list()
    tmp_dir = os.path.join(tempfile.gettempdir(), 'videocassette')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    for idx, frame in enumerate(frames):
        if height // frame.shape[0] == width // frame.shape[1]:
            print(f'Pre-processing frame: {idx + 1:>{len(str(frames.shape[0]))}}/{frames.shape[0]}')
            new_frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_NEAREST)
            if use_disk:
                cv2.imwrite(os.path.join(tmp_dir, f'{idx + 1:0>{len(str(frames.shape[0]))}}.jpg'), new_frame * 255)
            else:
                new_frames.append(new_frame)
        else:
            raise Exception('Resolution aspect ratio is not equal to frame aspect ratio.')
        
    # Write frames to video
    new_frames = np.array(new_frames)

    if use_disk:
        process = (
            ffmpeg.input(os.path.join(tmp_dir, f'%0{len(str(frames.shape[0]))}d.jpg'), pix_fmt='gray', s=f'{width}x{height}', r=framerate)
                .output(filename, pix_fmt='yuv420p', vcodec='libx264')
                .overwrite_output()
                .run()
        )
        shutil.rmtree(tmp_dir)
    else:
        process = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{width}x{height}', r=framerate)
                .output(filename, pix_fmt='yuv420p', vcodec='libx264')
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
        for frame in new_frames:
            process.stdin.write(
                frame.astype(np.uint8).tobytes()
            )
        process.stdin.close()
        process.wait()


def load_frames(filename: str, cols: int = 16, rows: int = 9) -> np.ndarray:
    ''' Load video file into frames as Numpy array.
        :param filename: The path + name of the video file to load.
        :param cols: The number of data blocks per row in the original video,
            defaults to 16.
        :param rows: The number of data blocks per column in the original video,
            defaults to 9.
        :returns: Array of frames as 3D Numpy array. Dimensions will be 
            (nframes, rows, cols).
    '''
    # Read video file
    import cv2
    vidcap = cv2.VideoCapture(filename)
    read_frames = list()
    frame_count = 0
    while True:
        success, image = vidcap.read()
        if success:
            frame_count += 1
            print(f'Reading frame: {frame_count + 1:>6}/?')
            small_img = cv2.resize(image, dsize=(cols, rows), interpolation=cv2.INTER_CUBIC)
            gray_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
            read_frames.append((gray_img > 128).astype(np.uint8))
        else:
            break

    return np.array(read_frames)


if __name__ == '__main__':
    import sys
    source = Tape.from_file(' '.join(sys.argv[1:]))
    source.write_video(resolution=(1280, 720), cols=320, rows=180)