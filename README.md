# Video Cassette

This is a module for storing any file or other binary content in video
form. This allows essentially infinite storage if the resulting files are
uploaded to YouTube.

Inspired by [this reddit post](https://www.reddit.com/r/Python/comments/j620fv/i_made_a_program_that_gives_me_infinite_storage/).

As a concept it works great, however in order to actually provide any sort
of practical use it needs to be faster by at least an order of magnitude.
Thanks to video compression, I've found I need around 8 pixels per bit for
a truly reliable transfer. Currently that means the encoding takes ages
for any files larger than a few megabytes. I've got a couple optimization
ideas at the moment, but if anyone sees anything obvious I'd love it if
you'd create an issue or even a pull request.

## Example usage

### Write a file to video

``` Python
from video_cassette import Tape

mike = Tape.from_file('michael_jackson.jpg')
# `write_video()` uses the original name by default, just changing the file
# extension to '.mp4'.
mike.write_video(resolution=(1280, 720), cols=320, rows=180)
```

### Read a file from video

``` Python
from video_cassette import Tape

# You must specify cols+rows when loading unless you encoded with the default
# of 16x9.
mike = Tape.from_video('mike.mp4', cols=320, rows=180)
# Original filename is stored within the video, calling `write_file()` uses
# that filename be default. This can be overridden.
mike.write_file()
```

### Using the header and serializing functionality without video encoding

``` Python
import fictional_tx_library as tx
from video_cassette import Tape

secrets = Tape.from_file('my_secrets.txt')
tx.transmit(secrets.bytes)
```
