The target and interference folders are meant to have many gigs of flac files but microsoft doesn't like that.
So, links to download audio for those folders are:

Target: https://www.openslr.org/12 - I chose the ```train-clean-100.tar.gz [6.3G]   (training set of 100 hours "clean" speech )```

Interference: https://www.openslr.org/17

Download these (large) files and extrac the files you want to be the target (at broadside 90 degrees) into the target folder. I have it set so that github ignores what you put in that folder.

Extract your interference into the interference folder. Same deal with above, but this audio will be simulated to be coming from an angle (40 deg). 

The whole thing is in an anechoic chamber with two miocrophones spaced 8cm apart.

The models end up being approximately twelve megabytes which is a suitable sized model for a mid-range phone.

I have been experimenting with different numbers of hours of speech.

**You will need to install Anaconda to run this stuff** - https://www.anaconda.com/docs/getting-started/anaconda/install#windows-installation

**In order to train, this has only been tested with nVidia GPUs so far. Not sure how to get it working with AMD cards yet**


After you have installed Anaconda and are in an anaconda terminal, run this command:

```conda create -n beamform python=3.10 -y```

then:

```conda activate beamform```

Once that's done, inside your conda session run

```pip install matplotlib numpy pyroomacoustics pesq pystol pystoi "tensorflow[and-cuda]" librosa soundfile wavinfo```

Then you can run the testing using

```python testing.py```

Or training with 

```python train_beamformer.py```

But make sure you make some changes, especially to the model name! Don't overwrite the old one!
