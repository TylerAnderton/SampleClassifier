# SampleClassifier

## Description

This project uses a Tensorflow convolution neural network to make predictions on user-provided one-shot drum samples. The classification model will attempt to identify whether the sample is a kick, snare, perc/tom, or cymbal. 

The UI was built using wxPython, which allows for the user to simply drag-and-drop files onto the interface in order to generate predictions.

With only a dataset of size n=2076, by using a 70/30 train/test split, I was able to achieve an accuracy of ~85% on my test dataset. I look forward to repeating this project in the future with a larger and more varied dataset, as I am confident that we would be able to see much more accurate results.

## Uncommon Dependencies

tensorflow-macos 2.10.0

tensorflow-metal 0.6.0

librosa 0.9.2

wxpython 4.2.0

## Operation

To run the program, simply navigate to the SampleClassifier directory in the terminal and run the command <code>pythonw main.py</code>. Attempting to run the file in a virtual or conda environment with just <code>python</code> may give you an error. 

Once the program is open, simply drag and drop any drum one-shot .wav file into the upper box, and the prediction will automatically appear in the lower box. Continue dragging and dropping new samples for as long as you want, and the program will automatically replace old predictions with new ones.

See the [demo video](demo_video.mov) for more information.

## Legal disclaimer

The samples used in the training of the model are purchased from [Splice.com](https://splice.com/) and are not licensed for distribution. They are included in this repository only for demonstration and should not be downloaded for use for any purpose.