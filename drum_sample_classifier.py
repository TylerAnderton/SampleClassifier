# import packages
import librosa
import tensorflow as tf
import numpy as np

# model_path = 'model_training/trained_hypermodel'

# create classifier class
class DrumClassifier:

    # define constants
    SAMPLE_RATE: int = 44100
    SAMPLE_LENGTH: int = 132300
    FRAME_LENGTH: int = 2**10
    FRAME_STEP: int = int(FRAME_LENGTH / 8)

    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

    def load_sample(self, filepath: str) -> tf.Tensor:
        audio, sample_rate = librosa.load(filepath, mono=True, sr=self.SAMPLE_RATE)
        return tf.convert_to_tensor(audio, dtype=tf.float32)
    
    def pad_sample(self, audio: tf.Tensor) -> tf.Tensor:
        audio = audio[:self.SAMPLE_LENGTH]

        zero_padding = tf.zeros([self.SAMPLE_LENGTH] - tf.shape(audio), dtype=tf.float32)
        return tf.concat([audio, zero_padding], axis=0)
    
    def normalize(self, audio: tf.Tensor) -> tf.Tensor:
        audio_max = tf.reduce_max(tf.abs(audio))
        scale_factor = 1 / audio_max
        return audio * scale_factor
    
    def apply_stft(self, audio: tf.Tensor) -> tf.Tensor:
        spectrogram = tf.signal.stft(audio, frame_length=self.FRAME_LENGTH, frame_step=self.FRAME_STEP) 
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2) #convolution neural net expects channels dimension
        return tf.expand_dims(spectrogram, axis=0) #model also expects batch size dimension
    
    def load_and_process(self, filepath: str) -> tf.Tensor:
        return self.apply_stft(self.normalize(self.pad_sample(self.load_sample(filepath))))
    
    def translate_prediction(self, prediction_array: list[float]) -> str:
        if np.argmax(prediction_array) == 0:
            return 'cymbal'
        elif np.argmax(prediction_array) == 1:
            return 'kick'
        elif np.argmax(prediction_array) == 2:
            return 'perc or tom'
        elif np.argmax(prediction_array) == 3:
            return 'snare'
        else:
            return None
        
    def make_prediction(self, file_path: str) -> str:
        return self.translate_prediction(self.model.predict(self.load_and_process(file_path)))
    
    def print_prediction(self, prediction: str) -> None:
        if prediction:
            print(f'Your sample is most likely a {prediction}!')
        else:
            print('The model could not make a prediction.')

    def make_and_return_prediction(self, file_path: str) -> str:
        prediction: str = self.make_prediction(file_path)
        if prediction:
            return f'Your sample is most likely a {prediction}!'
        else:
            return 'The model could not make a prediction.'