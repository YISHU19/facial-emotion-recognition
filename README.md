# Facial Emotion Recognition

A convolutional neural network that recognises seven facial expressions
(angry, disgust, fear, happy, neutral, sad, surprise) from 48x48 grayscale
faces, trained on FER-2013, plus a script that finds faces in a webcam feed,
video or photo and labels each one.

## Results

Test set: the 7,178 FER-2013 test images, never used for training or model
selection (a stratified 10% of the training set is held out for validation).

| Model | Test accuracy |
|---|---|
| Baseline: my original 4-layer CNN, 10 epochs | 59.2% |
| Improved: VGG-style CNN with batch norm and augmentation, 35 epochs | **61.2%** |

For reference, human accuracy on FER-2013 is estimated at about 65%, and
the images are small, low-contrast and sometimes mislabelled.

| Emotion | Precision | Recall | Test images |
|---|---|---|---|
| Happy | 0.86 | 0.84 | 1,774 |
| Surprise | 0.75 | 0.75 | 831 |
| Neutral | 0.56 | 0.64 | 1,233 |
| Angry | 0.50 | 0.58 | 958 |
| Sad | 0.46 | 0.55 | 1,247 |
| Fear | 0.43 | 0.26 | 1,024 |
| Disgust | 0.00 | 0.00 | 111 |

Where it goes wrong:
- **Disgust is never predicted.** It has only 436 training images (1.5% of
  the data), and 70% of disgust faces are called angry.
- **Fear is the hardest common class**: 30% of fear faces are called sad.
- **Sad and neutral blur into each other** (21% confused in each direction).

The confusion matrix, training curves and the most confident mistakes are in
the executed notebook.

## How the improved model differs

- Four VGG-style blocks (two 3x3 convolutions each) with batch normalisation
- Augmentation: horizontal flips, small rotations, zooms and shifts
- Global average pooling instead of a large flatten + dense layer
- Learning rate halved when validation loss plateaus, early stopping on
  validation loss with the best weights restored

## Run it

```bash
pip install -r requirements.txt
python download_data.py          # FER-2013 from a Hugging Face mirror into data/
```

**Use the trained model** (`emotion_cnn.keras` is in the repo):

```bash
python detect_emotions.py                        # webcam
python detect_emotions.py --source clip.mp4      # video
python detect_emotions.py --source photo.jpg --save labelled.jpg
```

Press `q` to quit the webcam or video window. Faces are found with OpenCV's
Haar cascade, cropped to 48x48 grayscale like the training data and labelled
with the predicted emotion and its confidence.

**Retrain:** run `emotion_recognition.ipynb` top to bottom. It works on any
Keras 3 backend (TensorFlow, PyTorch or JAX); on a laptop CPU the improved
model takes about 2.5 hours.

## Files

```
emotion_recognition.ipynb   data, baseline, improved CNN, evaluation (executed)
emotion_cnn.keras           trained improved model
detect_emotions.py          webcam / video / image detector
download_data.py            downloads FER-2013
```

## Limitations

- **Class imbalance**: disgust needs class weights, oversampling or merging
  with angry to be usable.
- **FER-2013 is posed and cropped**: real webcam faces at angles, in poor
  light or partly covered are harder than the test set.
- **Haar cascades miss turned and tilted faces**; a CNN face detector would
  find more of them.
- Expressions are not feelings: the model labels what a face shows, not what
  a person feels.
