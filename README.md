# Design and Implementation of a Multi-modal Framework for Disease Detection and Classification Using ML

This project presents the design and implementation of a multi-modal machine learning framework to improve disease detection and classification by integrating medical imaging and
clinical text data. The proposed framework employs deep learning models—ResNet50, DenseNet121, and EfficientNet—for analyzing CT and MRI scans, alongside language models
such as DistilBERT and BiomedBERT for processing diagnostic reports. The dataset was sourced from MedPix 2.0, a publicly available medical database containing paired images and textual case
descriptions. To address the class imbalance and enhance model generalization, various augmentation techniques were applied to both modalities. Multiple fusion strategies, including
early and late fusion, were explored to effectively combine visual and textual features. Model performance was evaluated using precision, recall, F1-score, and AUC-ROC metrics. The bestperforming
configuration—EfficientNet combined with Random Forest using weighted averaging late fusion—achieved an accuracy of 96.62% and F1-scores above 0.95. These results highlight
the potential of multi-modal learning in advancing AI-driven diagnostic systems.

## Text augmentation
The following are the three techniques that were employed to balance the classes:
- Synonym Replacements: The first text technique used was Synonym replacement, where we changed words with their synonyms to provide some balance to the dataset. Some
libraries, such as TextAttack and NLPAug, were used but didn't offer promising results as they changed the meaning behind the sentences drastically. Instead, we manually picked
widely used words and phrases found in the dataset and mapped them to their corresponding synonym to perform the balancing. Words such as "pain" were replaced
with "ache," "operation," "surgery," "swelling," "inflammation," and so on. Additionally, the patient's age was changed by adding or removing 1-2 years. This technique was able to
provide meaningful variations of data while preserving the context.
- Sentence Re-ordering: This technique involves changing the sequence of sentences within a text while maintaining the original meaning and logical flow. Fields that only included 1
sentence meant that re-ordering couldn't be used, so another technique was used, which was Active-Passive voice conversion. These sentences were rephrased to ensure the
structure of it was changed.
- Back Translation: This is a data augmentation technique used to improve diversity. This involved translating the text into French and then translating it back to English. This offered
a different variance of the original text while maintaining the medical meaning behind it. "Googletrans" was used, which is a Python library that provides an interface to the Google
Translate API.

## Image Augmentations:
The following are the two techniques that were employed to balance the classes further more:
- Contrast Adjustment: The first image augmentation is implemented using the PIL.ImageEnhance module. It increases the image's contrast by a factor of 1.5 (50%),
making dark areas darker and bright areas brighter. This is particularly useful for enhancing the visibility of subtle features in medical images, making certain visual features more
prominent, where differences in tissue density or pathology can be highlighted with better contrast. Applying contrast variation during training improves the model's robustness to
images captured under varying lighting conditions or device settings, ensuring it doesn't overly rely on consistent illumination to make predictions.
- Grid Distortion: it is applied using albumentations' GridDistortion class. This method divides the image into a grid and applies random distortions to each section, simulating
non-linear warping. Such distortions mimic real-world imaging artifacts, patient movement, or tissue deformations, encouraging the model to generalize better to spatially
inconsistent inputs. By introducing controlled geometric variability, this augmentation helps the model become less sensitive to exact spatial arrangements, promoting a focus on
structural patterns rather than precise pixel locations.

The following transformations were implemented to increase the number of records:
- Clockwise rotation: Images were rotated 30 degrees to the right.
- Anti-clockwise rotation: Images were rotated 30 degrees to the left

## Data Preprocessing
Images:
- Convert to RGB 3-channel tensors of shape [3, H, W]
- Resize to 224x224 pixels
- Normalizing pixel values from 0-255 to 0–1

Text:
- Cleaning: missing fields are converted to empty strings Unicode escape characters removed
- Concatenation: combine multiple text fields into one string:
- TF-IDF Vectorization: maximum of 3000 features, english stop words removal (“the”, “is” and “and”)

## Unimodal models designed:
Text:
- BioMedBERT
- SVM
- Random Forest
- Logistic Regression

Image:
- ResNet50
- DenseNet121
- EfficientNet-B0

## Multi-modal models designed:
- EfficientNet+TF-IDF Vectorizer(Intermediate Fusion with Cross-Modal Attention)
- EfficientNet+BiomedBert (Weighted Late Fusion)
- EfficientNet+BiomedBert (Average Late Fusion)
- EfficientNet+Random Forest(Averaging Late Fusion)
- EfficientNet+Random Forest(Averaging Late Fusion) Without Miscellaneous
- Resnet50+DistilBert

## Web Application
To make the diagnostic system accessible and interactive, a web application was developed and deployed. The application integrates three underlying models to support flexible input options:
a Random Forest model for text-only inputs, an EfficientNet model for image-only inputs, and a late fusion model combining both modalities when text and image are provided together. This
dynamic model selection ensures that the system remains functional and accurate regardless of the type of input received from the user.

The backend was built using Flask and deployed on Render, handling all inference logic and model orchestration. The frontend was developed with React and TypeScript, providing a
responsive and user-friendly interface, and is hosted on Vercel. This modular design allows seamless communication between the frontend and backend, enabling real-time predictions and a
smooth user experience. The deployed application serves as a practical demonstration of the proposed multimodal diagnostic framework and its adaptability to real-world clinical scenarios.
