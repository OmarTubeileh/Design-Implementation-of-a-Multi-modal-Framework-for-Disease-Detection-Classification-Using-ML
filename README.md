# Design and Implementation of a Multi-modal Framework for Disease Detection and Classification Using ML

This project presents the design and implementation of a multi-modal machine learning framework to improve disease detection and classification by integrating medical imaging and
clinical text data. The proposed framework employs deep learning models—ResNet50, DenseNet121, and EfficientNet—for analyzing CT and MRI scans, alongside language models
such as DistilBERT and BiomedBERT for processing diagnostic reports. The dataset was sourced from MedPix 2.0, a publicly available medical database containing paired images and textual case
descriptions. To address the class imbalance and enhance model generalization, various augmentation techniques were applied to both modalities. Multiple fusion strategies, including
early and late fusion, were explored to effectively combine visual and textual features. Model performance was evaluated using precision, recall, F1-score, and AUC-ROC metrics. The bestperforming
configuration—EfficientNet combined with Random Forest using weighted averaging late fusion—achieved an accuracy of 96.62% and F1-scores above 0.95. These results highlight
the potential of multi-modal learning in advancing AI-driven diagnostic systems.

## Text augmentation
The following are the 3 techniques that were employed to balance the classes:
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

