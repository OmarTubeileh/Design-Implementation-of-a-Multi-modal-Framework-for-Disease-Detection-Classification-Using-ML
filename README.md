# Design and Implementation of a Multi-modal Framework for Disease Detection and Classification Using ML

This project presents the design and implementation of a multi-modal machine learning framework to improve disease detection and classification by integrating medical imaging and
clinical text data. The proposed framework employs deep learning models—ResNet50, DenseNet121, and EfficientNet—for analyzing CT and MRI scans, alongside language models
such as DistilBERT and BiomedBERT for processing diagnostic reports. The dataset was sourced from MedPix 2.0, a publicly available medical database containing paired images and textual case
descriptions. To address the class imbalance and enhance model generalization, various augmentation techniques were applied to both modalities. Multiple fusion strategies, including
early and late fusion, were explored to effectively combine visual and textual features. Model performance was evaluated using precision, recall, F1-score, and AUC-ROC metrics. The bestperforming
configuration—EfficientNet combined with Random Forest using weighted averaging late fusion—achieved an accuracy of 96.62% and F1-scores above 0.95. These results highlight
the potential of multi-modal learning in advancing AI-driven diagnostic systems.
