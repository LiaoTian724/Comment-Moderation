# Comment Moderation

A Chinese comment moderation project based on Natural Language Processing (NLP), focusing on binary classification of safe and offensive comments.

The project explores traditional machine learning approaches using TF-IDF feature extraction and Logistic Regression, with the goal of identifying offensive language while preserving reasonable criticism.

## 1. Project Objective
Build a Chinese text classification system that distinguishes between:

- **Safe:** Normal comments, reasonable negative reviews, and criticism.
- **Offensive:** Profanity, abusive personal insults, and discriminatory attacks.

Examples:

|     Comment  | Label |
|--------------|-------|
| 服务员态度很差 | Safe  |
| 这家餐厅真垃圾 | Safe  |
| 这个服务员是个废物 | Offensive |

The objective is to avoid both over-moderation and missed offensive content.

## 2. Dataset

This project uses the [COLD Dataset](https://github.com/thu-coai/COLDataset), a benchmark for Chinese Offensive Language Detection.

Original paper: [COLD: A Benchmark for Chinese Offensive Language Detection (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.796/).

Dataset statistics:

| Split | Samples |
| Train | 25,726 |
| Dev | 6,431 |
| Test | 5,323 |
| Total | 37,480 |

Binary labels:
- `0`: Safe
- `1`: Offensive

The test set also provides fine-grained annotations.

Dataset files are not included in this repository. Please obtain them from the official source and follow its license and citation requirements.


