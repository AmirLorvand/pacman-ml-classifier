# Pacman ML Classifier

This repository contains my custom machine learning classifier implementation for a Pacman-based coursework task in the **Machine Learning** module at King’s College London.

The coursework required implementing a classifier inside `classifier.py` to control Pacman’s movement using feature vectors and training data from `good-moves.txt`. The classifier is designed to work inside the UC Berkeley/KCL Pacman environment through the provided `ClassifierAgent`.

---

## Project Timeline

* **Originally completed:** 2025
* **Published on GitHub:** 2026
* **Context:** MSc Artificial Intelligence Machine Learning coursework project

This repository has been cleaned and documented for portfolio purposes.

---

## Project Overview

The aim of the coursework was to train a classifier that predicts Pacman’s next move from a feature vector describing the local game state.

The Pacman framework provides binary feature vectors representing nearby walls, food, and ghosts. The classifier uses these features to predict an action such as moving north, south, east, or west.

This repository contains only my implemented classifier file, not the full Pacman framework.

---

## Classifier Implemented

The implementation uses a custom **Multi-Layer Perceptron** built from scratch with NumPy.

Main components include:

* input feature processing
* one hidden layer
* ReLU activation
* softmax output
* cross-entropy loss
* backpropagation
* gradient descent training
* prediction of Pacman movement actions

---

## Technologies Used

* Python
* NumPy
* Machine Learning
* Neural Networks
* Multi-Layer Perceptron
* Classification
* Pacman AI environment

---

## Repository Structure

```text
.
├── classifier.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Use

This repository does **not** include the full Pacman framework.

To run the classifier:

1. Place `classifier.py` inside the provided Pacman coursework environment.
2. Ensure the environment contains:

   * `pacman.py`
   * `classifierAgents.py`
   * `api.py`
   * `good-moves.txt`
3. Run the Pacman agent using:

```bash
python3 pacman.py --pacman ClassifierAgent
```

Depending on the coursework environment, the shorter option may also work:

```bash
python3 pacman.py -p ClassifierAgent
```

---

## Dataset

The classifier is trained using `good-moves.txt`, which is part of the Pacman coursework environment.

This file contains feature vectors and target movement labels collected from previous Pacman gameplay. The dataset is not included in this repository because it belongs to the coursework framework.

---

## Attribution

The Pacman environment is based on the UC Berkeley AI Pacman projects and was adapted for the KCL coursework environment.

This repository contains only my own classifier implementation in `classifier.py`.

---

## What I Learned

Through this coursework project, I practised:

* implementing a neural network classifier from scratch
* using NumPy for matrix operations
* applying ReLU activation and softmax classification
* implementing cross-entropy loss
* implementing backpropagation manually
* training a classifier using labelled feature vectors
* integrating a classifier into a decision-making agent
* working within a constrained existing software environment

---

## Future Improvements

Possible improvements include:

* adding validation accuracy tracking
* adding confusion matrix analysis
* tuning the hidden layer size and learning rate
* comparing the MLP with k-nearest neighbours or Naive Bayes
* adding clearer logging of training loss
* improving reproducibility with configurable random seeds
* creating a standalone demo script outside the Pacman framework

---

## Author

**Amir Lorvand**

MSc Artificial Intelligence student at King’s College London with interests in machine learning, neural networks, AI systems, and decision-making.
