#!/usr/bin/python3.6
""" knn_classification.py
    This script tests the KNN-based classification method.
    Author: Alan Rocha González
    Email: alan.rocha@udem.edu
    Institution: Universidad de Monterrey
    Created: Thursday 14th, 2020
"""
# Import Knn class 
from knn import Knn

def main():
    # Create a Knn object
    knn_classifier = Knn("diabetes.csv")

    # Set k as 20
    knn_classifier.set_k(5)
    knn_classifier.get_confusion_matrix()

main()