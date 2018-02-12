#!/usr/bin/env bash

echo "Creating required directories"
mkdir dataset -p
mkdir dataset/features -p

echo "Installing all Python packages for the project"
sudo pip3 install -r requirements.txt

echo "Installing the English model for the Spacy library"
sudo python3 -m spacy download en