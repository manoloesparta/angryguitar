# Angry Guitar

> Fake Distorion

## About

This project emulates the guitar distortion effect via neural networks. The specifcations are the following:

* ESP LTD James Hetfield Guitar
* Behringer UM2 Interface
* Sample rate of 44100
* Bit Depth of 32 bit float
* Garage Band DAW
* Cool Jazz Combo as clean
* Classic Drive as distorted

## Prerequisites

```bash
python 3.7 (or newer)
pip
virtualenv
```

## Setup

Get the clean.wav and distorted.wav files in order to create the dataset

```bash
./setup
virtualenv .env -p python3
source .env/bin/activate
pip install -r requirements.txt
python data.py
```

## Running

```bash
source .env/bin/activate
python main.py
```

## License

This project is under the MIT License
