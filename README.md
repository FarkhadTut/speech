# speech

This is speech recognition model based on deep learning neural networks written in Python using Pytorch Lightning.


Structure:
Network: Conv1D -> Dense -> LSTM -> Dense
Loss: CTCLoss


1). Install packages from 'requirements.txt' in your environment.
2). Link to kazakh speech dataset: https://www.openslr.org/102/
3). src directory should contain directories: 'dataset', 'saved_models'.
4). Place the dataset (or any, but change the alphabet content in 'dataloader.py') in 'dataset' directory.
5). Run training via 'speech.py'.
