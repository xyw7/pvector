# p-vectors: A Parallel-Coupled TDNN/Transformer Network for Speaker Verification

## Installation
Our experiments are based on Speechbrain toolkit https://github.com/speechbrain/speechbrain. Make sure Speechbrain is installed correctly.
But theoretically, with the model structures, any speech-processing toolkit can implement it. (eg. speechbrain/wespeaker/FunASR...) 

## Model structure
TDNN_branch.py 
Transformer_branch.py 
p_vectors.py

## Usage
For SpeechBrain, (1) copy TDNN_branch.py, Transformer_branch.py and p_vectors.py into speechbrain/lobes/models/; 

(2) modify the 'embedding_model' and 'classifier' in speechbrain/recipes/VoxCeleb/SpeakerRec/hparams/train_ecapa_tdnn.yaml; 

(3) run speechbrain/recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py 

For other toolkits, replace the model.py by TDNN_branch.py/Transformer_branch.py/p_vectors.py. 

## Training strategy (24+6)
### training stage 1
In stage 1, Independently traine TDNN branch and Transformer branch for 24 epochs, respectively. 
### training stage 2
In stage 2, (1) transfer the pre-trained weights of TDNN and Transformer branches into the p-vector; 

(2) freeze those transfered weights (see the details in p_vectors.py );

(3) train the unforzen weights (EAL and the brand new classifier) for 6 epochs.
### training stage 3
In stage 3, unfreeze all the weights in p_vectors and train it for 6 epochs.
