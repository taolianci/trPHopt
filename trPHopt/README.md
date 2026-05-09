# Structure-Guided Hybrid GCN-Transformer Framework for Enzyme Optimal pH Prediction

This tool performs pH prediction and classification for a single protein sequence using a pre-trained Two-Stage Predictor model. It integrates multiple feature modalities including ESMC representations, IUPred disorder predictions, and trRosetta structural features.

In the following, we take a enzyme sequence file as an example to show the prediction process. 
Here, the enzyme sequence file with uniprot_id 'A7RDD3'.
trPHopt uses the following dependencies:
* PyTorch  2.3.0
* Python  3.12
* CUDA  12.1
* numpy 1.26.4



## Feature Extraction
1. Physicochemical characteristics of amino acids
This feature has been integrated into utils.py and does not require additional extraction.
2. Intrinsic Disorder Region Features
a. Go to the website "https://iupred2a.elte.hu/dl_mail_sender" and then download iupred2a files. 
b. Put extr_iupred2a.py in the iupred2a files, and run extr_iupred2a.py (python extr_iupred2a.py A7RDD3.fasta) to get the outfile iupred_A7RDD3.npy.
3. ESM-C feature
a. Go to the website "https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12" and download the weights package.
b. run "ESM-C_extraction.py" (python ESM-C_extraction.py A7RDD3.fasta ) to get the out file "ESM-C_A7RDD3.npy"    
4. Structure feature
a. Go to the website "https://yanglab.qd.sdu.edu.cn/trRosetta/" andrRun trRosetta with the sequence of 'A7RDD3.fasta' as input to get the output file and rename it as 'tr_A7RDD3.npz'.
b. Run the python file of feature distillation, trrosetta_feature_distillation.py (python model/trrosetta_feature_distillation.py --input file --output file) to get the output file (input file directory which contains tr_A7RDD3.npz). 

## Prediction
a. Put "iupred_A7RDD3.npy", "ESM-C_A7RDD3.npy" and "tr_A7RDD3.npz" in the "feature" folder, Put "best_model.pt" and "config.json" in the "best_model" folder (the two files are downloaded in "trpHopt v1.0.0" release )and keep "A7RDD3.fasta", "model.py", "predict_single.py", "utils.py", "feature" folder and "best_model" folder in the same path.
b. Run the "predict_single.py" (python predict_single.py ./best_model ./feature/ESM-C_A7RDD3.npy ./feature/iupred_A7RDD3.npy ./feature/tr_A7RDD3.npz ./A7RDD3.fasta) to get the optimal pH.
