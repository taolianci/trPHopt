# Structure-Guided Hybrid GCN-Transformer Framework for Enzyme Optimal pH Prediction

This tool performs optimal pH prediction for a single enzyme sequence using the hybrid GCN-Transformer Framework. It integrates multiple feature modalities including ESM-C representations, protein disorder, and structural features.

In the following, we take a enzyme sequence file as an example to show the prediction process.
Here, the enzyme sequence file with uniprot\_id 'A7RDD3'.
trPHopt uses the following dependencies:

* PyTorch  2.3.0
* Python  3.12
* CUDA  12.1
* numpy 1.26.4



## Feature Extraction

1. Physicochemical characteristics of amino acids
This feature has been integrated into utils.py and does not require additional extraction.
2. Intrinsic Disorder Region Features
a. Go to the website "https://iupred2a.elte.hu/dl\_mail\_sender" and then download iupred2a files.
b. Put extr\_iupred2a.py in the iupred2a files, and run extr\_iupred2a.py (python extr\_iupred2a.py A7RDD3.fasta) to get the outfile iupred\_A7RDD3.npy.
3. ESM-C feature
a. Go to the website "https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12" and download the weights package.
b. run "ESM-C\_extraction.py" (python ESM-C\_extraction.py A7RDD3.fasta ) to get the out file "ESM-C\_A7RDD3.npy"
4. Structure feature
a. Go to the website "https://yanglab.qd.sdu.edu.cn/trRosetta/" and Run trRosetta with the sequence of 'A7RDD3.fasta' as input to get the output file and rename it as 'tr\_A7RDD3.npz'.
b. Run the python file of feature distillation, trrosetta_feature_distillation.py (python model/trrosetta_feature_distillation.py --input folder --output folder) to get the output file (input folder contains tr_A7RDD3.npz which is caculated from trRosetta server,  After running, the output folder will contain a distillation file with the same name ).

## Prediction

a. Put "iupred\_A7RDD3.npy", "ESM-C\_A7RDD3.npy" and "tr\_A7RDD3.npz" in the "feature" folder, Put "best\_model.pt" and "config.json" in the "best\_model" folder (the best\_model.pt file are downloaded in "trPHopt" release )and keep "A7RDD3.fasta", "model.py", "predict\_single.py", "utils.py", "feature" folder and "best\_model" folder in the same path.
b. Run the "predict\_single.py" (python predict\_single.py ./best\_model ./feature/ESM-C\_A7RDD3.npy ./feature/iupred\_A7RDD3.npy ./feature/tr\_A7RDD3.npz ./A7RDD3.fasta) to get the optimal pH.

