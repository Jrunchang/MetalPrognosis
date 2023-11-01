# MetalPrognosis
MetalPrognosis: a Biological Language Model-based Approach for Disease-Associated Mutations in Metal-Binding Site prediction

## Ststem requirement
MetalPrognosis is developed under Linux environment with:

`bio-transformers==0.1.17
biopython==1.81
torch==1.12.0+cu116
torchaudio==0.12.0+cu116
torchvision==0.13.0+cu116`

## Run MetalPrognosis for prediction
`python predict.py --fasta demo.fasta --site 20,33,58 --outpath /home/runchangjia/id_dict/result.csv`

## Webserver
<http://metalprognosis.unimelb-biotools.cloud.edu.au/>

