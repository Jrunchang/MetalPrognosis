# MetalPrognosis
__MetalPrognosis__: a Biological Language Model-based Approach for Disease-Associated Mutations in Metal-Binding Site prediction

## 1.Ststem requirement
MetalPrognosis is developed under **Linux** environment with:

`bio-transformers==0.1.17

biopython==1.81

torch==1.12.0+cu116

torchaudio==0.12.0+cu116

torchvision==0.13.0+cu116`

## 2. Download pre-trained language model
You need to prepare the pretrained language model ESMfold to run MetalPrognosis:<br>
Download esm1b_t33_650M_UR50S model at:<https://github.com/facebookresearch/esm>

## 3.Run MetalPrognosis for prediction
Users can choose to upload the metalloproteins **FASTA** sequence of Homo sapiens, or the protein **PDB** structure and the corresponding metal binding **sites**.

`python predict.py --input demo.fasta --site 20,33,58 --outpath /home/runchangjia/id_dict/result.csv`<br>

`python predict.py --input P50461.pdb --site 58,122 --outpath /home/runchangjia/id_dict/pye.csv`


## 4.Webserver
we developed and deployed a free online web server to predict disease-related mutation sites on metalloproteins at:
<http://metalprognosis.unimelb-biotools.cloud.edu.au/>

