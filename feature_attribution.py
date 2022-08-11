# %%
import torch
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import numpy as np
import pandas as pd

from src.transformer_model import TranscriptTIS
from TIS_transformer import prep_input
import argparse

gpus = True
device = torch.device('cuda') if gpus else torch.device('cpu')
max_seq_len = 25000

THRESH = 0.30
transcript_list = ['ENST00000507866', 'ENST00000400677']
baseline_value = 0

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint")
parser.add_argument("--tsv", type=str, help="Path to the tsv file")
parser.add_argument("--chr_path", type=str, help="Path to the chromosome file (numpy file)")
parser.add_argument("--out_dir", type=str, help="Path where the attribution matrix will be saved")
parser.add_argument("--batch_size", type=int, default=10,
                    help="Internal batch size used for integrated gradients")
parser.add_argument("--n_steps", type=int, default=50,
                    help="Number of steps for integrated gradients")
parser.add_argument("--data_steps", type=int, default=1,
                    help="Proportion of the database to use, 1 to use all data, 2 for half, and n for 1/n")

args=parser.parse_args()

transfer_checkpoint = args.checkpoint

# %%
def get_transcript_name_fasta(s):
    return s.split('|')[0]
def remove_version(s):
    return s.split('.')[0]

def vec2DNA(dna_vec):
    seq_list = ['A', 'T', 'C', 'G', 'N']
    dna_seq = []
    for id in dna_vec:
        dna_seq.append(seq_list[id])
    return ''.join(dna_seq)

# %%
tsv = pd.read_csv(args.tsv, sep='\t', skip_blank_lines=True, skiprows=1, 
                    dtype={'chr': str}, usecols=['transcript accession', 'protein accession numbers', 'protein type', 'start transcript coordinates', 'stop transcript coordinates', 'keep protein'])
tsv = tsv[tsv['keep protein']]

# Load the model
# %%
tis_tr = TranscriptTIS.load_from_checkpoint(transfer_checkpoint)
tis_tr.to(device)
"done"

# Reading the dna sequence

# %%
def forward_prediction(x, mask, nt_mask):
    out = F.softmax(tis_tr.forward(x, mask, nt_mask, apply_dropout=False), dim=1)[:,1]
    return out.reshape(x.size(0), x.size(1)-2)

def generate_input_prediction(chr_path, step=1):
    # step : step when iterating over transcripts
    data = np.load(chr_path, allow_pickle=True)
    indices = [i for i in range(0, len(data[:,0]), step) if len(data[:,0][i]) <  max_seq_len]
    x_data = np.array([data[:,0][i][:,0] for i in indices], dtype=object)
    tr_ids = data[:,1][indices] # tr_ids shoud have the same shape as x_data  
    inputs = []
    model_tis = []
    for x in x_data:
        inputs.append(prep_input(x, device))
        model_output = forward_prediction(inputs[-1][0], inputs[-1][1], inputs[-1][2]).detach().cpu().numpy()
        model_tis.append(np.argwhere(model_output[0]>THRESH).flatten())
    return tr_ids, inputs, model_tis

# Integrated gradients

# %%
tis_tr.zero_grad()
integrated_gradients = LayerIntegratedGradients(forward_prediction, [tis_tr.model.token_emb, tis_tr.model.pos_emb])

# %%
torch.cuda.empty_cache()

# %%
tsv_groups = tsv.groupby('transcript accession')
matrix = []

tr_ids_array, input_array, model_tis_array = generate_input_prediction(args.chr_path, args.data_steps)

for transcript_name, input, model_tis in zip(tr_ids_array, input_array, model_tis_array):
    print(transcript_name)
    try:
        tr_df = tsv_groups.get_group(transcript_name)
    except KeyError: # non coding transcript
        continue
    baseline = torch.ones_like(input[0]) * baseline_value    

    for tis in model_tis:
        protein_name = tr_df[tr_df['start transcript coordinates'] == tis+1]['protein accession numbers']
        if len(protein_name) == 0:
            print("tis does not correspond to any proteins")  
            continue
        protein_name = protein_name.iloc[0]
        print(protein_name)
    
        attributions_ig, delta = integrated_gradients.attribute(inputs=input[0], baselines=baseline,target=int(tis),
                                                            additional_forward_args = (input[1], input[2]), return_convergence_delta=True,
                                                            n_steps=args.n_steps, internal_batch_size=args.batch_size)
        attribution = attributions_ig[0].sum(dim=2).detach().cpu().numpy().flatten()[1:-1] + attributions_ig[1].sum(dim=2).detach().cpu().numpy().flatten()[1:-1]

        matrix.append([transcript_name, protein_name, tis+1, delta.cpu().numpy()[0], attribution])

np.save(args.out_dir, np.array(matrix, dtype=object), allow_pickle=True)

# matrix : [ ...
#           [tr_name, protein name, coordinate, delta, [attribution]]
#          ... ]           
