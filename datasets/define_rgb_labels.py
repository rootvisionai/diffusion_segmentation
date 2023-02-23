import torch
import torch.nn as nn
from transformers import Data2VecTextModel, AutoTokenizer
from sklearn.decomposition import PCA
import tqdm
import numpy as np


class TextEncoder(nn.Module):
    def __init__(self,
                 device):

        super(TextEncoder, self).__init__()

        self.device = device

        # Initializing a model (with random weights) from the facebook/data2vec-text-base style configuration
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/data2vec-text-base")
        self.model = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base")

    def forward(self, text):
        inp = self.tokenizer(text, return_tensors="pt")
        out = self.model(**inp.to("cuda")).pooler_output
        return out

def criterion(similarity):
    # makes the angles between proxies maximum.
    loss = -torch.log(1 - similarity) * 2
    loss = torch.clamp(loss, min=0, max=8)
    return loss

def sim_func(vectors, debug=False, device="cuda"):

    sim_mat = torch.nn.functional.linear(vectors.to(device), vectors.to(device))
    similarity_vector = torch.triu(sim_mat,
                                   diagonal=1)
    combinations = torch.nonzero(similarity_vector, as_tuple=True)
    similarity_vector = similarity_vector[combinations]

    return similarity_vector

encoder = TextEncoder(device="cuda")
encoder.to("cuda")
text_list = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

vectors = []
for text in text_list:
    vectors.append(encoder(text))

vectors = torch.stack(vectors, dim=0).squeeze(1)
a_norm = vectors.norm(dim=-1, p=2)  # (batch size, 1)
vectors = vectors.divide(a_norm.unsqueeze(1))

pca = PCA(n_components=3)
vectors_reduced = pca.fit_transform(vectors.detach().cpu().numpy())
vectors_reduced = torch.from_numpy(vectors_reduced)
a_norm = vectors_reduced.norm(dim=-1, p=2)  # (batch size, 1)
vectors_reduced = vectors_reduced.divide(a_norm.unsqueeze(1))

vectors_reduced = torch.nn.Parameter(vectors_reduced)
vectors_reduced.requires_grad = True
optimizer = torch.optim.AdamW(params=[vectors_reduced], lr=0.1)

pbar = tqdm.tqdm(range(100))
for i in pbar:
    optimizer.zero_grad()
    distance_vector = sim_func(vectors_reduced)

    if distance_vector.mean() > 0.98 and i == 0:
        distance_vector -= 0.02

    loss = criterion(distance_vector).sum()
    loss.backward()
    angle = torch.rad2deg(torch.acos(torch.clip(distance_vector,
                                                -0.9999, 0.9999))).detach().cpu().numpy()

    text = f"ITER [{i}] | LOSS [{loss.item()}] | MIN ANGLE [{angle.min()}]"
    pbar.set_description(text)

    optimizer.step()
    if angle.min()>90:
        break

vectors_reduced = vectors_reduced.detach().cpu().numpy()
vectors_reduced = vectors_reduced - np.expand_dims(vectors_reduced.min(axis=1), axis=1)
vectors_reduced = vectors_reduced / np.expand_dims(vectors_reduced.max(axis=1), axis=1)
vectors_reduced *= 255
vectors_reduced = vectors_reduced.astype(np.uint8)

print("COPY RESULTS BELOW TO datasets/pascal_voc.py in place of VOC_COLORMAP_NEW\n")
print("VOC_COLORMAP = [")
for vr in vectors_reduced:
    print(f'    {tuple(vr)},')
print("]")
print("\nCOPY RESULTS ABOVE TO datasets/pascal_voc.py in place of VOC_COLORMAP_NEW")