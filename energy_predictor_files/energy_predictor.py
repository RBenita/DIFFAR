import logging
import csv
import numpy as np
import os
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from energy_predictor_files.energy_predictor_utils import Accuracy
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
writer = SummaryWriter()

logger = logging.getLogger(__name__)

phoneme_to_index_dict = { "" : 1, "AA0": 2, 'AA1': 3, 'AA2': 4, 'AE0': 5 ,'AE1': 6
 ,'AE2': 7 ,'AH0': 8 ,'AH1': 9 ,'AH2': 10 ,'AO0': 11 ,'AO1': 12 ,'AO2': 13 ,'AW0': 14 ,'AW1': 15
   ,'AW2': 16 ,'AY0': 17 ,'AY1': 18 ,'AY2': 19,'B': 20 ,'CH': 21, 'D' : 22, 'DH' :23,
   'EH0':24, 'EH1':25, 'EH2':26 ,'ER0':27, 'ER1':28,'ER2':29 ,'EY0':30 , 'EY1':31 ,'EY2':32 ,'F':33 ,
   'G':34 ,'HH':35 ,'IH0':36 ,'IH1':37 ,'IH2':38 ,'IY0':39, 'IY1':40 , 'IY2':41 , 'JH':42,
    'K':43,  'L':44,  'M':45,  'N':46 , 'NG':47,  'OW0':48, 'OW1':49, 'OW2':50,'OY0':51, 'OY1':52, 
    'OY2':53, 'P':54, 'R':55, 'S':56, 'SH':57, 'T':58, 'TH':59, 'UH0':60, 'UH1':61,
     'UH2':62, 'UW0':63, 'UW1':64, 'UW2':65, 'V':66, 'W':67, 'Y':68, 'Z':69, 'ZH':70, 'spn':71, 'sil':72
}

def save_ckpt(model, path, model_class):
    ckpt = {
        "state_dict": model.state_dict(),
        "padding_token": model.padding_token,
        "model_class": model_class,
    }
    torch.save(ckpt, path)

def load_ckpt(path):
    ckpt = torch.load(path)
    ckpt["model_class"]["_target_"] = "energy_predictor_files.energy_predictor.CnnPredictor"
    model = hydra.utils.instantiate(ckpt["model_class"])
    model.load_state_dict(ckpt["state_dict"])
    model.padding_token = ckpt["padding_token"]
    model = model.cpu()
    model.eval()
    return model

def l2_log_loss(input, target):
    target_1 = target
    return F.mse_loss(
        input=input.float(),
        target=torch.nan_to_num(torch.log(target_1.float() + 1), nan=2),
        reduce=False
    ) 
    
class Collator:
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        ## padding the whole tensors on the batch to have dimenstion as the longes one.
        x = [item[0] for item in batch]
        lengths = [len(item) for item in x]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.padding_idx)
        y = [item[1] for item in batch]
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.padding_idx)
        mask = (x != self.padding_idx)
        return x, y, mask, lengths

class EnergyDataset(Dataset):
    def __init__(self, phonemes_duration_path, energy_npy_file,frame_len,hop_len, substring=""):
        with open(phonemes_duration_path, 'r') as file:
            reader = csv.reader(file)
            total_data = []
            for row in reader:
                total_data.append(row)

        self.header = total_data[0]
        total_data = np.array(total_data)
        self.phonemes_lines = total_data[1:,2]
        self.duration_lines = total_data[1:,3]
        logger.info(f"loaded {len(self.phonemes_lines)} files")
        self.wav_energy_lines = np.load(energy_npy_file, allow_pickle=True)
        logger.info(f"loaded {len(self.wav_energy_lines)} energy files")
        
    def __len__(self):
        return len(self.phonemes_lines)

    def __getitem__(self, i):
        cur_phoneme = self.phonemes_lines[i]
        cur_phoneme = cur_phoneme.split(",")
        cur_phoneme = [phoneme_to_index_dict[x] for x in cur_phoneme]
        
        cur_duration = self.duration_lines[i]
        cur_duration = cur_duration.split(",")
        cur_duration = list(map(float, cur_duration))
        
        cur_energy = self.wav_energy_lines[i]
        cur_energy = [int(x*100) for x in cur_energy]
        
        cur_phoneme = torch.LongTensor(cur_phoneme)
        cur_energy = torch.tensor(cur_energy)
        
        return cur_phoneme, cur_energy

class Predictor(nn.Module):
    def __init__(self, n_tokens, emb_dim):
        super(Predictor, self).__init__()
        self.n_tokens = n_tokens
        self.emb_dim = emb_dim
        self.padding_token = n_tokens
        # add 1 extra embedding for padding token, set the padding index to be the last token
        # (tokens from the clustering start at index 0)
        # therefore indexes 0-n_tokens-1 are for the phonemes, and index n_tokens for padding
        self.emb = nn.Embedding(n_tokens + 1, emb_dim, padding_idx=self.padding_token)

    def inflate_input(self, batch):
        """ get a sequence of tokens, predict their durations
        and inflate them accordingly """
        batch_durs = self.forward(batch)
        batch_durs = torch.exp(batch_durs) - 1
        batch_durs = batch_durs.round()
        output = []
        for seq, durs in zip(batch, batch_durs):
            inflated_seq = []
            for token, n in zip(seq, durs):
                if token == self.padding_token:
                    break
                n = int(n.item())
                token = int(token.item())
                inflated_seq.extend([token for _ in range(n)])
            output.append(inflated_seq)
        output = torch.LongTensor(output)
        return output

class CnnPredictor(Predictor):
    def __init__(self, n_tokens, emb_dim, channels, kernel, output_dim, dropout, n_layers):
        super(CnnPredictor, self).__init__(n_tokens=n_tokens, emb_dim=emb_dim)
        layers = [
            Rearrange("b t c -> b c t"),
            nn.Conv1d(emb_dim, emb_dim, kernel_size=kernel, padding=(kernel - 1) // 2),
            nn.Conv1d(emb_dim, channels, kernel_size=kernel-2, padding=(kernel-2 - 1) // 2),
            Rearrange("b c t -> b t c"),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers-1):
            layers += [
                Rearrange("b t c -> b c t"),
                nn.Conv1d(channels, channels, kernel_size=kernel, padding=(kernel - 1) // 2),
                nn.Conv1d(channels, channels, kernel_size=kernel-2, padding=(kernel-2 - 1) // 2),
                Rearrange("b c t -> b t c"),
                nn.ReLU(),
                nn.LayerNorm(channels),
                nn.Dropout(dropout),
            ]
        self.conv_layer = nn.Sequential(*layers)
        self.proj = nn.Linear(channels, output_dim)
    
    def forward(self, x):
        x = self.emb(x)
        x = self.conv_layer(x)
        x = self.proj(x)
        x = x.squeeze(-1)
        return x

def train(cfg):
    device = "cuda:0"
    model = hydra.utils.instantiate(cfg[cfg.model]).to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    # add 1 extra embedding for padding token, set the padding index to be the last token
    # (tokens from the clustering start at index 0)
    collate_fn = Collator(padding_idx=model.padding_token)
    logger.info(f"data: {cfg.train_data_csv}")
    train_ds = EnergyDataset( cfg.train_data_csv, cfg.train_energy_npy, cfg.frame_len, cfg.hop_len, substring=cfg.substring)
    valid_ds = EnergyDataset( cfg.valid_data_csv, cfg.valid_energy_npy, cfg.frame_len, cfg.hop_len, substring=cfg.substring)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    best_loss = float("inf")
    for epoch in tqdm(range(cfg.epochs)):
        train_loss, train_loss_scaled = train_epoch(model, train_dl, l2_log_loss, optimizer, device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_scaled", train_loss_scaled, epoch)
        valid_loss, valid_loss_scaled, *acc = valid_epoch(model, valid_dl, l2_log_loss, device)
        writer.add_scalar("Loss/val", valid_loss, epoch)
        writer.add_scalar("Loss/val_scaled", valid_loss_scaled, epoch)
        acc0, acc1, acc2, acc3 = acc
        if valid_loss_scaled < best_loss:
            path = f"{os.getcwd()}/energy_predictor_model.ckpt"
            save_ckpt(model, path, cfg[cfg.model])
            best_loss = valid_loss_scaled
            logger.info(f"saved checkpoint: {path}")
            logger.info(f"[epoch {epoch}] train loss: {train_loss:.3f}, train scaled: {train_loss_scaled:.3f}")
            logger.info(f"[epoch {epoch}] valid loss: {valid_loss:.3f}, valid scaled: {valid_loss_scaled:.3f}")
            logger.info(f"acc: {acc0,acc1,acc2,acc3}")
    writer.flush()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_loss_scaled = 0
    for x, y, mask, _ in tqdm(loader):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        yhat = model(x)
        loss = criterion(yhat, y) * mask
        loss = torch.mean(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        yhat_scaled = torch.exp(yhat) - 1
        scaled_loss = torch.mean(torch.abs(yhat_scaled - y) * mask)
        epoch_loss_scaled += scaled_loss.item()
    return epoch_loss / len(loader), epoch_loss_scaled/ len(loader)

def valid_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_loss_scaled = 0
    acc = Accuracy()
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        yhat = model(x)
        loss = criterion(yhat, y) * mask
        loss = torch.mean(loss)
        epoch_loss += loss.item()
        yhat_scaled = torch.exp(yhat) - 1
        scaled_loss = torch.sum(torch.abs(yhat_scaled - y) * mask) / mask.sum()
        acc.update(yhat_scaled[mask].view(-1).float(), y[mask].view(-1).float())
        epoch_loss_scaled += scaled_loss.item()
        acc.update(yhat[mask].view(-1).float(), y[mask].view(-1).float())

    y_list = y[0, :15].tolist()
    yhat_scales_list = yhat_scaled[0, :15].tolist()

    logger.info(f"example y: {[round(num, 3) for num in y_list]}")
    logger.info(f"example yhat: {[round(num, 3) for num in yhat_scales_list]}")
    # logger.info(f"example y: {y[0, :10].tolist()}")
    # logger.info(f"example yhat: {yhat_scaled[0, :10].tolist()}")
    acc0 = acc.acc(tol=0)
    acc1 = acc.acc(tol=1)
    acc2 = acc.acc(tol=2)
    acc3 = acc.acc(tol=3)
    logger.info(f"accs: {acc0,acc1,acc2,acc3}")
    return epoch_loss / len(loader), epoch_loss_scaled / len(loader), acc0, acc1, acc2, acc3

@hydra.main(config_path=".", config_name="energy_predictor.yaml")
def main(cfg):
    logger.info(f"{cfg}")
    train(cfg)
    # test(cfg)

if __name__ == "__main__":
    main()