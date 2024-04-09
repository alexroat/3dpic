import cv2
import torch
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import caffeine

from model import *

print("MPS available:",torch.backends.mps.is_available())
print("MPS built:",torch.backends.mps.is_built())
device = torch.device("mps")
print(device)
dtype = torch.float

D = 1

# Creare un'istanza dell'autoencoder
autoencoder = Autoencoder().to(device)


# Definisci l'ottimizzatore e la loss function
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

K=2
checkpoint_path = 'checkpoint.pth'

# Attiva il mantenimento attivo del computer
caffeine.on(display=True)

# Funzione per caricare il video e processare i frame
def process_video(video_path, autoencoder):
    cap = cv2.VideoCapture(video_path)
    total_error = 0
    
    if os.path.exists(checkpoint_path):
        # Carica lo stato del modello solo se il file esiste
        autoencoder.load_state_dict(torch.load(checkpoint_path))

    i=0
    while cap.isOpened():
    #for i in range(2*K):
        ret, frame = cap.read()
        if not ret:
            break
        framecount_container.write(f"frame {i}")
        
        frame = frame[..., [2, 1, 0]]
        frame = cv2.resize(frame, (1024, 1024))
        frame = frame / 255.0  # Normalizzazione tra 0 e 1
                
        #converte il frame in tensore
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)


        #print(frame.shape)
        
        # Esegue l'inferenza con l'autoencoder
        with torch.set_grad_enabled(True):
            decoded_frame = autoencoder(frame)

            # Calcola l'errore e aggiorna il totale
            loss = criterion(decoded_frame, frame)
            progress_container.write(f"Loss: {loss.item()}")
            loss.backward()
        
        i+=1
        if not (i%K):
            # Esecuzione della backpropagation
            optimizer.step()
            
            # Azzeramento dei gradienti
            optimizer.zero_grad()
            
            torch.save(autoencoder.state_dict(), 'checkpoint.pth')

            
                

        # Visualizza il fotogramma originale e quello ricostruito
        frame = frame.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        decoded_frame = decoded_frame.squeeze().permute(1, 2, 0).cpu().detach().numpy()




        # Aggiorna il contenuto del contenitore con l'immagine
        original_column, decoded_column = images_container.columns(2)
        original_column.image(frame, caption="Original Frame", use_column_width=True)
        decoded_column.image(decoded_frame, caption="Decoded Frame", use_column_width=True)
      
    cap.release()
    return total_error


# Interfaccia utente Streamlit
st.title("Autoencoder Video Reconstruction")

images_container = st.empty()
progress_container = st.empty()
framecount_container = st.empty()

# Processa il video e visualizza i frame originali e ricostruiti
while True:
    total_error = process_video("D04_V_indoorYT_move_0001.mp4", autoencoder)
