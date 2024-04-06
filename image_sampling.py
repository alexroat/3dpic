import torch
import cv2

def image_to_ft(frame):
    # Trasforma il frame in un tensore PyTorch
    tensor = torch.tensor(frame, dtype=torch.uint8)
    
    # Estrai le coordinate i, j
    i, j = torch.meshgrid(torch.arange(tensor.shape[0]), torch.arange(tensor.shape[1]))
    
    # Concatena i, j con i valori di colore r, g, b
    tensor_with_coords = torch.cat((i.unsqueeze(-1), j.unsqueeze(-1), tensor.flip(2)), dim=-1)
    
    return tensor_with_coords


def ft_to_image(tensor, original_height, original_width):
    # Estrai le coordinate e le intensità di colore dal tensore
    coords = tensor[..., :2]
    rgb = tensor[..., 2:].to(torch.uint8)
    
    print(rgb)
    
    # Inverti l'ordine dei canali di colore da BGR a RGB
    #rgb = rgb.flip(-1)

    # Reshape il tensore per ottenere l'immagine
    image = torch.zeros((original_height, original_width, 3), dtype=torch.uint8)
    image[coords[..., 0], coords[..., 1]] = rgb.view(original_height, original_width, 3)

    # Converti l'immagine in un array NumPy
    image_np = image.numpy()

    # Converti l'immagine da BGR a RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    return image_rgb



def add_coordinates_channels(image):
    # Trasforma il frame in un tensore PyTorch
    image = torch.tensor(image, dtype=torch.uint8)
    h, w, _ = image.shape
    x_coords, y_coords = torch.meshgrid(torch.arange(w), torch.arange(h))
    coordinates = torch.stack((x_coords, y_coords), dim=-1)
    coordinates = coordinates.float()
    coordinates /= max(w, h)  # normalizzazione delle coordinate
    image_with_coords = torch.cat([image, coordinates], dim=-1)
    return image_with_coords


# Esempio di utilizzo
frame = cv2.imread("marter.jpeg")
h,w,_=frame.shape
tensor = image_to_ft(frame)

nframe=ft_to_image(tensor,h,w)

add_coordinates_channels(nframe)

#print(tensor)
cv2.imshow("Reconstructed Image", nframe)
cv2.waitKey(0)
cv2.destroyAllWindows()