# radiownet_model.py
from __future__ import print_function, division
import os
import torch
import numpy as n
import modules as modules

# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

class RadioWNetModel:
    def __init__(self, model_path=None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "models", "Trained_Model_SecondU.pt")
        torch.set_default_dtype(torch.float32)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = modules.RadioWNet(phase="secondU")
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()
        # summary(self.model, input_size=(2, 256, 256))
        print(f"Model loaded and moved to {self.device}")

    def predict(self, input_image):
        t_input = torch.tensor(input_image, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out1, out2 = self.model(t_input)
        out1 = (256 * out1.cpu().numpy()).astype(n.uint8)
        out2 = (256 * out2.cpu().numpy()).astype(n.uint8)
        return out1, out2
