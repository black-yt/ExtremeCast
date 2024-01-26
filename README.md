# Exformer

Download required data and checkpoints from [GoogleDrive](https://drive.google.com/drive/folders/1UyLCFGxZnBx-sCIVr8snxkWNUJAqnyKu)

Store files according to the following structure:

<pre>.
├── 2017-12-31T18:00:00.npy
├── 2018-01-01T00:00:00.npy
├── 2018-01-01T06:00:00.npy
├── BoostEns.py
├── climatology-2018-01-01T06:00:00.npy
├── data_mean.npy
├── data_std.npy
├── diffusion_max.npy
├── diffusion_min.npy
├── max_logvar.npy
├── min_logvar.npy
├── model
│   ├── attend.py
│   └── denoising_diffusion_pytorch.py
├── model_d.onnx
├── model_g.pth
├── pic.py
├── run.py
└── utils.py </pre>


Run run.py on GPU device

For the input and output of the model, their dimensions are [B, C, H, W], where C=69, each channel corresponds to a weather variable, and their correspondence is shown in the [variable-order](https://docs.google.com/spreadsheets/d/1KNY0P4_zkH9r1RIEu_VhvZic65Apz9BX/edit?usp=sharing&ouid=117415241894938396384&rtpof=true&sd=true)
