# GNN-RAG Animal Image Editing

This repository contains a Jupyter notebook that compares three ways to generate animal images with Stable Diffusion:

- Plain text-to-image generation.
- CLIP retrieval followed by image-to-image generation (normal RAG).
- Graph-based retrieval followed by image-to-image generation (GNN-RAG).

The notebook uses the Animals with Attributes 2 (AwA2) dataset as an image and attribute memory. It builds attribute profiles from AwA2 class attributes and CLIP image evidence, then chooses a retrieved image and image-to-image strength based on the requested edits. Message passing is training-free; the notebook does not train a GNN.

## Repository contents

| File | Description |
|---|---|
| `gnn_rag_image_pipeline.ipynb` | Data setup, retrieval, generation, calibration, benchmark, scoring, optional human-rating app, and chatbot. |
| `requirements.txt` | Python packages used by the notebook. Versions are not pinned. |

Generated images, calibration data, benchmark rows, and figures are not included in the repository. The notebook writes these to the `graph_rag_image_diffusion` folder in Google Drive.

## Run the notebook

The notebook is configured for Google Colab with a CUDA GPU (a T4 is the target setup) and Google Drive. It stops if CUDA is unavailable or Drive is not mounted.

1. Open `gnn_rag_image_pipeline.ipynb` in Google Colab.
2. In Colab, select **Runtime → Change runtime type → T4 GPU** and connect Google Drive when prompted.
3. Run the notebook cells in order. The first install cell installs `diffusers`, `remotezip`, and `gradio`, and removes `torchao` because it can conflict with the installed Diffusers version.
4. If Diffusers was imported earlier in the same session, restart the runtime after the install cell, then continue from the top.

AwA2 images and model weights are downloaded when needed. The notebook uses Stable Diffusion 1.5, SD-Turbo, CLIP ViT-B/32 for retrieval, and CLIP ViT-L/14 for scoring. Check the dataset and model terms before using or redistributing generated material.

The calibration and benchmark steps can take substantial GPU time and storage. They save progress to Google Drive so a run can resume after interruption.

## Experiments in the notebook

The benchmark includes plain diffusion, normal RAG, GNN-RAG, matched-step comparisons, retrieval and planning ablations, and SD-Turbo baselines. CLIP scores, attribute gains, animal-class predictions, denoising steps, and generation time are reported. Confidence intervals are bootstrapped over prompts, with multiple seeds averaged per prompt.

An optional Gradio app supports blind A/B image ratings. The notebook also includes an interactive chatbot for comparing the three main pipelines.

## Scope and limitations

- AwA2 provides 50 animal classes and 85 attributes; edits outside that vocabulary are treated as unknown.
- Prompt parsing uses rules, and the graph message passing has no learned weights.
- Recolouring affects the whole image, and exact counting edits (such as requesting five legs) are not supported reliably.
- CLIP is used for automatic image scoring. Human ratings require running the app and collecting responses separately.
- The notebook's results depend on the GPU, package versions, and model versions used for a run.
