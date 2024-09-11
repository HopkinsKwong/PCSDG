# PCSDG
This is the repository for the source code of the paper "Structure-Aware Single-Source Generalization with Pixel-Level Disentanglement for Joint Optic Disc and Cup Segmentation."

**The paper primarily investigates the "single-source domain generalization" problem for "optic cup and disc joint segmentation." **

## Model training and inference
The complete source code is located in the `src` folder. To use this project, follow the steps below:
```
# change the folder name
mv ./src ./deeplearning
cd  deeplearning

# Train
python -m deeplearning.training.run_training

# Inference
python -m deeplearning.inference.run_inference
```


