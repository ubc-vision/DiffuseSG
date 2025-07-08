# DiffuseSG: Diffusion-based Grounded Scene Graph Generation

DiffuseSG is a novel approach to scene graph generation using diffusion models. This repository contains the implementation of our method for generating grounded scene graphs using diffusion models.

This is the official implementation of the DiffuseSG method proposed in the paper "Joint Generative Modeling of Grounded Scene Graphs and Images via Diffusion Models".

## Installation

### Using Python Virtual Environment
```bash
# Create and activate virtual environment
python -m venv venvdiffusesg
source venvdiffusesg/bin/activate

# Install dependencies
pip install -U pip
pip install -r setup/requirements.txt
```

### Using Conda
```bash
# Create and activate conda environment
conda create -n diffusesg python=3.8 -y
conda activate diffusesg

# Install dependencies
conda install conda-forge::gcc
pip install -U pip
pip install -r setup/requirements.txt

# Note: You may need to conda deactivate and conda activate to ensure everything works properly
```

### Dataset Preparation
```bash
# Prepare scene graph datasets
unzip data_scenegraph/data_scenegraph.zip -d data_scenegraph/
```

## Usage

### Training

#### COCO Dataset
```bash
export W_IOU=1.0 && torchrun --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT train.py \
    --ddp \
    -c config/edm_diffuse_sg/edm_diffuse_sg_regular_coco.yaml \
    --node_encoding bits \
    --edge_encoding bits \
    --iou_loss_type giou \
    --iou_loss_weight ${W_IOU} \
    --max_epoch 5001 \
    --batch_size=1024 \
    --eval_size=1000 \
    --sample_interval 500 \
    --save_interval 50 \
    -m=loss_giou_w${W_IOU}_$(hostname)
```

#### Visual Genome Dataset
```bash
export W_IOU=1.0 && torchrun --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT train.py \
    --ddp \
    -c config/edm_diffuse_sg/edm_diffuse_sg_regular_visual_genome.yaml \
    --node_encoding bits \
    --edge_encoding bits \
    --iou_loss_type giou \
    --iou_loss_weight ${W_IOU} \
    --max_epoch 5001 \
    --batch_size=512 \
    --eval_size=1000 \
    --sample_interval 500 \
    --save_interval 50 \
    -m=loss_giou_w${W_IOU}_$(hostname)
```

### Evaluation and Sampling
```bash
export BS=1024 && python eval.py -p ${CKPT_PATH} --eval_size 0 --batch_size ${BS} -m=$(hostname)
```

## Configuration

The model can be configured through YAML files in the `config/` directory. Key configuration options include:

- Model architecture parameters
- Training hyperparameters
- Dataset settings
- Evaluation metrics

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{xu2024joint,
  title={Joint generative modeling of scene graphs and images via diffusion models},
  author={Xu, Bicheng and Yan, Qi and Liao, Renjie and Wang, Lele and Sigal, Leonid},
  journal={arXiv preprint arXiv:2401.01130},
  year={2024}
}
```

## License

This project is licensed under the terms of the LICENSE file included in this repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
