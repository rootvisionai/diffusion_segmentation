# diffusion_segmentation
Use de-noising diffusion model to segment the objects on the image step by step.
Currently, in experimentation and training process.
- Any contribution is appreciated!

[Model checkpoints to download](https://drive.google.com/drive/folders/18pPK4EkeSxQ2-VMhLBbJQ37S8FOVI4DY?usp=sharing)

### Output from ongoing experiment:

![1](./keep/image.png)
![2](./keep/mask.png)
![3](./keep/pred.png)

## Plans & Updates
- Made the model work, it can train and converge. Because we try to generate colors 
to represent the classes for each pixel, it will be more difficult than usual to revert
back the classes. This will be done later.
- DONE: ~~The colors that are assigned to the classes are pre-defined. The color vectors have 
no semantic relation to the classes. This needs to change. I will define vectors (colors) 
that are semantically related to the classes, so that model will generate more meaningful
vectors for each pixel. This will enable the model to be used for transfer learning and few-shot
semantic segmentation tasks. I believe that it may make the results better as well.~~
