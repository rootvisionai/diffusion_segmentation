import os
import torch
import tqdm
import torchvision

def visualize(net, eval_loader, device="cuda"):
    """
    For visual/manual evaluation
    predictions are saved to /inference_examples
    """
    if not os.path.isdir("./inference_examples"):
        os.makedirs("./inference_examples")

    net.eval()
    pbar = tqdm.tqdm(enumerate(eval_loader))
    for i, (image, mask) in pbar:
        with torch.no_grad():

            if i > 99:
                break

            pred = net.infer(image.to(device)).cpu()
            pred = pred[0].cpu()
            image = image[0].cpu()
            concat_image_to_save = torch.stack([image, mask.squeeze(0), pred], dim=0)
            torchvision.utils.save_image(concat_image_to_save, f"./inference_examples/pred{i}.png")
            pbar.set_description(f"[{i}/{len(eval_loader)}]")

    net.train()