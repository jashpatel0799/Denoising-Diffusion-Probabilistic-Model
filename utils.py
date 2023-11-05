import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# SAVE MODEL
def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Save pytorch model to a traget dir
    Args:
        model: A traget pytorch model
        target_dir: Directory to save the model to
        model_name: File name to save model. should include ".pth" or ".pt" at the end of the file extention

    Example usage:
        save_model(model = model_0, target_dir = "models", model_name="model.pth")

    """

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents = True, exist_ok = True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should be end with .pth or .pt"
    model_save_path = target_dir_path / model_name

    print(f"\nSaving Model At: {model_save_path}")
    torch.save(obj = model.components['unet'].state_dict(), f = model_save_path)


# LOAD MODEL
def load_model(model: torch.nn.Module, model_path: str):
    """
    Load pytorch model from source dir
    Args:
        model: A model which need to load
        source_dir: path where trained model is saved. should be full path including model name

    Example usage:
        load_model(model = model_0, source_path = "models/model.pth")
    """
    model.load_state_dict(torch.load(f = model_path, map_location=torch.device('cpu')))
    print("\nModel Loaded.")



# PLOT Function
def plot_graph(train_losses: list, test_losses: list, fig_name: str):
    """
    Plot the grapoh of loss abd accuray of the model
    Args:
        train_losses: list of train loss
        test_losses: list of test loss
        train_accs: list of train accuracy
        test_accs: list of test accuracy
        fig_name: name of image file which with you want to save plot image and must include .jpg 

    Example usage:
        plot_graph(train_losses = train_loss, test_losses = test_loss, train_accs = train_acc,
                   test_accs = test_acc, fig_name = "plot.jpg")
    """
    plt.figure(figsize = (10, 6))
    # plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label = "Train Loss")
    plt.plot(range(len(test_losses)), test_losses, label = "Test Loss")
    plt.legend()
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    # plt.show()

    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(train_accs)), train_accs, label = "Train Accuracy")
    # plt.plot(range(len(test_accs)), test_accs, label = "Test Accuracy")
    # plt.legend()
    # plt.xlabel("Epoches")
    # plt.ylabel("Accuracy")
    # # plt.show()
    plt.savefig(fig_name)


def fid_graph(fid_scores: list, fig_name: str):
    """
    Plot the grapoh of loss abd accuray of the model
    Args:
        fid_scores: list of fids
        fig_name: name of image file which with you want to save plot image and must include .jpg 

    Example usage:
        fid_graph(fid_scores = fids, fig_name = "fid.jpg")
    """
    plt.figure(figsize = (10, 6))
    # plt.subplot(1, 2, 1)
    plt.plot(range(len(fid_scores)), fid_scores, label = "FID")
    # plt.plot(range(len(test_losses)), test_losses, label = "Test Loss")
    plt.legend()
    plt.xlabel("FIDS")
    plt.ylabel("Loss")
    plt.savefig(fig_name)



def show_images(image, title: str, fig_name: str):

  if type(image) is torch.Tensor:

    image = image.detach().cpu().numpy()

  fig = plt.figure(figsize = (9, 9))
  row = int(len(image) ** (1/2))
  col = round(len(image) / row)

  idx = 0
  for r in range(row):
    for c in range(col):
      fig.add_subplot(row, col, idx + 1)

      # if idx < len(image):
      plt.axis(False)
      # print(image[idx].shape)
      plt.imshow(np.transpose(image[idx], (1, 2, 0)))#, cmap="gray")
      idx += 1

  fig.suptitle(title, fontsize = 20)
  plt.axis(False)
  plt.savefig(fig_name)
#   plt.show()