import sys
import os
import parse
import PIL

from matplotlib import pyplot as plt


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python visualize.py <images_dir> <outname>")
        return
    else:
        images_dir = sys.argv[1]
        outname = sys.argv[2]

    fig, axs = plt.subplots(3, 3, figsize=(21, 21))

    idx = 0
    idy = 0

    for filename in sorted(os.listdir(images_dir)):
        plot_num = parse.parse("track_{}.png", filename)
        if plot_num is not None:
            plot_name = f"track_{plot_num[0]}.png"
            img = PIL.Image.open(os.path.join(images_dir, plot_name))

            # crop the image to remove the white space
            crop_size = 150
            img = img.crop(
                (crop_size, crop_size, img.width - crop_size, img.height - crop_size)
            )

            axs[idx, idy].imshow(img)
            axs[idx, idy].set_title(f"Episode {plot_num[0]}")
            axs[idx, idy].axis("off")
            idy += 1
            if idy == 3:
                idy = 0
                idx += 1
            if idx == 3:
                break

    plt.tight_layout()
    plt.savefig(outname)


if __name__ == "__main__":
    main()
