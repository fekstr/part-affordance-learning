import io

from PIL import Image
import matplotlib.pyplot as plt


def get_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.clf()
    return img