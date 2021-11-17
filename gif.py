from matplotlib import animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import gym

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
def save_frames_as_gif(frames, path='./', filename='gym_animation.mp4'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0][0].shape[1] / 72.0, frames[0][0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0][0])
    plt.axis('off')
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 50)

    def animate(i):
        img_pil = Image.fromarray(frames[i][0])
        draw = ImageDraw.Draw(img_pil)
        draw.text((1, 1), 'episode:' + str(frames[i][1]) + ', score:' + str(round(frames[i][2], 3)), 'red', font=font, align="left")
        patch.set_data(img_pil)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=0.05)
    anim.save(path + filename, writer='ffmpeg', fps=30)
