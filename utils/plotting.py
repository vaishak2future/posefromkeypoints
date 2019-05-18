# from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from PIL import Image

def plot_mesh(img, verts, faces, R, t, K):
    ax = plt.subplot(111)
    plt.imshow(img)
    verts_2d = np.matmul(K, np.matmul(R, verts.T) + t).T
    verts_2d = verts_2d[:,:2] / verts_2d[:,2,None]

    patches = []
    for face in faces:
        points = [verts_2d[i_vertex-1] for i_vertex in face]
        poly = Polygon(points, True)
        patches.append(poly)
    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)
    ax.add_collection(p)
    plt.show()

if __name__ == '__main__':
    data = np.load('../data/data.npz')
    vertices = data['vertices']
    faces = data['faces']

    img = Image.open('../images/train_9.jpg')
    R = np.array([[-0.4245,-0.9032, -0.0634],
                  [-0.4304, 0.2629, -0.8635],
                  [ 0.7966,-0.3392, -0.5004]])
    T = np.array([[-0.1676], [0.0652],[0.8749]])
    from IPython.core.debugger import Pdb
    Pdb().set_trace()

    K = data['K']

    plt.close('all')
    ax = plt.subplot(111)

    plt.imshow(img)
    plot_mesh(vertices, faces, R, T, K, ax)
    plt.show()
