import os
import sys
import argparse
import numpy as np
import scipy
from scipy import io, misc, sparse
from sklearn.preprocessing import normalize

PIC_DIR = 'pics'
EXTENSION = '.bmp'

def cartesian(arrays, out=None):

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def rgb2yuv(rgb):
    rgb = rgb / 255.0
    y = np.clip(np.dot(rgb, np.array([0.299, 0.587, 0.114])), 0, 1)
    u = np.clip(np.dot(rgb, np.array([-0.14713, -0.28886, 0.436])), -0.492, 0.492)
    v = np.clip(np.dot(rgb, np.array([0.615, -0.51499, -0.10001])), -0.877, 0.877)
    yuv = rgb[:, :, :]
    yuv[:, :, 0] = y
    yuv[:, :, 1] = u
    yuv[:, :, 2] = v
    return yuv

def yuv2rgb(yuv):
    r = np.dot(yuv, np.array([1.0, 0.0, 1.13983]))
    g = np.dot(yuv, np.array([1.0, -0.39465, -0.58060]))
    b = np.dot(yuv, np.array([1.0, 2.03211, 0.0]))
    rgb = yuv[:, :, :]
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return np.clip(rgb, 0.0, 1.0) * 255.0

def find_marked_space(bw, marked):
    diff = marked - bw
    colored_mine = [set(zip(*np.nonzero(diff[:, :, i]))) for i in [0, 1, 2]]
    result_mine = colored_mine[0].union(colored_mine[1]).union(colored_mine[2])
    return result_mine

def find_neighbor(position_matrix, r):
    d = 1
    l1, h1 = max(r[0]-d, 0), min(r[0]+d, position_matrix.shape[0])
    l2, h2 = max(r[1]-d, 0), min(r[1]+d, position_matrix.shape[1])
    return position_matrix[l1:h1 + 1, l2:h2 + 1]

def generate_std2_matrix(Y):
    res = np.zeros(Y.shape, dtype='float32')
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            res[i, j] = np.square(np.std(find_neighbor(Y, [i, j])))
    return res

def calc_weight(r, S, Y, std2):
    result = []
    for s in S:
        result.append(-1 * np.exp(-1 * np.square(Y[r] - Y[s]) / 2 * std2[r]))
    return result

def generate_weight_matrix(Y):
    (height, width) = Y.shape[0:2]
    cart = cartesian([range(height), range(width)])
    cart_r = cart.reshape(height, width, 2) # cart_r[i, j] is [i, j]
    size = height * width
    xy2idx = np.arange(size).reshape(height, width) # linear rank of [i, j]
    W = sparse.lil_matrix((size, size)) # sparse matrix map (h, w) -> (h, w)
    std2_matrix = generate_std2_matrix(Y) # std2 matrix (h, w)
    for i in range(height):
        for j in range(width):
            current_index = xy2idx[i, j]
            neighbors = find_neighbor(cart_r, [i, j]).reshape(-1, 2)
            neighbors = [tuple(item) for item in neighbors] # list of [i, j] of current point
            neighbors.remove((i, j))
            neighbor_indexes = [xy2idx[pos] for pos in neighbors]
            current_weights = calc_weight((i, j), neighbors, Y, std2_matrix)
            W[current_index, neighbor_indexes] = np.asmatrix(current_weights)
    Wn = normalize(W, norm = 'l1', axis = 1)
    Wn[np.arange(size), np.arange(size)] = 1
    return Wn, xy2idx
    
def main(arguments):
    # ---------------- Data Preparation ------------------
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help='Specify the name of the black and white picture.', type=str, default='example')
    parser.add_argument('--marked', help='Specify the name of the marked picture', type=str, default='example_marked')
    parser.add_argument('--output', help='Specify the output name you want.', type=str, default='cahl_output')

    args = parser.parse_args(arguments)
    bw_name = args.input
    marked_name = args.marked
    out_name = args.output

    bw_file_path = os.path.join(PIC_DIR, bw_name + EXTENSION)
    marked_file_path = os.path.join(PIC_DIR, marked_name + EXTENSION)
    out_file_path = os.path.join(PIC_DIR, out_name + EXTENSION)
    
    bw_rgb = misc.imread(bw_file_path)
    marked_rgb = misc.imread(marked_file_path)
    
    bw = rgb2yuv(bw_rgb)
    Y = np.array(bw[:, :, 0], dtype='float64')
    marked = rgb2yuv(marked_rgb)
    (height, width) = Y.shape[0:2]
    size = height * width
    # ---------------- Find marked space ------------------
    colored = find_marked_space(bw_rgb, marked_rgb)
    # ---------------- Generate weight matrix ------------------
    Wrs, xy2idx = generate_weight_matrix(Y)
    Wrs = Wrs.tolil()
    for idx in [xy2idx[pos] for pos in colored]:
        Wrs[idx] = sparse.csr_matrix(([1.0], ([0], [idx])), shape=(1, size))

    # ---------------- Optimization ------------------
    LU = scipy.sparse.linalg.splu(Wrs.tocsc())
    
    b1 = marked[:, :, 1].flatten()
    b2 = marked[:, :, 2].flatten()

    x1 = LU.solve(b1)
    x2 = LU.solve(b2)

    sol = np.zeros(bw.shape)
    sol[:, :, 0] = Y
    sol[:, :, 1] = x1.reshape((height, width))
    sol[:, :, 2] = x2.reshape((height, width))
    sol_rgb = yuv2rgb(sol)

    misc.imsave(out_file_path, sol_rgb)
    print('Colorized picture saved to', out_file_path)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))