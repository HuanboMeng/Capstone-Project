# Gray Scale
from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
from PIL import Image
import time
start_time = time.time()
A = Image.open('p2.png')
A = A.resize((150, 10))
A = cp.array(A)
A = cp.mean(A[:, :, :3], axis=2)  
A = A - cp.min(A) 
A = A / cp.max(A)
# A = A.reshape(A.shape[0], A.shape[1], 1)
# A = np.repeat(A, 3, axis=2)
# B  = np.repeat(A[:, :, np.newaxis], 3, axis=2)
# # plt.imshow(B)
# np.set_printoptions(threshold=np.inf)


def apply_stencil(Phi, i, j):
    Phi_xx = Phi[i+1, j] - 2 * Phi[i, j] + Phi[i-1, j]
    Phi_yy = Phi[i, j+1] - 2 * Phi[i, j] + Phi[i, j-1]
    Phi_xy = (Phi[i+1, j+1] - Phi[i-1, j+1] - Phi[i+1, j-1] + Phi[i-1, j-1]) / 4
    Phi_x = (Phi[i+1, j] - Phi[i-1, j]) / 2
    Phi_y = (Phi[i, j+1] - Phi[i, j-1]) / 2
    kappa_0 = (Phi_xx * Phi_y**2 - 2 * Phi_x * Phi_y * Phi_xy + Phi_yy * Phi_x**2) / (cp.power(Phi_x**2 + Phi_y**2, 1.5) + 1e-6)
    kappa = cp.maximum(cp.minimum(kappa_0, 5), -5)
    return kappa

def H(x):
    return 1/2 * (1 + 2 / cp.pi * np.arctan(x))
    
def delta(x):
    return 1 / (cp.pi * (x**2 + 1))

def initial_value(sz):
    m, n = sz
    x = cp.arange(1, m+1)
    y = cp.arange(1, n+1)
    X, Y = cp.meshgrid(x, y, indexing='ij')
    Phi = cp.sqrt((X - m/2)**2 + (Y - n/2)**2) - 0.4 * n
    return Phi

def coefficients(Phi, A):
    c1 = cp.sum(A * H(Phi)) / cp.sum(H(Phi))
    c2 = cp.sum(A * (1 - H(Phi))) / cp.sum(1 - H(Phi))
    return c1, c2

def update(Phi, A):
    m, n = A.shape
    DeltaPhi = cp.zeros((m, n))
    c1, c2 = coefficients(Phi, A)
    for i in range(1, m-1):
        for j in range(1, n-1):
            kappa = apply_stencil(Phi, i, j)
            DeltaPhi[i, j] = 100 * delta(Phi[i, j]) * (0.2 * kappa - (A[i, j] - c1)**2 + (A[i, j] - c2)**2)
    return DeltaPhi

def image_segment(A, maxiter=10000):
    Phi = initial_value(A.shape)
    for i in range(maxiter):
        DeltaPhi = update(Phi, A)
        if np.max(cp.abs(DeltaPhi)) < 0.02:
            break
        Phi += DeltaPhi
    return Phi


# Assuming you load an image using a method that returns a CuPy array.
# If using plt.imread, you must convert it to a CuPy array:

# 创建一个50x50的零矩阵
# matrix_size = 50
# A = np.ones((matrix_size, matrix_size))

# # 设置圆心(h, k)和半径r
# h, k, r = 25, 25, 20

# # 填充圆形的值
# for x in range(matrix_size):
#     for y in range(matrix_size):
#         if (x - h) ** 2 + (y - k) ** 2 <= r ** 2:
#             A[x, y] = 0

# # Convert to grayscale
# if A.ndim == 3:
#     A = np.mean(A[:, :, :3], axis=2)  # Assuming A has three color channels
Phi = image_segment(A)

# Visualize the results
plt.figure()
plt.imshow(cp.repeat(A[:, :, cp.newaxis], 3, axis=2).get())
plt.contour(Phi.get(), levels=[0.])
plt.savefig('CVS2.png')
plt.show()
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")