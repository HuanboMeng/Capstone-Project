import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
import imageio
import io
import time
plot_every = 25 # plot only every n iterations 
start_time = time.time()
def distance(x1, y1, x2, y2):
    return cp.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Given points
# xa = cp.array([100.      , 150.0668  , 201.132928, 251.759784, 299.983238,
#        249.249078, 198.877536, 148.950948, 100.      ])

# ya = cp.array([200.      , 215.343074, 214.400412, 208.73658 , 199.748558,
#        196.290098, 193.249184, 191.54748 , 200.      ])

# Separating the points into x and y coordinates

xa = cp.array([100, 300, 300, 100])
ya = cp.array([300, 350, 250, 300])
def is_inside(xp,yp, xa, ya):
    cnt = 0
    for i in range(len(xa)-1):
        x1,x2 = xa[i], xa[i+1]
        y1,y2 = ya[i], ya[i+1]
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            cnt += 1

    return cnt%2 == 1


def main():
    frames = []
    # sizes of Lattice 
    Nx = 1000 # 400 Lattice in x coordinate
    Ny = 600 # 100 Lattice in y coordinate
    # define kinematic viscosity tau
    tau= 0.53
    # define amount of iterations in time
    Nt = 30000
    # lattice speeds and weights
    NL = 9
    # direction of descrete velocitiers in order of Sudoku
    # store in GPU
    # Sudoku order 5->2->3->6->9->8->7->4->1
    # node order is 0,1,2,3,4,5,6,7,8 start at zero
    cxs = cp.array([0,0,1,1, 1, 0,-1,-1,-1])
    cys = cp.array([0,1,1,0,-1,-1,-1, 0, 1])
    # weight
    weights = cp.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # initial condition for mesoscopic velocities
    # we need three dimension array, the first two dimension is nodes coordinate, the last is velocities
    # the random is inconsistencies in the fluid to make it more realistic
    F = cp.ones((Ny, Nx, NL)) + .01 * cp.random.rand(Ny, Nx, NL)
    # we need apply a velocity to every single lattice(cell)
    F[:,:,3] = 2 # apply 2.3 on lattice_6, the right mid one

    # define the cylinder
    cylinder = cp.full((Ny,Nx), False)
    # cp.full((2,2),False) = array([[False, False],[False, False]])
    # iterate every point in the space, 
    # and check whether points are a certain distance away from the cylinder(whether its' within that radius)
    for y in range(0, Ny):
        for x in range(0, Nx):
            # xc1 = Nx//4 #the position base on the grid
            # yc1 = Ny//2 #the position base on the grid can be adjusted after
            # if distance(xc1, yc1, x, y) < 13: # 13 is the distance between center of cylinder and lattice(cell)
            # #     #change 13 to any value will reuslt the cross-area(boundaries) of cylinder changing
            #      cylinder[y][x] = True
            if is_inside(x,y, xa,ya):
                cylinder[y][x] = True
    # main loop
    for it in range(Nt):
        # print(it)
        # loop for each lattice(cell)
        # go through every single node and roll its in the direction of its' corresponding discrete velocity
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:,:,i] = cp.roll(F[:,:,i],cx,axis = 1)
            F[:,:,i] = cp.roll(F[:,:,i],cy,axis = 0)
        # calculate the collision with our cylinder
        # get all of the points where the velocity is inside our cylinder 
        # and invert the velocity to opposite direction node
        # such that node 3 is opposite to node 7
        bndryF = F[cylinder , :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]] # this is the opposite node order
        
        # ===== fluid variable =====
        # density
        rho = cp.sum(F, axis = 2) # add dim 3 is the sum of velocity
        
        # momentum 
        ux = cp.sum(F * cxs, axis = 2) / rho
        uy = cp.sum(F * cys, axis = 2) / rho
        # set fluid movement within a boundary so all of those are going to be set to zero
        F[cylinder, :] = bndryF
        ux[cylinder] = 0
        uy[cylinder] = 0

        # collision
        Feq = cp.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:,:,i] = rho * w * (
                1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy)**2 / 2 - 3 * (ux**2 + uy**2) / 2
            )
        F = F + -(1/tau) *(F - Feq)
        if it % plot_every == 0:     
            magnitude = cp.sqrt(ux**2 + uy**2)
            # Convert the magnitude array to a NumPy array if necessary for plotting
            magnitude_np = cp.asnumpy(magnitude)
            plt.imshow(magnitude_np,interpolation='sinc',cmap = 'plasma') # show the magnitude of velocity(taking the sqrt of the component of velocity)
            
            # # the magnitude of vectors 
            # ux_np = 0.03 * cp.asnumpy(ux)
            # uy_np = 0.03 * cp.asnumpy(uy)
            # # interval of each vector
            # skip = 20
            # X, Y = np.meshgrid(np.arange(0, Nx, skip), np.arange(0, Ny, skip))
            # plt.quiver(X, Y, ux_np[::skip, ::skip], uy_np[::skip, ::skip], color='r')
            # plt.pause(.01) # to pause for a second give us some time to look at the graph
            
             # Save plot to a temporary buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frame = imageio.imread(buf)
            frames.append(frame)
            plt.close()

# Save frames as a GIF
    output_path = 'velocity_magnitude5.gif'

# Use the defined path when saving the GIF
    imageio.mimsave(output_path, frames, fps=10)



if __name__ ==  "__main__":
    main()
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")