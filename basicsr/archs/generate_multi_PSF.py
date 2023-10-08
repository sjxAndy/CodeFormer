import numpy as np
from math import ceil
import matplotlib.pyplot as plt
# from motion_blur.generate_trajectory import Trajectory
from basicsr.archs.generate_trajectory import Trajectory


class MultiPSF(object):
    def __init__(self, canvas=None, num=1):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        self.trajectory = []
        for i in range(num):
            self.trajectory.append(Trajectory(canvas=canvas, expl=0.005).fit(show=False, save=False).x)

        self.fraction = [1]
        self.PSFnumber = len(self.fraction)
        self.iters = [len(self.trajectory[i]) for i in range(len(self.trajectory))]
        self.PSFs = []

    def fit(self):

        for k in range(len(self.trajectory)):
            PSF = np.zeros(self.canvas)

            triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
            triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
            for j in range(self.PSFnumber):
                if j == 0:
                    prevT = 0
                else:
                    prevT = self.fraction[j - 1]

                for t in range(len(self.trajectory[k])):
                    # print(j, t)
                    if (self.fraction[j] * self.iters[k] >= t) and (prevT * self.iters[k] < t - 1):
                        t_proportion = 1
                    elif (self.fraction[j] * self.iters[k] >= t - 1) and (prevT * self.iters[k] < t - 1):
                        t_proportion = self.fraction[j] * self.iters[k] - (t - 1)
                    elif (self.fraction[j] * self.iters[k] >= t) and (prevT * self.iters[k] < t):
                        t_proportion = t - (prevT * self.iters[k])
                    elif (self.fraction[j] * self.iters[k] >= t - 1) and (prevT * self.iters[k] < t):
                        t_proportion = (self.fraction[j] - prevT) * self.iters[k]
                    else:
                        t_proportion = 0

                    # m2 = int(np.minimum(self.canvas[1] - 1, np.maximum(1, np.math.floor(self.trajectory[t].real))))
                    m2 = int(np.minimum(self.canvas[1] - 2, np.maximum(0, np.math.floor(self.trajectory[k][t].real))))
                    M2 = int(m2 + 1)
                    # m1 = int(np.minimum(self.canvas[0] - 1, np.maximum(1, np.math.floor(self.trajectory[t].imag))))
                    m1 = int(np.minimum(self.canvas[0] - 2, np.maximum(0, np.math.floor(self.trajectory[k][t].imag))))
                    M1 = int(m1 + 1)

                    PSF[m1, m2] += t_proportion * triangle_fun_prod(
                        self.trajectory[k][t].real - m2, self.trajectory[k][t].imag - m1
                    )
                    PSF[m1, M2] += t_proportion * triangle_fun_prod(
                        self.trajectory[k][t].real - M2, self.trajectory[k][t].imag - m1
                    )
                    PSF[M1, m2] += t_proportion * triangle_fun_prod(
                        self.trajectory[k][t].real - m2, self.trajectory[k][t].imag - M1
                    )
                    PSF[M1, M2] += t_proportion * triangle_fun_prod(
                        self.trajectory[k][t].real - M2, self.trajectory[k][t].imag - M1
                    )

                self.PSFs.append(PSF / (self.iters[k]))


        return self.PSFs


def plot_canvas(PSFs, path_to_save):
    if len(PSFs) == 0:
        raise Exception("Please run fit() method first.")
    else:
        plt.close()
        fig, axes = plt.subplots(1, len(PSFs), figsize=(10, 10))
        for i in range(len(PSFs)):
            axes[i].imshow(PSFs[i], cmap='gray')
    plt.savefig(path_to_save)


if __name__ == '__main__':
    # trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
    #     psf = PSF(canvas=64, trajectory=trajectory).fit()
    psf = MultiPSF(canvas=19, num=100)
    import time
    start = time.time()
    psfs = psf.fit()
    print('time:', time.time() - start)
    print(len(psfs), psfs[0].shape, np.sum(psfs) / len(psfs), np.sum(psfs[0]), np.sum(psfs[1]), np.sum(psfs[2]), np.sum(psfs[3]))
    # plot_canvas(psfs, 'tmp/psf41.png')
    # psf = PSF(canvas=128, path_to_save='tmp/psf.png')
    # psf.fit(show=False, save=True)