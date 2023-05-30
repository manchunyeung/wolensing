import numpy as np

class SIS():

    def hessian(x, y, theta_E, center_x=0, center_y=0):
        print(x, y, 'x and y')
        x_shift = x - center_x
        y_shift = y - center_y
        theta_E = 8.052888061582411e-07 
        print(x_shift, y_shift, 'shift')
        R = (x_shift*x_shift + y_shift*y_shift)**(3./2)
        if isinstance(R, int) or isinstance(R, float):
            print('if')
            print(R, max(0.000001, R), 'R and max')
            prefac = theta_E / max(0.000001, R)
            print(prefac)
            print(theta_E, theta_E * 1**6, 'theta')
        else:
            print('else')
            prefac = np.empty_like(R)
            r = R[R > 0]  # in the SIS regime
            prefac[R == 0] = 0.
            prefac[R > 0] = theta_E / r

        print(prefac, 'prefac')
        f_xx = y_shift*y_shift * prefac
        f_yy = x_shift*x_shift * prefac
        f_xy = -x_shift*y_shift * prefac
        return f_xx, f_xy, f_xy, f_yy
