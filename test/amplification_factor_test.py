import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import pytest

@pytest.fixture
def test_time_data():
    fixed_ts = np.loadtxt('./test/test_sis_ts.txt')
    fixed_F_tilde = np.loadtxt('./test/test_sis_F_tilde.txt')

    return fixed_ts, fixed_F_tilde

@pytest.fixture
def test_freq_data():
    fixed_fs = np.loadtxt('./test/test_sis_ws.txt')/(2*np.pi)
    fixed_Fws = np.loadtxt('./test/test_sis_Fws.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})

    return fixed_fs, fixed_Fws

def test_integrator_gpu(sis_amp, test_time_data):
    """
    Testing time integration of standard SIS model with gpu
    """
    amp_ts, amp_F_tilde = sis_amp.integrator(gpu=True)
    amp_ts, amp_F_tilde = np.asarray(amp_ts), np.asarray(amp_F_tilde)

    fixed_ts, fixed_F_tilde = test_time_data

    assert np.allclose([amp_ts, amp_F_tilde], [fixed_ts, fixed_F_tilde])

def test_integrator_cpu(sis_amp, test_time_data):
    """
    Testing time integration of standard SIS model with cpu
    """
    amp_ts, amp_F_tilde = sis_amp.integrator(gpu=False)
    amp_ts, amp_F_tilde = np.asarray(amp_ts), np.asarray(amp_F_tilde)

    fixed_ts, fixed_F_tilde = test_time_data
    
    assert np.allclose([amp_ts, amp_F_tilde], [fixed_ts, fixed_F_tilde])

def test_freq_amp(sis_amp, test_time_data, test_freq_data):
    """
    Testing freqeuncy sis_amp factor with Fourier transform
    """
    fixed_ts, fixed_F_tilde = test_time_data
    sis_amp.importor(time=True, ts=fixed_ts, F_tilde=fixed_F_tilde)
    amp_fs, amp_Fws = sis_amp.fourier()

    fixed_fs, fixed_Fws = test_freq_data

    assert np.allclose([amp_fs, np.abs(amp_Fws)], [fixed_fs, np.abs(fixed_Fws)])

def test_plot_time(sis_amp, test_time_data):
    """
    Testing plotting in time domain
    """

    fixed_ts, fixed_F_tilde = test_time_data

    from scipy.signal import savgol_filter
    F_smooth = savgol_filter(fixed_F_tilde, 51, 3)
    
    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    sis_amp.importor(time=True, ts=fixed_ts, F_tilde=fixed_F_tilde)
    ax = sis_amp.plot_time()

    x_plot, y_plot = ax.lines[0].get_xydata().T
    np.testing.assert_array_equal(y_plot, F_smooth)

def test_plot_freq_abs(sis_amp, test_freq_data):
    """
    testing plotting in frequency domain of absolute value
    """

    fixed_fs, fixed_Fws = test_freq_data

    from scipy.signal import savgol_filter
    Fa_fil = savgol_filter(np.abs(fixed_Fws), 51, 3)
    
    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    sis_amp.importor(freq=True, fs=fixed_fs, Fws=fixed_Fws)
    ax = sis_amp.plot_freq(abs=True, smooth=True)

    x_plot, y_plot = ax.lines[0].get_xydata().T
    np.testing.assert_array_equal(y_plot, Fa_fil)

def test_plot_freq_pha(sis_amp, test_freq_data):
    """
    testing plotting in frequency domain of argument
    """

    fixed_fs, fixed_Fws = test_freq_data

    from scipy.signal import savgol_filter
    Fp_fil = savgol_filter(np.angle(fixed_Fws), 51, 3)

    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    sis_amp.importor(freq=True, fs=fixed_fs, Fws=fixed_Fws)
    ax = sis_amp.plot_freq(pha=True, smooth=True)

    x_plot, y_plot = ax.lines[0].get_xydata().T
    np.testing.assert_array_equal(y_plot, Fp_fil)
