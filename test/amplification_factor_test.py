import numpy as np
import amplification.amplificaiton_factor as af

def test_integrator_gpu(amplification):
    """
    Testing time integration of standard SIS model with gpu
    """
    amp_ts, amp_F_tilde = amplification.integrator(gpu=True)

    fixed_ts = np.loadtxt('./test_sis_ts.txt')
    fixed_F_tilde = np.loadtxt('test_sis_F_tilde.txt')

    assert np.allclose([amp_ts, amp_F_tilde], [fixed_ts, fixed_F_tilde])

def test_integrator_cpu(amplification):
    """
    Testing time integration of standard SIS model with cpu
    """
    amp_ts, amp_F_tilde = amplification.integrator(gpu=False)

    fixed_ts = np.loadtxt('./test_sis_ts.txt')
    fixed_F_tilde = np.loadtxt('test_sis_F_tilde.txt')

    assert np.allclose([amp_ts, amp_F_tilde], [fixed_ts, fixed_F_tilde])

def test_freq_amp(amplification):
    """
    Testing freqeuncy amplification factor with Fourier transform
    """
    amp_fs, amp_Fws = amplification.fourier()

    fixed_fs = np.loadtxt('./test_sis_ws.txt')
    fixed_Fws = np.loadtxt('./test_sis_Fws.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})

    assert np.allclose([amp_fs, amp_Fws], [fixed_fs, fixed_Fws])

def test_plot_time(amplification):
    """
    Testing plotting in time domain
    """

    fixed_ts = np.loadtxt('./test_sis_ts.txt')
    fixed_F_tilde = np.loadtxt('test_sis_F_tilde.txt')

    from scipy.signal import savgol_filter
    F_smooth = savgol_filter(fixed_F_tilde, 51, 3)
    
    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    amplification.importer(time=True, ts=fixed_ts, F_tilde=fixed_F_tilde)
    ax = amplification.plot_time()

    x_plot, y_plot = ax.lines[0].get_xydata().T
    np.testing.assert_array_equal(y_plot, Fa_fil)

def test_plot_freq_abs(amplification):
    """
    testing plotting in frequency domain of absolute value
    """

    fixed_fs = np.loadtxt('./test_sis_ws.txt')
    fixed_Fws = np.loadtxt('./test_sis_Fws.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})

    from scipy.signal import savgol_filter
    Fa_fil = savgol_filter(np.abs(fixed_Fws), 51, 3)
    
    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    amplification.importer(freq=True, fs=fixed_fs, Fws=fixed_Fws)
    ax = amplification.plot_freq(abs=True)

    x_plot, y_plot = ax.lines[0].get_xydata().T
    np.testing.assert_array_equal(y_plot, Fa_fil)

def test_plot_freq_pha(amplification):
    """
    testing plotting in frequency domain of argument
    """

    fixed_fs = np.loadtxt('./test_sis_ws.txt')
    fixed_Fws = np.loadtxt('./test_sis_Fws.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})

    from scipy.signal import savgol_filter
    Fp_fil = savgol_filter(np.angle(fixed_Fws), 51, 3)
    
    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    amplification.importer(freq=True, fs=fixed_fs, Fws=fixed_Fws)
    ax = amplification.plot_freq(pha=True)

    x_plot, y_plot = ax.lines[0].get_xydata().T
    np.testing.assert_array_equal(y_plot, Fp_fil)
