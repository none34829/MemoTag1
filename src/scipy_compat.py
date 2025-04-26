"""
Compatibility module for scipy and librosa to handle version differences.
"""

import scipy.signal
import numpy as np
import importlib.util

# Check if hann is already defined
if not hasattr(scipy.signal, 'hann'):
    # In newer scipy versions, hann has been renamed to windows.hann
    if hasattr(scipy.signal, 'windows') and hasattr(scipy.signal.windows, 'hann'):
        # Create alias for backwards compatibility
        scipy.signal.hann = scipy.signal.windows.hann
    else:
        # If it's not available, create a simple implementation
        def hann(M, sym=True):
            """
            Create a Hann window.
            
            Parameters
            ----------
            M : int
                Number of points in the output window.
            sym : bool, optional
                Whether the window is symmetric. Default is True.
                
            Returns
            -------
            w : ndarray
                The window.
            """
            if M < 1:
                return np.array([])
            if M == 1:
                return np.ones(1)
            
            if not sym:
                # non-symmetric window
                n = np.arange(M)
                w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
            else:
                # symmetric window
                n = np.arange(M)
                w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
            
            return w
        
        # Add our implementation to scipy.signal
        scipy.signal.hann = hann

# Verify it worked
assert hasattr(scipy.signal, 'hann'), "Failed to add hann function to scipy.signal"

# Fix for librosa.util.peak_pick
try:
    import librosa.util

    # The original implementation, modified to handle newer librosa versions
    # Check the peak_pick function signature
    import inspect
    peak_pick_signature = inspect.signature(librosa.util.peak_pick)
    
    # If the function now expects a single argument (in newer versions),
    # patch it to accept the older style multiple arguments
    if len(peak_pick_signature.parameters) == 1:
        original_peak_pick = librosa.util.peak_pick
        
        def patched_peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
            """
            Wrapper for peak_pick that handles the older API style with multiple arguments.
            """
            # Convert the multiple arguments to the format expected by the new API
            return original_peak_pick({
                'x': x,
                'pre_max': pre_max,
                'post_max': post_max,
                'pre_avg': pre_avg,
                'post_avg': post_avg,
                'delta': delta,
                'wait': wait
            } if isinstance(original_peak_pick.__code__.co_varnames[0], dict) else x)
        
        # Replace the library function with our patched version
        librosa.util.peak_pick = patched_peak_pick
        print("Patched librosa.util.peak_pick for compatibility")
except Exception as e:
    print(f"Failed to patch librosa.util.peak_pick: {e}")
