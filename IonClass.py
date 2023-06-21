import numpy as np

class Ion:

    def __init__(self, name, size,color):
        self.name          = name
        self.list_lines    = []
        self.spectrum      = np.ones(size)
        self.v             = [0, np.nan]
        self.b             = [20, np.nan]
        self.fcov          = [0.99, np.nan]
        self.color         = color
        if name == 'HI':
            self.logn = [18, np.nan]
        else:
            self.logn = [14, np.nan]

    def update_parameters(self, fitted_parameters):
        """
        Update the parameter values of the absorption line profile based on the fitted parameters obtained from a curve fit.

        Args:
            fitted_parameters (object): A Parameters object obtained from the curve fit with new values for the fitting parameters.

        Returns:
            None

        """
        # Update the velocity parameters
        self.v[0] = fitted_parameters.params['v'+self.name].value
        if(fitted_parameters.params['v'+self.name].stderr is not None):
            self.v[1] = fitted_parameters.params['v'+self.name].stderr
        else:
            self.v[1] = np.nan

        # Update the Doppler broadening parameters
        self.b[0] = fitted_parameters.params['b'+self.name].value
        if(fitted_parameters.params['b'+self.name].stderr is not None):
            self.b[1] = fitted_parameters.params['b'+self.name].stderr
        else:
            self.b[1] = np.nan

        # Update the column density parameters
        self.logn[0] = fitted_parameters.params['logn'+self.name].value
        if(fitted_parameters.params['logn'+self.name].stderr is not None):
            self.logn[1] = fitted_parameters.params['logn'+self.name].stderr
        else:
            self.logn[1] = np.nan

        # Update the covering fraction parameters
        self.fcov[0] = fitted_parameters.params['fcov'+self.name].value
        if(fitted_parameters.params['fcov'+self.name].stderr is not None):
            self.fcov[1] = fitted_parameters.params['fcov'+self.name].stderr
        else:
            self.fcov[1] = np.nan

        return

    def update_MW_parameters(self, fitted_parameters):
        """
        Update the parameter values of the absorption line profile based on the fitted parameters obtained from a curve fit.

        Args:
            fitted_parameters (object): A Parameters object obtained from the curve fit with new values for the fitting parameters.

        Returns:
            None

        """
        # Update the velocity parameters
        self.v[0] = fitted_parameters.params['vMW'+self.name].value
        if(fitted_parameters.params['vMW'+self.name].stderr is not None):
            self.v[1] = fitted_parameters.params['vMW'+self.name].stderr
        else:
            self.v[1] = np.nan

        # Update the Doppler broadening parameters
        self.b[0] = fitted_parameters.params['bMW'+self.name].value
        if(fitted_parameters.params['bMW'+self.name].stderr is not None):
            self.b[1] = fitted_parameters.params['bMW'+self.name].stderr
        else:
            self.b[1] = np.nan

        # Update the column density parameters
        self.logn[0] = fitted_parameters.params['lognMW'+self.name].value
        if(fitted_parameters.params['lognMW'+self.name].stderr is not None):
            self.logn[1] = fitted_parameters.params['lognMW'+self.name].stderr
        else:
            self.logn[1] = np.nan

        # Update the covering fraction parameters
        self.fcov[0] = fitted_parameters.params['fcovMW'+self.name].value
        if(fitted_parameters.params['fcovMW'+self.name].stderr is not None):
            self.fcov[1] = fitted_parameters.params['fcovMW'+self.name].stderr
        else:
            self.fcov[1] = np.nan

        return

    def update_spectrum(self, spectrum):
        self.spectrum = spectrum

    def print_ion_parameters(self):
        """
        Prints the parameters
        """
        print(f"Line name: {self.name}")
        print(f"Velocity (v): {self.v[0]:.1f} +- {self.v[1]:.1f} km/s")
        print(f"Doppler parameter (b): {self.b[0]:.1f} +- {self.b[1]:.1f} km/s")
        print(f"Column density (log N): {self.logn[0]:.2f} +- {self.logn[1]:.2f} cm^-2")
        print(f"Covering fraction (f_cov): {self.fcov[0]:.3f} +- {self.fcov[1]:.2f}")
