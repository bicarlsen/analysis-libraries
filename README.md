# Analysis Libraries

> python -m pip install bric-analysis-libraries 

An assortment of analysis libraries.

## Components
There are five components of the libraries divided by experiemnt type. Each component has data prep modules to prepare the data from different softwares into a canonical form. The prepared data can then be anlayzed with the analysis modules.

**Use**

To `import` the modules use the form
```python
from bric_analysis_libraries[.<component>] import <module>
# or
import bric_analysis_libraries[.<component>].<module>
```
where `<component>` is the name of the component (if needed) and `<module>` is the name of the module. Any modules in the Standard Component do not require a component name, while modules in all other components do.

**Examples**
```python
from bric_analysis_libraries import standard_functions as std
# or
import bric_analysis_libraries.standard_functions as std
```

```python
from bric_analysis_libraries.jv import aging_analysis as aging
# or
import bric_analysis_libraries.jv.aging_analysis as aging
```

---

### Standard Component
> No component requried

Contains standard functions.

#### Standard Functions
Provides standard functions.

#### Plot
Provides plotting functionality and helper functions.

---

### JV Component
> Component name: `jv`

Contains data prep and analysis packages for JV experiments.

#### Aging Data Prep
> Module name: `aging_data_prep`

Data prep from the a custom built stability lab for solar cell degradation.

#### Biologic Data Prep
> Module name: `biologic_data_prep`

Data prep

#### EC Lab Analysis
> Module name: `ec_lab_analysis`

Analysis of EC experiments

#### EC Lab Data Prep
> Module name: `ec_lab_data_prep`

Data prep of experiments form EC Lab.

#### Igor JV Data Prep
> Module name: `igor_jv_data_prep`

Data prep of JV experiments coming from the old IV setup.

#### JV Analysis
> Module name: `jv_analysis`

Analysis of JV experiments.

#### JV Data Prep
> Module name: `jv_data_prep`

Data prep for general JV data.

---

### PL Component
> Component name: `pl`

Contains data prep and analysis packages for PL experiments.

#### Andor Data Prep
> Module name: `andor_data_prep`

Data prep for PL experiments from Andor Solis software.

#### Lifespec Data Prep
> Module name: `lifespec_data_prep`

Data prep for Lifespec II TRPL instrument.

#### Ocean Optics Data Prep
> Module name: `ocean_optics_data_prep`

Data prep for PL experiments from Ocean Optics.

#### PL Analysis
> Module name: `pl_analysis`

Analysis of PL experiments.

#### PL Data Prep
> Module name: `pl_data_prep`

Data prep for general PL experiments.

#### TRPL Analysis
> Module name: `trpl_analysis`

Analysis for TRPL experiments.

#### Blackbody Analsysis
> Module name: `blackbody_analysis`

Methods for analyzing blackbody characteristics of PL spectra.

---

### SCAPS Component
> Moved to the [PySCAPS package](https://pypi.org/project/pyscaps/).

Functions for analyzing [SCAPS](https://scaps.elis.ugent.be/) simulation results.

#### Common
> Module name: `common`

Common functions with low level functionality.

#### General
> Module name: `gnr`

Formats general data into Pandas DataFrames.

#### JV
> Module name: `iv`

Formats IV data into Pandas DataFrames.

#### Energy Band
> Module name: `eb`

Formats energy band data into Pandas DataFrames.

#### Generation and Recombination
> Module name: `gen`

Formats generation and recombination and data into Pandas DataFrames.

#### Model
> Module name: `model`

For analyzing models.

---

### Misc Component
> Component name: `misc`

Contains other components.

#### Function Matcher
> Module name: `function_matcher`

Creates a linear combination of basis functions to match a target function as close as possible.

#### QSoft Data Prep
> Module name: `qsoft_data_prep` 

Data prep for QSoft quartz crystal microbalance acquisition software.

#### QCM Analysis
> Module name: `qcm_analysis`

Analysis of quartz crystal microbalance experiments.


#### Cary UV/Vis Absorption Spectrometer Data Prep
> Component name: `cary_absorption_data_prep`

Data prep for absorption spectra from a Cary UV/Vis spectrometer.

#### XRD Data Prep
> Component name: `xrd_data_prep`

Data prep for XRD spectra.

#### ARS Cryostat Data Prep
> Component name: `ars_cryostat_data_prep`

Data prep for measurements taken using the ARS cryostat.

---

### Utilities Component
> Component name: `utils`

#### Metadata
> Component name: `metadata`

Function for extracting metadata from files.