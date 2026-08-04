# PHT3D-FSP - A Python tool to support the use of PHT3D in FloPy

Authors: Stephan L. Seibert (stephan.seibert@uol.de) and Janek Greskowiak (janek.greskowiak@uol.de)

Cite as: Stephan L. Seibert, & Janek Greskowiak. (2024). PHT3D-FSP: PHT3D FloPy Supporting Package (v0.16). Zenodo. https://doi.org/10.5281/zenodo.7559750

This package is supporting the implementation of PHT3D (Prommer and Post, 2010) in FloPy (Bakker et al., 2016). PHT3D couples MT3DMS (Zheng and Wang, 1999) to the hydrogeochemical reaction code PHREEQC-2 (Parkhurst and Appelo, 1999). Visit www.pht3d.org for further information.

1.) Aim of PHT3D-FSP

Firstly, to provide an easy way of creating "phtd3d_ph.dat" using a .xlsx spreadsheet and defining reactants in it and, secondly, to provide an easy way of defining initial concentrations ("sconc") and initiating (time-variant) concentration data arrays for the MT3DMS-FloPy BTN and SSM packages, respectively, based on the species provided in the .xlsx spreadsheet.
  
2.) General instructions

Firstly, paste the file "pht3d_fsp.py" into your environment site-package folder, e.g., /home/user/anaconda3/envs/flopy/lib/python3.9/site-packages/
Alternatively, it might also be possible to simply store "pht3d_fsp.py" in the folder of the FloPy script.

Important note: the "pht3d_fsp.create()" function requires that the packages "numpy" and "pandas" are installed in the environment.

Secondly, within a flopy .py script, after having run a MODFLOW/SEAWAT simulation and, thereby, having produced "mt3d_link.ftl", the following line needs to be called:
	
	spec = pht3d_fsp.create(xlsx_path="./",xlsx_name="pht3d_species.xlsx",nlay=1,nrow=1,ncol=1,ph_os=2,ph_temp=25.0,pht3d_path='./',
              ph_asbin=0, ph_eps_aqu=1e-10, ph_ph=1e-3,ph_print=0,ph_cb_offset=0,ph_surf_calc_type="-diffuse_layer",write_ph="yes")

The pht3d_fsp.create function variables are as follows:

	xlsx_path -> define the path to the .xlsx spreadsheet (default: "./")
	xlsx_name -> define the name of the .xlsx spreadsheet (default: "pht3d_species.xlsx")
	nlay, nrow, ncol -> number of the model layer, row and column (default = "1")
	ph_os -> PHT3D flag for the operator-splitting scheme (default = "2")
	ph_temp -> PHT3D temperature used for chemical reactions with temperature dependence. A negative value refers to the absolute species number that represents temperature (default: "25.0")
	pht3d_path -> define the destination path for "pht3d_ph.dat" (default: "./")
 	ph_asbin -> PHT3D flag that determines if binary and ASCII outputs are generated (default = "0")
	ph_eps_aqu -> PHT3D activation/deactivation criterion for species concentrations (default = "1e-10")
	ph_ph -> PHT3D activation/deactivation criterion for pH (default = "1e-3")
	ph_print -> PHT3D flag to print additional PHREEQC-2 output (default = "0")
	ph_cb_offset -> PHT3D flag to indicate if charge imbalances of a solution are transported (default = "0")
	ph_surf_calc_type -> PHT3D flag to indicate which type of SCM calculation will be executed by PHREEQC-2 (default = "-diffuse_layer")
	write_ph -> define if "pht3d_ph.dat" shall be created. Note that the file is saved in the path of the .py script (default = "yes")

If pht3d_fsp.create() is run, default values will be applied. However, note that nlay, nrow and ncol always need to be defined according to the model set-up.

The function will read the "pht3d_species.xlsx" spreadsheet. Therefore, fill out the spreadsheet as follows:

	Column 01: define names of species (can be freely chosen, no link to "pht3d_datab.dat")
	Column 02: define initial concentrations of species [mol/L]
	Column 03: define species/mineral names, in line with species/mineral names in "pht3d_datab.dat"
	Column 04: [optional] for type "A"/"C" (compare 3.): define initial amount of reactant [moles]; for type "B": define PHREEQC-2 argument; for type "D": define mineral target saturation index
	Column 05: define species/mineral type, compare "A" to "H" below
	Column 06: define mobility of species as "mobile" or "immobile" (required for calculation of MT3DMS "ncomp" and "mcomp" variables)
	Column 07: [required for exchange reactions] define if species particpates in exchange reaction; state "yes" if it does
	Column 08: [required for exchange reactions] define exchange stoichiometry of species with exchange master species; not required for exchange (master) species
	Column 09: [optional for exchange reactions] define exchange master species ("X-" is assumed, if noting is defined); not required for exchange (master) species
	Column 10: [required for surface complexation reactions] define specific surface area of a surface [m2/g or m2/mol]
	Column 11: [required for surface complexation reactions] define mass of solid
	Column 12: [optional for surface complexation reactions] define pure phase or kinetic reactant if surface binding sites are to be coupled to this compound
	Column 13: [optional for surface complexation reactions] define type of coupled compound, i.e., "equilibrium_phase" or "kinetic_reactant"
	Column 14: [required for kinetic species "A"/"C"] define stoichimoetry of reaction associated with kinetic species
	Column 15-115: [required for kinetic species "A"/"C"/"G" if not already defined in "pht3d_datab.dat"] define parameters of kinetic species/mineral (this script holds space for up to 100 parameters for each kinetic species/mineral; table may be extended to include more parameters, but the script has to be adopted in this case)

3.) Definition of different PHT3D species/mineral types (Column 05 in .xlsx spreadsheet):

	A -> kinetic, non-LEA (Local Equilibrium Assumption) mobile species with rate expression
	B -> LEA aqueous species (note: pH and pe have to be defined as the two last species under type "B", and they have to be defined as immobile species, as they are internally calculated from total hydrogen and total oxygen mass balances, respectively)
	C -> kinetic, non-LEA immobile species with rate expression
	D -> LEA minerals
	E -> exchange species and/or exchange master species
	F -> surface master species
	G -> kinetic, non-LEA minerals with rate expression
	H -> kinetic, non-LEA surface complexation species with rate expression (currently not supported by PHT3D)

Important note: it is mandatory to keep the correct order of the species, i.e., species must be defined from A to H in the .xlsx spreadsheet. 

4.) Definition of "sconc" (BTN package) and (time-variant) "crch" and other ssm stress period data (SSM package)

- "sconc" variables need to be defined for each species in flopy. For this purpose, the function "pht3d_fsp.create.sconc_btn" creates a text string, which passes initial concentration arrays on for all species defined in the .xlsx spreadsheet. "pht3d_fsp.create.sconc_btn" needs to be called in the flopy BTN package (see 5. below). While the individual initial concentration arrays for each species, i.e., "spec['SPECIES']", are filled with the values defined in Column 02 of the .xlsx spreadsheet, the arrays might be modified as desired by re-defining "spec['SPECIES']". Note that it is mandatory that "nlay", "nrow" and "ncol" are correctly defined when calling "pht3d_fsp.create()" (see 2.).

- "crch" variables need to be defined for each species in flopy, in case the RCH package is used in combination with the SSM package. For this purpose, the function "pht3d_fsp.create.crch_ssm" creates a text string, which passes concentration array names on for all species defined in the .xlsx spreadsheet, i.e., "rch_spec['SPECIES']", to the flopy SSM package (see 5. below). Therefore, these data arrays need to be manually created in flopy prior to calling the function "pht3d_fsp.create.crch_ssm" in the SSM package (see 5. below).

- depending on the type of boundary condition, further stress period data needs to be defined for the SSM package, which is called by the flopy SSM package via the variable "stress_period_data" (see 5.) This is, for example, the case if the WEL or CHD packages shall be used by PHT3D.

5a.) Call of BTN and SSM packages

To invoke the "ncomp", "mcomp", "sconc_btn", "crch_ssm" and "ssm_data" variables, the flopy BTN and SSM packages need to be called as follows:

	BTN --> exec(f'btn = flopy.mt3d.Mt3dBtn([ADD OTHER PACKAGE VARIABLES HERE], ncomp={pht3d_fsp.create.ncomp}, mcomp={pht3d_fsp.create.mcomp}, {pht3d_fsp.create.sconc_btn})')

	SSM --> exec(f'ssm = flopy.mt3d.Mt3dSsm([ADD OTHER PACKAGE VARIABLES HERE], stress_period_data=ssm_data, {pht3d_fsp.create.crch_ssm})')

5b.) Further functions

The following additional functions are available:

	"pht3d_fsp.create.species_no" -> yields the total number of species defined in the .xlsx file

6.) Example

The example "pht3d_fsp_example.ipynb" is provided in the subfolder "./example" to showcase the application of PHT3D-FSP in a FloPy script. Also, compare "pht3d_fsp_example.ipynb" to see how the RCH and SSM stress period data was defined. The example requires the provided PHREEQC-2 database, i.e., "pht3d_datab.dat", as well as the PHT3D-FSP species spreadsheet, i.e., "pht3d_species_example.xlsx".

7.) Funding

This work was realized in the Deutsche Forschungsgemeinschaft (DFG) project Reactive Transport (GR 4514/3-1) within the research unit FOR 5094: The dynamic deep subsurface of high-energy beaches (DynaDeep).

8.) License

This work is published under the GNU GENERAL PUBLIC LICENSE Version 3.

9.) Final remark

This is a preliminary file version. Contact the authors in case you experience errors.

Literature:

Bakker, Mark, Post, Vincent, Langevin, C. D., Hughes, J. D., White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733–739, https://doi.org/10.1111/gwat.12413

Parkhurst, D. L., & Appelo, C. a. J. (1999). User’s guide to PHREEQC (Version 2): A computer program for speciation, batch-reaction, one-dimensional transport, and inverse geochemical calculations. In Water-Resources Investigations Report (No. 99–4259). U.S. Geological Survey. https://doi.org/10.3133/wri994259

Prommer, H., & Post, V. (2010). A Reactive Multicomponent Transport Model for Saturated Porous Media (p. 200). www.pht3d.org

Zheng, C., & Wang, P. (1999). MT3DMS: A Modular Three-Dimensional Multispecies Transport Model for Simulation of Advection, Dispersion, and Chemical Reactions of Contaminants in Groundwater Systems; Documentation and User’s Guide.
