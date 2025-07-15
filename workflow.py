import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import itertools
from mf6rtm import utils, mup3d
from collections import defaultdict
from flopy.utils.gridintersect import GridIntersect
from collections.abc import Iterable

datadir = os.path.join("data")
dis_ws = os.path.join(datadir, 'dis')
props_ws = os.path.join(datadir, 'props')


def append_values_to_inner_lists(d, values, *, in_place=False):
    """
    Append a single value or all values from an iterable to every inner list
    inside a {key: list[list]} dictionary.

    Parameters
    ----------
    d : dict
        Your nested‑list dictionary.
    values : any or Iterable
        * If `values` is not an Iterable (or is str/bytes), it’s treated as a
          single item and appended once.
        * If `values` is an Iterable (list/tuple/set/range…), each element is
          appended in order.
    in_place : bool, default False
        True  → modify `d` directly and return it.  
        False → leave `d` unchanged and return a *new* dictionary.

    Returns
    -------
    dict
        The dictionary with updated inner lists.
    """
    # Decide whether to work on the original or a shallow copy
    target = d if in_place else {k: [lst[:] for lst in v] for k, v in d.items()}

    # Determine if we have “one thing” or “many things”
    is_iterable = (
        isinstance(values, Iterable) and
        not isinstance(values, (str, bytes))  # treat strings/bytes as scalars
    )

    for outer in target.values():
        for inner in outer:
            if is_iterable:
                inner.extend(values)   # add every element in order
            else:
                inner.append(values)   # add the single value

    return target

def run_model(sim):
    import pyemu
    pyemu.os_utils.run('mf6', cwd=sim.sim_path)

def reformat_arrays(input_file, rows, cols, save_file=False):
    # Read the content of the file
    with open(input_file, 'r') as f:
        content = f.read()

    # Flatten the values into a 1D array and remove spaces or NaNs
    values = [val for val in content.split() if val.strip() and val.lower() != 'nan']

    # Convert values to a numpy array
    values_array = np.array(values)

    # Ensure the array can be reshaped to the desired shape
    if len(values_array) != rows * cols:
        raise ValueError("The number of values does not match the specified shape")

    # Reshape the array into the specified shape
    reshaped_array = values_array.reshape((rows, cols))

    # Save the reshaped array to a new file with _format extension
    if save_file:
        root, ext = os.path.splitext(input_file)
        output_file = f"{root}_formatted{ext}"
        print(f"Saving reformated file to {output_file}")
        np.savetxt(output_file, reshaped_array, fmt='%s')
    
    # return reshaped array
    return reshaped_array

def calculate_vertical_conductivity(vcont, thickness, nlay, nrow, ncol):
    nlay, nrow, ncol = vcont.shape
    k33 = np.zeros((nlay, nrow, ncol))

    for i in range(nlay - 1):
        for j in range(nrow):
            for k in range(ncol):
                k33[i, j, k] = vcont[i, j, k] * ((thickness[i, j, k] / 2) + (thickness[i + 1, j, k] / 2))
    # Set the vertical hydraulic conductivity of layer 12 equal to layer 11
    k33[-1] = k33[-2]
    
    return k33

def make_obs_pack(gwf):
    ix = GridIntersect(gwf.modelgrid)
    obsloc = pd.read_csv(os.path.join(datadir, "obs_loc.csv"))

    obs_list=[]

    for obsid in obsloc.obsid.unique():
        x,y = obsloc.loc[obsloc.obsid==obsid,['x','y']].values[0]
        cellid = ix.intersect([(x,y)],shapetype="point").cellids
        if len(cellid)==0:
            print(f"{obsid} not in model domain")
            continue
        else:
            cellid = cellid[0]
            print(f"{obsid} is in model domain") 
        obs_layer = int(obsloc.loc[obsloc.obsid==obsid,'layer'].values[0])
        obs_list.append((obsid, 'concentration', (obs_layer, cellid[0], cellid[1])))

    # obs_recarray = {'obs.head.sim.csv':obs_list}
    obs_recarray = {f'obs_{gwf.name}.csv':obs_list}
    print(obs_list)
    obs_package = flopy.mf6.ModflowUtlobs(gwf, 
                                        digits=0, #print_input=True,
                                        pname=f'obs_{gwf.name}',
                                        continuous=obs_recarray)
    return obs_package

def make_dis(gwf):
    # Define model parameters for the dummy model
    nlay = 12  # Number of layers
    nrow = 10  # Number of rows
    ncol = 51  # Number of columns
    ib = np.ones((nlay, nrow, ncol))
    delc = [4, 40, 24, 16, 12,
            10, 6, 4, 4, 4]  # Column spacing (meters)
    delr = [4, 16, 8, 4, 4, 4, 4, 3, 2, 2, 
            2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
            3, 4, 4, 4, 4, 4, 4, 12, 24, 40, 
            4]  # Column Spacing (meters)
    top = -273  # Top elevation (constant, meters above mean sea level)
    botm = np.zeros((nlay, nrow, ncol))  # Bottom elevations of each layer (meters)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        idomain=ib,
        botm=botm,
        xorigin=-148,
        yorigin=0
    )
    top_elev = reformat_arrays(os.path.join(dis_ws, 'tops', 'top_elev.txt'), 
                        nrow, 
                        ncol)
    
    layer_txt_files = [f"layer_{i}.txt" for i in range(1, 13)]

    botm96 = [
        reformat_arrays(os.path.join(dis_ws, 'bottoms', filename), nrow, ncol)
        for filename in layer_txt_files
    ]

    dis.top = top_elev # meters above mean sea level
    dis.botm = botm96 # meters above mean sea level
    dis.set_all_data_external()
    return dis

def initialize_chemistry(ws, nlay, nrow, ncol):
    # get chemistry

    solutionsdf = pd.read_csv(os.path.join(datadir,"ic_aq_chem.csv"), index_col = 0)

    ## let's process the injection chem
    injdf = pd.read_csv(os.path.join(datadir,"wellin.csv"), index_col = 0)
    injdf = injdf[['layer'] + solutionsdf.index.tolist()].copy()
    injdf
    layers_inj = list(injdf.layer.unique())

    frames = []                     

    for per in injdf.index.unique():
        df = (
            injdf.loc[per].reset_index()      
                .drop(columns='kper')        
                .groupby('layer').mean()    
                .T                          
        )
        # give columns names like  0_1, 0_2, …   (or f"{per}_{layer}")
        df.columns = [f"{per}_{layer}" for layer in df.columns]

        frames.append(df)            

    injdf = pd.concat(frames, axis=1) 
    solutionsdf = pd.concat([solutionsdf, injdf], axis=1)

    solutions = utils.solution_df_to_dict(solutionsdf)

    sol_ic = np.ones((nlay, nrow, ncol), dtype=float)

    solution = mup3d.Solutions(solutions)
    solution.set_ic(sol_ic)
    solution

    excdf = pd.read_csv(os.path.join(os.path.join(datadir,"ic_exchanger.csv")), comment = '#')
    ex_names = {
                "Ca_ex": "CaX2",
                "Fe_ex":"FeX2" ,
                "K_ex":"KX" ,
                "Mg_ex":"MgX2", 
                "Na_ex":"NaX"
                }
    # we need to rename to match the database
    excdf['name'] = excdf['var'].map(ex_names)
    excdf = excdf.pivot(index="name", columns="layer", values="value")

    exchangerdic = utils.solution_df_to_dict(excdf)
    exchanger = mup3d.ExchangePhases(exchangerdic)

    # exchanger_ic = np.ones((nlay, nrow, ncol), dtype=float)
    layer_vals = np.arange(1, nlay + 1, dtype=float)   # shape (nlay,)

    # Broadcast to full 3‑D grid
    exchanger_ic = layer_vals[:, None, None] * np.ones((1, nrow, ncol))
    exchanger.set_ic(exchanger_ic)

    # we need to equilibrate the exchangers with the background solution
    # init background solution is the same (number 1) for the whole domain
    # the next array has a length of nlay, one for each layer
    eq_solutions = [1] * nlay
    exchanger.set_equilibrate_solutions(eq_solutions)

    mindf = pd.read_csv(os.path.join(os.path.join(datadir,"ic_surfaces.csv")), comment = '#')
    mindf = mindf.pivot(index="var", columns="layer", values="value")
    
    # only ferrihydrite and orgmatter are in eq
    eq_m0 = utils.solution_df_to_dict(mindf.loc[['Ferrihydrite', "Orgmatter"],:])

    # we need the initial Sat indeces SI
    # following original model init SI is 0
    si = 0

    eq_dic = {}
    #lets add pyrite first
    for ly in range(nlay):
        # si followed by m0 (init moles)
        eq_dic[ly+1] = {key: [si, eq_m0[key][ly]] for key in eq_m0.keys()}
    eq_dic
    equilibriums = mup3d.EquilibriumPhases(eq_dic)
    equilibriums.set_ic(exchanger_ic)

    # pyrite is kinetic
    py_m0 = utils.solution_df_to_dict(mindf.loc[["Pyrite"],:])

    # we need the kinetic params for Py
    kin_py_params = [1.600000e+01, 
                     6.700000e-01, 
                     5.000000e-01, 
                     -1.100000e-01]
    kin_dic = {}

    # lets add kinetics for Organic carbon (Orgc)
    kin_orgc_params = [1.570000e-09, 1.670000e-11, 1.000000e-13]
    # orgc also have a custom formula
    orgc_form =  "Orgc -1.0 CH2O 1.0"

    #lets add pyrite first
    for ly in range(nlay):
        # print(ly+1)
        for key in py_m0.keys():
            kin_dic[ly+1] = {key: [py_m0[key][ly], kin_py_params]}
    kin_dic

    # lets add orgc now with a m0 of zero for all layers
    for key in kin_dic.keys():
        kin_dic[key]['Orgc'] = [0.0, kin_orgc_params, orgc_form]
    kin_dic
    kinetics = mup3d.KineticPhases(kin_dic)
    kinetics.set_ic(exchanger_ic)
    kinetics.data

    model = mup3d.Mup3d('dizon36',solution, nlay, nrow, ncol)


    # #set model workspace
    model.set_wd(ws)

    # set database
    database = os.path.join(datadir, f'datab.dat')
    model.set_database(database)

    postfix = os.path.join(datadir, f'postfix.phqr')
    model.set_postfix(postfix)
    model.set_exchange_phases(exchanger)
    model.set_phases(kinetics)
    model.set_phases(equilibriums)

    model.initialize()

    return model

def make_wel_in(gwf, one_compound = None, mup3d_m=None, nper=39):
    coords_in = [
        (1, 9, 38),
        (2, 9, 38),
        (3, 9, 38),
        (5, 9, 38),
        (7, 9, 38),
    ]
    layers_inj = [i[0] for i in coords_in]

    df_inj = pd.read_csv(os.path.join(datadir, "wellin.csv"))
    wellin_sp_data = defaultdict(list)

    if one_compound is not None:
        assert one_compound in df_inj.columns, print("compound not in wellin csv")
        for _, r in df_inj.iterrows():
            cell = (int(r["layer"]), int(r["row"]), int(r["column"]))  # zero‑indexed
            wellin_sp_data[int(r["kper"])].append([cell, r["rate"], r[f"{one_compound}"]])
        wel_in  = flopy.mf6.ModflowGwfwel(gwf, 
                                       stress_period_data=wellin_sp_data,
                                       auxiliary=one_compound,
                                       pname = 'welin',
                                       filename=f'{gwf.name}.welin')
        wel_in.set_all_data_external()

    else:
        wel_chem_dir = {}
        indices = [list(range(i, i + 5)) for i in range(2, 197, 5)]
        for per in range(nper):
            sol_spd = indices[per]
            wellchem = mup3d.ChemStress('per_'+str(per))
            wellchem.set_spd(sol_spd)
            mup3d_m.set_chem_stress(wellchem)
            wel_chem_dir[per] = wellchem.data

        for _, r in df_inj.iterrows():
            cell = (int(r["layer"]), int(r["row"]), int(r["column"]))  # zero‑indexed
            wellin_sp_data[int(r["kper"])].append([cell, r["rate"]])

        for per in range(nper):
            for e, layer in  enumerate(layers_inj):
                chem_arr = wel_chem_dir[per][e]
                wellin_sp_data[per][e].extend(chem_arr)
        wel_in  = flopy.mf6.ModflowGwfwel(gwf, 
                                        stress_period_data=wellin_sp_data,
                                        auxiliary=mup3d_m.components,
                                        pname = 'welin',
                                        filename=f'{gwf.name}.welin')
        wel_in.set_all_data_external()
        return wel_in

def make_wel_out(gwf, one_compound = None, mup3d_m=None, nper=39):

    coords_out = [ # (row, col, layer) 
        (1, 9,  9),
        (3, 9,  9),
        (5, 9,  9),
    ]
    init_rates_out  = [-300,  -30,  -30]                # 3 negatives
    fini_rates_out  = [-400,  -40,  -40]

    init_sp = range(0, 36)   # stress periods 0 – 35
    fini_sp = range(36, nper)  # stress periods 36 – 38
    all_sp  = (*init_sp, *fini_sp)


    def make_rows(coords, rates, add_conc=True):
        if len(coords) != len(rates):
            raise ValueError("Coordinate and rate lists must be the same length")
        return [
            ([cell, q] if add_conc else [cell, q])
            for cell, q in zip(coords, rates)
        ]

    # Time‑invariant blocks for each phase
    wellout_init  = make_rows(coords_out, init_rates_out)
    wellout_fini  = make_rows(coords_out, fini_rates_out, add_conc=True)
    wellout_sp_data = {sp: (wellout_init if sp in init_sp else wellout_fini)
                    for sp in all_sp}

    if one_compound is not None:
        wellout_sp_data = append_values_to_inner_lists(wellout_sp_data, 0.0)
        wel_out = flopy.mf6.ModflowGwfwel(gwf, 
                                            stress_period_data=wellout_sp_data, 
                                            auxiliary=one_compound,
                                            pname = 'welout' ,
                                            filename=f'{gwf.name}.welout')
        wel_out.set_all_data_external()
    else:
        wellout_sp_data = append_values_to_inner_lists(wellout_sp_data, [0.0]*len(mup3d_m.components))
        wel_out = flopy.mf6.ModflowGwfwel(gwf, 
                                            stress_period_data=wellout_sp_data, 
                                            auxiliary=mup3d_m.components,
                                            pname = 'welout' ,
                                            filename=f'{gwf.name}.welout')
        wel_out.set_all_data_external()
    return wel_out

def prep_model_dir(name="model"):
    model_ws = os.path.join(name)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)

    # Re‑create an empty folder
    os.makedirs(model_ws, exist_ok=True)
    utils.prep_bins(model_ws)
    return os.path.join(name)

def make_gwf(ws, model_name = "gwf", tracer = 'Cl', mup3d_m = None):

    nper = 39  # Number of stress periods

    perioddata= [(2, 2, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (7, 7, 1),
                (7, 7, 1), (7, 7, 1), (7, 7, 1), (14, 14, 1), (14, 14, 1), 
                (15, 15, 1), (13, 13, 1), (14, 14, 1), (14, 14, 1), (14, 14, 1), 
                (21, 21, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (35, 35, 1), (35, 35, 1),(28, 28, 1)]

    steady = [False] * nper  # All stress periods are transient

    # specify the mf6 gw object & add relevant components
    sim = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', sim_ws='.')
    sim.set_sim_path(ws)

    # specify tdis
    tdis = flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS", 
                                 nper=nper, perioddata=perioddata)

    ims = flopy.mf6.ModflowIms(sim, 
                            #    pname="ims", 
                            complexity="COMPLEX",
                            filename=f"{model_name}.ims")
    sim.register_ims_package(ims, 
                             [model_name])
    
    # start model build to refine 
    model_nam_file = "{}.nam".format(model_name)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, 
                               model_nam_file=model_nam_file, exe_name='mf6')

    # make dis
    dis = make_dis(gwf)

    nlay = dis.nlay.get_data()
    nrow = dis.nrow.get_data()
    ncol = dis.ncol.get_data()

    # get properties
    botm96 = np.zeros((nlay, nrow, ncol))
    
    thick96 = np.zeros((nlay, nrow, ncol))
    scoeff96 = np.zeros((nlay, nrow, ncol))
    trans96 = np.zeros((nlay, nrow, ncol))
    k = np.zeros((nlay, nrow, ncol))
    k33 = np.zeros((nlay, nrow, ncol))
    vcont96 = np.zeros((nlay, nrow, ncol))

    # now let's start filling data from the mf96 arrays to populate the mf6 model
    layer_txt_files = [f"layer_{i}.txt" for i in range(1, 13)]

    
    for i, layer_txt_files in enumerate(layer_txt_files):

        thick96[i] = reformat_arrays(os.path.join(dis_ws, 'thicknesses', layer_txt_files), 
                                    nrow, 
                                    ncol)
        scoeff96[i] = reformat_arrays(os.path.join(props_ws, 's', layer_txt_files), 
                                    nrow, 
                                    ncol)
        trans96[i] = reformat_arrays(os.path.join(props_ws, 'T', layer_txt_files), 
                                    nrow, 
                                    ncol)
        vcont96[i] = reformat_arrays(os.path.join(props_ws, 'vcont', layer_txt_files), 
                                    nrow, 
                                    ncol)

    # first load in mf96 top elev array reformt and assign to mf6 dis

    ihead = 0 #(meters)
    start = ihead * np.ones((nlay, nrow, ncol))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)
    ic.set_all_data_external()

    karr=trans96/thick96
    k33_array = calculate_vertical_conductivity(vcont96, thick96, nlay, nrow, ncol)

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=karr,
        k33=k33_array,
    )
    # npf.k = trans96/thick96
    npf.set_all_data_external()

    sto = flopy.mf6.ModflowGwfsto(gwf, ss=scoeff96/thick96, iconvert=0)
    sto.set_all_data_external()

    # CHD boundaries
    l_hd= 0
    chdspd = []

    for i in range(nlay):          # layers
        for j in range(nrow):      # rows
            chdspd.append([(i, j, 0), l_hd])           # left boundary
            chdspd.append([(i, j, ncol - 1), l_hd])    # right boundary

    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=True,
        pname="CHD",
        filename=f"{model_name}.chd",
    )
    chd.set_all_data_external()

    wel_out = make_wel_out(gwf, nper =39, one_compound=tracer, mup3d_m=mup3d_m)
    # make wel in
    wel_in = make_wel_in(gwf, nper =39, one_compound=tracer, mup3d_m=mup3d_m)

    # create the output control
    headfile = f"{model_name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{model_name}.cbb"
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    printrecord = [("HEAD", "LAST")]
    
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,)
    # sim.write_simulation()
    return sim


######## transport ###############

def make_gwt(sim, tracer = 'Cl', mup3d_m=None):

    gwf = sim.get_model("gwf")
    nlay = gwf.dis.nlay.get_data()

    ne = 0.35 # effective porosity (-) constant across all layers
    long_disp = 0.01 # Longitudinal dispersivity (m)constant across all layers
    disp_tr_vert = long_disp*0.01 # Transverse vertical dispersivity (m) constant across all layers
    disp_tr_hor = long_disp*0.1 # Transverse horizontal dispersivity (m) constant across all layers
    diffc = 0 # diffusion coefficient constant across all layers
    pbulk = 1850 # bulk density in constant across all layers (m/L^3) 


    ne = 0.35              # effective porosity (-)
    long_disp = 0.01       # longitudinal dispersivity (m)
    disp_tr_vert = long_disp * 0.01  # transverse vertical dispersivity (m)
    disp_tr_hor = long_disp * 0.1    # transverse horizontal dispersivity (m)
    diffc = 0              # diffusion coefficient
    pbulk = 1850           # bulk density (m/L^3)

    if mup3d_m is not None:
        components = mup3d_m.components
    else:
        components = [tracer]

    for comp in components:
        model_name = comp
        gwt = flopy.mf6.MFModel(
            sim,
            model_type="gwt6",
            modelname=model_name,
            model_nam_file=f"{model_name}.nam"
        )

        imsgwt = flopy.mf6.ModflowIms(sim, 
                                    #   pname="ims", 
                                complexity="COMPLEX",
                                filename=f"{model_name}.ims")
        sim.register_ims_package(imsgwt, 
                                [model_name])

        dis = gwf.dis

        dis = flopy.mf6.ModflowGwtdis(
                gwt,
                nlay=gwf.dis.nlay.get_data(),
                nrow=gwf.dis.nrow.get_data(),
                ncol=gwf.dis.ncol.get_data(),
                delr=gwf.dis.delr.get_data(),
                delc=gwf.dis.delc.get_data(),
                top=gwf.dis.top.get_data(),
                botm=gwf.dis.botm.get_data(),
                idomain=gwf.dis.idomain.get_data(),
                filename=f"{model_name}.dis",
                xorigin=gwf.dis.xorigin.get_data(),
                yorigin=gwf.dis.yorigin.get_data()
            )
        dis.set_all_data_external()

        nlay = dis.nlay.get_data()
        nrow = dis.nrow.get_data()
        ncol = dis.ncol.get_data()

        strt = mup3d_m.sconc[comp]
        ic = flopy.mf6.ModflowGwtic(gwt, strt=strt, 
                                    filename=f"{model_name}.ic")
        ic.set_all_data_external()

        adv = flopy.mf6.ModflowGwtadv(
            gwt,
            scheme="tvd",
        )
        adv.set_all_data_external()

        alpha_l = np.ones(shape=(nlay,ncol, nrow))*long_disp  # Longitudinal dispersivity ($m$)
        alpha_th = np.ones(shape=(nlay,ncol, nrow))*disp_tr_hor  # Transverse horizontal dispersivity ($m$)
        alpha_tv = np.ones(shape=(nlay,ncol, nrow))*disp_tr_vert  # Transverse vertical dispersivity ($m$)

        dsp = flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=alpha_l,
            ath1=alpha_th,
            atv = alpha_tv,
            diffc = diffc,
            filename=f"{model_name}.dsp",
        )
        dsp.set_all_data_external()

        sourcerecarray = [["welin", "aux", model_name],
                        ["welout", "aux", model_name]]

        ssm = flopy.mf6.ModflowGwtssm(
                gwt,
                sources=sourcerecarray,
                save_flows=True,
                print_flows=True,
                filename=f"{model_name}.ssm",
            )
        ssm.set_all_data_external()

        mst = flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=ne,
            first_order_decay=None,
            decay = None,
            decay_sorbed=None,
            sorption= None,
            bulk_density=None, 
            distcoef=None, #Kd m3/mg
            sp2 = None,
            filename=f"{model_name}.mst",
        )
        mst.set_all_data_external()
        oc = flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{model_name}.cbb",
            concentration_filerecord=f"{model_name}.ucn",
            concentrationprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 10, "GENERAL")
                                        ],
            saverecord=[("CONCENTRATION", "ALL"), 
                        ],
            printrecord=[("CONCENTRATION", "LAST"), 
                            ],
        )
        flopy.mf6.ModflowGwfgwt(
            sim,
            exgtype="GWF6-GWT6",
            exgmnamea='gwf',
            exgmnameb=f'{model_name}',
            filename=f"{model_name}.gwfgwt",
        )
        make_obs_pack(gwt)
    
    
    sim.write_simulation() 
    return sim
# lin_sorp_distr_coeff = {

#     'Orgc':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'O(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'C(4)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'C(-4)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Ca':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Cl':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
#     'Fe(2)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Fe(3)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'K':            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Mg':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(3)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(5)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Na':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'S(-2)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'S(6)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Si':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Amm':          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Tmp':          [2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04,
#                      2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04, 2.1141E-04],
#     'pH':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'pe':           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Ferrihydrite': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Orgmatter':    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Ca_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Fe_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'K_ex':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Mg_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Na_ex':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Pyrite':       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

# }

# # define initial concentration conditions in all 12 layers
# initial_concentrations = {

#     'Orgc':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'O(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'C(4)':         [0.008446, 0.008446, 0.008446, 0.008446, 0.008446, 0.008446, 
#                      0.008446, 0.008446, 0.008446, 0.008446, 0.008446, 0.008446],
#     'C(-4)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Ca':           [0.002057, 0.002057, 0.002057, 0.002057, 0.002057, 0.002057, 
#                      0.002057, 0.002057, 0.002057, 0.002057, 0.002057, 0.002057],
#     'Cl':           [0.000254, 0.000254, 0.000254, 0.000254, 0.000254, 0.000254, 
#                      0.000254, 0.000254, 0.000254, 0.000254, 0.000254, 0.000254], 
#     'Fe(2)':        [0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041, 
#                      0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041, 0.0001041],
#     'Fe(3)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'K':            [0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336,
#                      0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336, 0.0001336],
#     'Mg':           [0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875, 
#                      0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875, 0.0005875],
#     'N(3)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(5)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Na':           [0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748, 
#                      0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748, 0.0006748],
#     'S(-2)':        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'S(6)':         [5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05,
#                      5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05, 5.516E-05],
#     'Si':           [0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497, 
#                      0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497, 0.0003497],
#     'Amm':          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'N(0)':         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Tmp':          [0.017, 0.017, 0.017, 0.017, 0.017, 0.017, 
#                      0.017, 0.017, 0.017, 0.017, 0.017, 0.017],
#     'pH':           [6.601, 6.601, 6.601, 6.601, 6.601, 6.601, 
#                      6.601, 6.601, 6.601, 6.601, 6.601, 6.681],
#     'pe':           [-2.449, -2.449, -2.449, -2.449, -2.449, -2.449, 
#                      -2.449, -2.449, -2.449, -2.449, -2.449, -2.449],
#     'Ferrihydrite': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     'Orgmatter':    [2.410781, 0.3873438, 1.445313, 0.1907813, 0.1907813, 0.1907813,
#                      1.925156, 1.925156, 1.925156, 1.925156, 1.925156, 1.925156],
#     'Ca_ex':        [0.1188, 0.02138, 0.07358, 0.01891, 0.01891, 0.01891,
#                      0.05919, 0.05919, 0.05919, 0.05919, 0.05919, 0.05919],
#     'Fe_ex':        [0.002451, 0.0004411, 0.001518, 0.0003902, 0.0003902, 0.0003902,
#                      0.001221, 0.001221, 0.001221, 0.001221, 0.001221, 0.001221],
#     'K_ex':         [0.00157, 0.0002825, 0.0009723, 0.0002499, 0.0002499, 0.0002499, 
#                      0.0007822, 0.0007822, 0.0007822, 0.0007822, 0.0007822, 0.0007822],
#     'Mg_ex':        [0.02131, 0.003835, 0.0132, 0.003393, 0.003393, 0.003393, 
#                      0.01062, 0.01062, 0.01062, 0.01062, 0.01062, 0.01062],
#     'Na_ex':        [0.001596, 0.0002872, 0.0009888, 0.0002541, 0.0002541, 0.0002541,
#                      0.0007954, 0.0007954, 0.0007954, 0.0007954, 0.0007954, 0.0007954],
#     'Pyrite':       [0.0629, 0.0222, 0.0999, 0.01295, 0.01295, 0.01295, 
#                      0.13135, 0.13135, 0.13135, 0.13135, 0.13135, 0.13135]
    
# }

# # define icbund arrays for each layer for the transport model from the previous pht3d model
# layer_txt_files = ['layer_1.txt', 'layer_2.txt', 'layer_3.txt', 'layer_4.txt', 
#                    'layer_5.txt', 'layer_6.txt', 'layer_7.txt', 'layer_8.txt', 
#                    'layer_9.txt', 'layer_10.txt', 'layer_11.txt', 'layer_12.txt']

# transport_ws = os.path.join(os.getcwd(), 'transport')

# pht3d_icbund = np.zeros((nlay, nrow, ncol))
# for i, layer_txt_files in enumerate(layer_txt_files):
#     pht3d_icbund[i] = reformat_arrays(os.path.join(transport_ws, 'props', 'icbund', layer_txt_files), 
#                                 os.path.join(transport_ws, 'props', 'icbund', 'format_'+str(layer_txt_files)), 
#                                 nrow, 
#                                 ncol)
    
# # define parameters for source-sink-mixing (SSM) package defined from SSM package file
# # specify six booleans in line 1 for types
# FWEL = 'T' # fwel flag is true
# FDRN = 'F' # fdrn flag is false
# FRCH = 'F' # frch flag is false
# FEVT = 'F' # fevt flag is false
# FRIV = 'F' # friv flag is false
# FGHB = 'F' # fghb flag is false

# ssm_line_1 = pd.DataFrame([[str(FWEL), str(FDRN), str(FRCH), str(FEVT), str(FRIV), str(FGHB)]])
# ssm_line_1.columns = [0, 1, 2, 3, 4, 5]  # Assuming the first column is the index column

# # define ssm line 2
# MXSS = 2245 # maximum number of source-sinks defined from SSM package file
# ssm_line_2 = pd.DataFrame([[MXSS]], columns=[0])

# # define all of the source-sink terms for the well boundary conditions with chemistry for all 29 constituents
# # lay, row, col, CSS (0=dummy value per manual), type (2=well), 
# # Orgc, O(0), C(4), C(-4), Ca, 
# # Cl, Fe(2), Fe(3), K, Mg, 
# # N(3), N(5), Na, S(-2), S(6), 
# # Si, Amm, N(0), Tmp, pH, 
# # pe, Ferrihydrite, Orgmatter, Ca_ex, Fe_ex, 
# # K_ex, Mg_ex, Na_ex, Pyrite

# # load in csv of ssm entries
# ssm_entry_rows = pd.read_csv(os.path.join(os.getcwd(), 'transport', 'ssm', 'ssm_entries.csv'), index_col=False)
# # Remove column headers from the CSV data
# ssm_entry_rows.columns = range(ssm_entry_rows.shape[1])  # Reset column names to default integer-based names

# # combine all entries to make ssm package dataframe

# ssm_combined = pd.concat([ssm_line_1, ssm_line_2, ssm_entry_rows], ignore_index=True, axis=0)

# # Check the first 5 columns for numeric values and round them to nearest integer for lay row col css type
# # We only modify the first 5 columns (0, 1, 2, 3, 4)
# for col in range(5):  # Loop through the first 5 columns
#     # Convert values to numeric, keeping 'T' and 'F' booleans intact
#     ssm_combined.iloc[1:, col] = pd.to_numeric(ssm_combined.iloc[1:, col], errors='coerce')

#     # Round numeric values to nearest whole number (integer), but leave 'T' and 'F' unaffected
#     ssm_combined.iloc[1:, col] = ssm_combined.iloc[1:, col].round()

# # print combined ssm 
# print(ssm_combined)
# # Now let's write the DataFrame to a text file while skipping NaN values
# file_path = os.path.join(os.getcwd(), 'transport', 'ssm', 'ssm_pkg_test.txt')  # Specify your file path
# # Convert DataFrame to string (without NaN values)
# ssm_combined_cleaned = ssm_combined.fillna('')  # Replace NaN with empty string if needed
# output_str = ssm_combined_cleaned.to_string(index=False, header=False)

# # Write the string to a text file
# with open(file_path, 'w') as file:
#     file.write(output_str)

# # import phinp.dat from pht3d model using phreeqc rm
# nxyz = nlay * ncol * nrow
# nthreads = 3

# phreeqc_rm = phreeqcrm.PhreeqcRM(nxyz, nthreads)

# database_fpth = os.path.join(os.getcwd(), 'transport', 'database', 'dizon.pht3d_database')

# # Load the database
# phreeqc_rm.LoadDatabase(database_fpth)

# pqi_fpth = os.path.join(os.getcwd(), 'transport', 'pqi', 'phinp.dat')

# # Run the input file
# phreeqc_rm.RunFile(True, True, True, pqi_fpth)

def main():
    ws = prep_model_dir()
    nlay = 12
    nrow = 10
    ncol = 51
    mup3d_m=initialize_chemistry(ws, nlay, nrow, ncol)
    sim = make_gwf(ws, tracer=None,mup3d_m=mup3d_m)
    sim = make_gwt(sim, tracer='Cl', mup3d_m=mup3d_m)
    run_model(sim)
if __name__ == "__main__":
    main()