import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import pyemu
import geopandas as gpd
import platform
import shapefile as sf
from mf6rtm import utils, mup3d
from collections import defaultdict
from flopy.utils.gridintersect import GridIntersect
from collections.abc import Iterable
from pypestutils.pestutilslib import PestUtilsLib
lib = PestUtilsLib()

datadir = os.path.join("data")
dis_ws = os.path.join(datadir, 'dis')
props_ws = os.path.join(datadir, 'props')

nper = 39  # Number of stress periods

perioddata= [(2, 2, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (7, 7, 1),
            (7, 7, 1), (7, 7, 1), (7, 7, 1), (14, 14, 1), (14, 14, 1), 
            (15, 15, 1), (13, 13, 1), (14, 14, 1), (14, 14, 1), (14, 14, 1), 
            (21, 21, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
            (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
            (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
            (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
            (28, 28, 1), (35, 35, 1), (35, 35, 1),(28, 28, 1)]

def get_avg_distance(points, npoints=10):
    """
    Calculate the average distance to the nearest n points for each point in a set of points.

    Parameters
    ----------
    points : numpy array
        Array of points.
    npoints : int
        Number of nearest points to calculate the average distance to.
    Returns
    -------
    average_distances : numpy array
        Array of average distances to the nearest n points for each point in the input array.
    """

    from scipy.spatial import distance
    distances = distance.cdist(points, points, 'euclidean')
    np.fill_diagonal(distances, np.inf)
    nearest_n = np.partition(distances, npoints, axis=1)[:, :npoints]
    average_distances = np.mean(nearest_n, axis=1)
    return average_distances

def interpolate_property_to_grid(gwf, ws, property_array, property_name="k"):
    """
    Interpolate a 3D property array (shape: nlay, nrow, ncol) onto the grid using kriging.
    """
    bps = gpd.read_file(os.path.join(datadir, 'botm.shp'))
    bps['x'] = bps.geometry.centroid.x
    bps['y'] = bps.geometry.centroid.y

    ppeasting = bps.x.values
    ppnorthing = bps.y.values
    anis = 1
    bearing = 0.0
    aa = 1.5 * get_avg_distance(bps[['x','y']].values, 10).max()

    easting = gwf.modelgrid.xcellcenters.flatten()
    northing = gwf.modelgrid.ycellcenters.flatten()
    max_pts = 50
    min_pts = 1
    search_dist = 1.e+10
    aa_pp = aa
    zone_pp = np.ones_like(ppeasting, dtype=int)
    fac_file = os.path.join(ws, f"factors.bin")

    ib = np.ones_like(easting, dtype=int)
    ipts = lib.calc_kriging_factors_2d(
        ppeasting, ppnorthing, zone_pp,
        easting, northing, ib.flatten(),
        "exp", "ordinary",
        aa_pp, anis, bearing, search_dist, max_pts, min_pts, fac_file, "binary"
    )

    nlay = gwf.dis.nlay.get_data()
    nrow = gwf.dis.nrow.get_data()
    ncol = gwf.dis.ncol.get_data()
    icpls = nrow * ncol

    interpolated = []
    for layer in range(nlay):
        # Flatten the property array for this layer
        ppval = property_array[layer].flatten()
        result = lib.krige_using_file(
            fac_file, "binary", icpls, "ordinary", "log",
            np.array(ppval), np.zeros_like(icpls), 0
        )
        # check if result['targval'] has negative values
        if np.any(result['targval'] < 0):
            print(f"Negative values found in result['targval'] for prop {property_name} layer {layer+1}")
        interpolated.append(np.round(result['targval'], 6))
    # check if inteprolated has negative values
    interpolated = np.array(interpolated).reshape((nlay, nrow, ncol))
    if np.any(interpolated < 0):
        print(f"Negative values found in interpolated array for prop {property_name}")
        raise ValueError("Interpolated values contain negative values.")
    return np.array(interpolated)


def get_properties_from_og(ws, gwf, prop = ['k', 'ss', 'sy']):
    sim_og = flopy.mf6.MFSimulation.load(sim_ws = ws,
                                  sim_name = 'gwf', 
                                  version='mf6',
                                    exe_name='mf6',
                                    verbosity_level=0)
    gwf_og = sim_og.get_model("gwf")
    dict_prop = {}
    for p in prop:
        # print(p)
        if p in ['k', 'k33']:
            # print(f"property: {p} min: {karr.min()} max: {karr.max()}")
            og_arr = getattr(gwf_og.npf, p).get_data()
        if p in ['ss', 'sy']:
            og_arr = getattr(gwf_og.sto, p).get_data()
            # print(f"property: {p} min: {sarr.min()} max: {sarr.max()}")
        prop_arr = interpolate_property_to_grid(gwf, ws, og_arr, p)
        dict_prop[p] = prop_arr

    return dict_prop

def get_botms(gwf, ws):
    bps = gpd.read_file(os.path.join(datadir, 'botm.shp'))
    bps['x'] = bps.geometry.centroid.x
    bps['y'] = bps.geometry.centroid.y
    bps

    ppeasting = bps.x.values
    ppnorthing = bps.y.values
    anis = 1
    bearing= 0.0
    aa = 1.5 * get_avg_distance(bps[['x','y']].values, 2).max()

    ib = gwf.dis.idomain.get_data()
    # cellids = df.loc[df.layer==layer+1].icpl.values - 1 # zero-based
    easting = gwf.modelgrid.xcellcenters.flatten()
    northing = gwf.modelgrid.ycellcenters.flatten()

    max_pts = 50 # pp are same as cell centers, so kind of irrelevant
    min_pts = 1
    search_dist = 1.e+10
    aa_pp = aa #?
    zone_pp = np.ones_like(ppeasting,dtype=int)
    fac_file = os.path.join(ws,f"factors.bin")

    ib = np.ones_like(easting,dtype=int)
    ipts = lib.calc_kriging_factors_2d(ppeasting,
                                    ppnorthing,
                                    zone_pp,
                                    easting,
                                    northing,
                                    ib.flatten(),
                                    "exp","ordinary",
                                    aa_pp,anis,bearing,search_dist,max_pts,min_pts,fac_file,"binary")

    botms = []
    icpls = gwf.dis.nrow.get_data() * gwf.dis.ncol.get_data()
    for layer in range(1, 13):
        # get COND multiplier
        ppval = bps[f"botm_{layer}"].values
        result = lib.krige_using_file(os.path.join(ws,f"factors.bin"),
                                        "binary",
                                        icpls,
                                        "ordinary",
                                        "none",
                                        np.array(ppval),
                                        np.zeros_like(icpls),
                                        0)
        botms.append(np.round(result['targval'], 1))
    return botms

def make_gwf_structured(ws, model_name = "gwf", tracer = 'Cl', mup3d_m = None):

    nper = 39  # Number of stress periods

    perioddata= [(2, 2, 1), (4, 4, 1), (4, 4, 1), (4, 4, 1), (7, 7, 1),
                (7, 7, 1), (7, 7, 1), (7, 7, 1), (14, 14, 1), (14, 14, 1), 
                (15, 15, 1), (13, 13, 1), (14, 14, 1), (14, 14, 1), (14, 14, 1), 
                (21, 21, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (35, 35, 1), (35, 35, 1), (28, 28, 1), (28, 28, 1), (28, 28, 1), 
                (28, 28, 1), (35, 35, 1), (35, 35, 1),(28, 28, 1)]

    # specify the mf6 gw object & add relevant components
    sim = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', sim_ws='.')
    sim.set_sim_path(ws)

    # specify tdis
    tdis = flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS", 
                                 nper=nper, perioddata=perioddata)
    outer_dvclose = 1e-7
    inner_dvclose = 1e-7
    ims = flopy.mf6.ModflowIms(sim, 
                            #    pname="ims", 
                            complexity="complex",
                            outer_dvclose=outer_dvclose,
                            inner_dvclose=inner_dvclose,
                            filename=f"{model_name}.ims")
    sim.register_ims_package(ims, 
                             [model_name])
    
    # start model build to refine 
    model_nam_file = "{}.nam".format(model_name)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, 
                               model_nam_file=model_nam_file, exe_name='mf6')

    length_units = "METERS"

    domain = gpd.read_file(os.path.join(datadir, 'modeldis.shp'))
    Ly = domain.geometry.total_bounds[3] - domain.geometry.total_bounds[1]
    Lx = domain.geometry.total_bounds[2] - domain.geometry.total_bounds[0]

    xul = domain.geometry.total_bounds[0]
    yul = domain.geometry.total_bounds[1]

    dis = make_dis_structured(gwf, Ly=Lx, Lx=Ly, xul=xul, yul=yul)
    botms = get_botms(gwf, ws)
    dis.botm.set_data(botms)
    dis.set_all_data_external()

    nlay = dis.nlay.get_data()
    nrow = dis.nrow.get_data()
    ncol = dis.ncol.get_data()

    # dis.idomain.export(os.path.join("output", "dis_botm.vtk"), fmt="vtk")
    dict_prop = get_properties_from_og(ws=os.path.join("model", "reactive_demo"),
                            gwf=gwf, prop = ['k', 'k33', 'ss', 'sy'])

    ihead = 0 #(meters)
    strt = ihead * np.ones((nlay, nrow, ncol))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)
    ic.set_all_data_external()

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=dict_prop['k'],
        k33=dict_prop['k33'],
        # k33overk = True
    )
    npf.set_all_data_external()
    ss = np.ones((nlay, nrow, ncol)) * 1.e-4
    sto = flopy.mf6.ModflowGwfsto(gwf, 
                                  ss=dict_prop['ss'], 
                                  iconvert=0,
                                    # steady_state={0: False},
                                    transient={0: True})
    sto.set_all_data_external()

    chd = make_chd(gwf, one_compound=tracer, mup3d_m=mup3d_m)

    wel_out = make_wel_out(gwf, nper =39, one_compound=tracer, mup3d_m=mup3d_m, from_shp=True)
    # # make wel in
    wel_in = make_wel_in(gwf, nper =39, 
                        one_compound=tracer, mup3d_m=mup3d_m, from_shp=True)

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
    sim.write_simulation()
    return sim


def make_dis_structured(gwf, Ly=10, Lx=10, xul=0, yul=0):
    # Define model parameters for the dummy model
    del_ = 8  # Grid spacing
    nrow = int(Lx / del_)+1
    ncol = int(Ly / del_)+1
    nlay = 12  # Number of layers
    ib = np.ones((nlay, nrow, ncol))

    top = -273  # Top elevation (constant, meters above mean sea level)
    botm = np.zeros((nlay, nrow, ncol))  # Bottom elevations of each layer (meters)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=del_,
        delc=del_,
        top=top,
        idomain=ib,
        botm=botm,
        xorigin=xul,
        yorigin=yul
    )
    return dis
# def copy_gwt_model_files_from_parent(ws=".", parent_model_dir = 'gwtbenzene',
#                                      dsp_par =  ['alh', 'ath1', 'atv'], 
#                                  ist_par = ['porosity', 'zetaim', 'volfrac', 'bulk_density'],
#                                  mst_par = ['buk_density']
#                                  ):
#     import shutil
#     def get_transport_input_filenames(tag, template_ws=os.path.join('pest','pst_template'), 
#                                     gwtdir = 'gwtbenzene'):
#         template_ws = os.path.join(template_ws, gwtdir)
#         files = [os.path.join(gwtdir,f) for f in os.listdir(template_ws) if tag in f.lower() and f.endswith(".txt")] 
#         return files 
#     def flatten(xss):
#         return [x for xs in xss for x in xs]
#     tag = []
#     modeldirs = [name for name in os.listdir(ws) if (os.path.isdir(os.path.join(ws, name)) ) and (name.startswith('gwt'))]

#     assert parent_model_dir in modeldirs, f'{parent_model_dir} does not exist in dir of models'
    
#     modeldirs.remove(parent_model_dir)

#     for e, par in enumerate(dsp_par):
#         tag.append(f"dsp_{par}_")

#     for  e, par in enumerate(ist_par):
#          tag.append(f"ist_{par}_")

#     for  e, par in enumerate(mst_par):
#          tag.append(f"mst_{par}_")
         
#     fnames_to_copy = sorted([get_transport_input_filenames(t, template_ws=ws, gwtdir = parent_model_dir) for t in tag])
#     fnames_to_copy = flatten(fnames_to_copy)
    
#     for dir in modeldirs:
#         fnames_to_replace = sorted([get_transport_input_filenames(t, template_ws=ws, gwtdir = dir) for t in tag])
#         fnames_to_replace = flatten(fnames_to_replace)
#         assert sorted([x.split('.')[1] for x in fnames_to_copy]) == sorted([x.split('.')[1] for x in fnames_to_replace]), f'list of files to replace and to copy does not contain the same files names '
#         # sort fnames_to_copy and fnames_to_replace according to the assert above
#         fnames_to_copy = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_copy], fnames_to_copy))]
#         fnames_to_replace = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_replace], fnames_to_replace))]
#         fileszipped = list(zip(fnames_to_copy, fnames_to_replace))
#         [shutil.copyfile(os.path.join(ws, f[0]), os.path.join(ws, f[1])) for f in fileszipped]
#     return fileszipped

def clean_obs_chem(datadir = "data",
                    input_path="obs_chem_raw_0.csv", 
                   output_path="obs_chem_cleaned.csv"):
    """Load, clean, and save obs_chem data by converting units of specific variables."""
    input_path = os.path.join(datadir, input_path)
    output_path = os.path.join(datadir, output_path)
    obsdata = pd.read_csv(input_path)
    obsdata.loc[:, 'obsid'] = obsdata['obsid'].str.lower()

    data_source2 = pd.read_csv(os.path.join("data", "obs_chem_raw_1.csv"))
    to_pull = ['Fe2', 'Fe3', 'tic']
    data_source2=data_source2[data_source2["var"].isin(to_pull)].copy()
    data_source2['units'] = 'mol_l'
    data_source2=data_source2[obsdata.columns]

    obsdata = pd.concat([obsdata, data_source2])

    # Standardize column name
    obsdata.rename(columns={'var': 'variable'}, inplace=True)
    
    # Unit conversions
    unit_conversions = {
        # 'Cl': 1/(58.44*1e3),   # mmol/L → mol/L
        'Tmp': 1e-3,  # apply scaling to be consistent with outputs
    }

    for var, factor in unit_conversions.items():
        obsdata.loc[obsdata['variable'] == var, 'value'] *= factor

    # Save cleaned file
    obsdata.to_csv(output_path, index=False, float_format="%.5e")

    return obsdata

def time_interpolate(sim_times, sim_vals, obs_times):
    import numpy as np
    from scipy import interpolate
    t0 = min(sim_times)
    sim_times = [(i-t0).astype(float) for i in sim_times]
    obs_times = [(i-t0).astype(float) for i in obs_times]

    # Create interpolation function
    f = interpolate.interp1d(sim_times, sim_vals, fill_value='extrapolate')
    # Interpolate at new times
    new_values = f(obs_times)
    return new_values

def create_output_pairs(perioddata, output_interval=5):
    pairs = []
    cumulative_day = 0
    next_output_day = 0
    
    for kper, (perlen, nstp, tsmult) in enumerate(perioddata):
        period_days = int(perlen)
        
        # Check each day in this stress period
        for day_in_period in range(period_days):
            if cumulative_day == next_output_day:
                pairs.append((kper+1, day_in_period+1))
                next_output_day += output_interval
            
            cumulative_day += 1
    
    return pairs

def append_values_to_inner_lists(d, values, *, in_place=False):
    """
    Append a single value or all values from an iterable to every inner list
    inside a {key: list[list]} dictionary.

    Parameters
    ----------
    d : dict
        Your nested list dictionary.
    values : any or Iterable
        * If `values` is not an Iterable (or is str/bytes), its treated as a
          single item and appended once.
        * If `values` is an Iterable (list/tuple/set/range), each element is
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
    # print(obs_list)
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
    excdf['layer'] -= 1  # convert to zero‑indexed
    excdf = excdf.pivot(index="name", columns="layer", values="value")
    exchanger_dict = excdf.to_dict()
    for k, subdict in exchanger_dict.items():
        for key in subdict:
            subdict[key] = {'m0': subdict[key]}

    exchanger = mup3d.ExchangePhases(exchanger_dict)

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
    mindf['value'] = [utils.concentration_volbulk_to_volwater(i,0.35) for i in mindf['value'].values]
    mindf = mindf.pivot(index="var", columns="layer", values="value")
    
    # only ferrihydrite and orgmatter are in eq
    eq_m0 = utils.solution_df_to_dict(mindf.loc[['Ferrihydrite', "Orgmatter"],:])

    # we need the initial Sat indeces SI
    # following original model init SI is 0
    si = 0

    eq_dic = {}
    #lets add pyrite first
    for ly in range(nlay):
        for key in eq_m0.keys():
            # create a dictionary for each layer with key as the mineral name
            # and values as a dictionary with si and m0
        # si followed by m0 (init moles)
            eq_dic[ly] = {key: {}}
            eq_dic[ly][key]['si'] = si
            eq_dic[ly][key]['m0'] = eq_m0[key][ly]
            # eq_dic[ly+1] = {key: [si, eq_m0[key][ly]] for key in eq_m0.keys()}
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
    orgc_steps = "8.640000e+04 in 1 steps"

    #lets add pyrite first
    for ly in range(nlay):
        # print(ly+1)
        for key in py_m0.keys():
            kin_dic[ly] = {key: {}}
            kin_dic[ly][key]['m0'] = py_m0[key][ly]
            kin_dic[ly][key]['parms'] = kin_py_params
            # kin_dic[ly+1] = {key: [py_m0[key][ly], kin_py_params]}

    # lets add orgc now with a m0 of zero for all layers
    for key in kin_dic.keys():
        kin_dic[key]['Orgc'] = {}
        kin_dic[key]['Orgc']['m0'] = 1.0
        kin_dic[key]['Orgc']['parms'] = kin_orgc_params
        kin_dic[key]['Orgc']['formula'] = orgc_form
        kin_dic[key]['Orgc']['steps'] = orgc_steps
        # [1.0, kin_orgc_params, orgc_form, orgc_steps]
    kinetics = mup3d.KineticPhases(kin_dic)
    kinetics.set_ic(exchanger_ic)
    model = mup3d.Mup3d('dizon36',solution, nlay, nrow, ncol)

    # #set model workspace
    model.set_wd(ws)

    # set database
    # shutil copy datab
    shutil.copy(os.path.join(datadir, f'datab.dat'), os.path.join(model.wd, f'datab.dat'))
    database = os.path.join(f'datab.dat')
    model.set_database(database)

    postfix = os.path.join(datadir, f'postfix.phqr')
    model.set_postfix(postfix)
    model.set_exchange_phases(exchanger)
    model.set_phases(kinetics)
    model.set_phases(equilibriums)
    # model.set_charge_offset(1e-3)
    tsteps = create_output_pairs(perioddata, output_interval=2)

    # machine learning variables
    targetvars = [
    'Orgc','O0','tic','C_4',
    'Fe2','Fe3','N3','NO3',
    'S_2','SO4','Amm',
    'N0','pH',
    'pe','EQUI_Ferrihydrite',
    'EQUI_Orgmatter',
    'MOL_CaX2','MOL_FeX2',
    'MOL_KX','MOL_MgX2',
    'MOL_NaX','KIN_Pyrite'
    ]
    featvars = [
    'Orgc',
    # 'O0',
    # 'tic',
    # 'C_4',
    # 'Fe2',
    # 'Fe3',
    # 'N3',
    # 'NO3',
    # 'S_2',
    # 'SO4',
    # 'Amm',
    # 'N0',
    # 'pH',
    # 'pe',
    'EQUI_Ferrihydrite',
    'EQUI_Orgmatter',
    'MOL_CaX2','MOL_FeX2',
    'MOL_KX','MOL_MgX2',
    'MOL_NaX','KIN_Pyrite'
    ]

    model.set_config(
                    reactive_timing='all',
                    reactive_externalio=True,
                    emulator_training_data=True,
                    emulator_target_variables=targetvars,
                    emulator_feature_variables=featvars,
                    # tsteps=tsteps
                    )
    model.set_componenth2o(True)
    model.initialize(add_charge_flag=True)
    return model

def get_wel_coords(gwf, name  = "wellin"):
    mg = gwf.modelgrid
    ix = GridIntersect(mg)
    wells = pd.read_csv(os.path.join(datadir, "wells.csv"))
    wells = gpd.GeoDataFrame(wells, geometry=gpd.points_from_xy(wells.x, wells.y))

    assert name in wells.name.values, f"{name} not in well"
    wells = wells[wells.name == name]
    geom = wells.geometry[wells.name==name].values

    assert len(geom)==1, f"more than one well with name {name} in wells.csv"
    cellid = ix.intersect(geom[0], 'point').cellids
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    mg.plot(ax=ax)
    ix.plot_point(ix.intersect(geom[0], 'point'), ax=ax)
    print(cellid[0])
    return cellid[0]

def make_wel_in(gwf, one_compound = None,
                mup3d_m=None, nper=39, from_shp=False):
    layers = [1,2,3,5,7]
    if from_shp:
        # coords_in = {}
        cellid = get_wel_coords(gwf, name  = "wellin")
        coords_in = {lay: (lay, cellid[0], cellid[1]) for lay in layers}

        print(cellid)
    else:
        coords_in = [
            (1, 9, 38),
            (2, 9, 38),
            (3, 9, 38),
            (5, 9, 38),
            (7, 9, 38),
        ]
    df_inj = pd.read_csv(os.path.join(datadir, "wellin.csv"))
    wellin_sp_data = defaultdict(list)

    if one_compound is not None:
        assert one_compound in df_inj.columns, print("compound not in wellin csv")
        #get all unique cells
        for _, r in df_inj.iterrows():
            if from_shp:
                layer = int(r["layer"])
                cell = coords_in[layer]  # zero‑indexed
            else:
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
            if from_shp:
                layer = int(r["layer"])
                cell = coords_in[layer]  # zero‑indexed
            else:
                cell = (int(r["layer"]), int(r["row"]), int(r["column"]))  # zero‑indexed
            wellin_sp_data[int(r["kper"])].append([cell, r["rate"]])

        for per in range(nper):
            for e, layer in  enumerate(layers):
                chem_arr = wel_chem_dir[per][e]
                wellin_sp_data[per][e].extend(chem_arr)
        wel_in  = flopy.mf6.ModflowGwfwel(gwf, 
                                        stress_period_data=wellin_sp_data,
                                        auxiliary=mup3d_m.components,
                                        pname = 'welin',
                                        filename=f'{gwf.name}.welin')
        wel_in.set_all_data_external()
        return wel_in


def make_wel_out(gwf, one_compound = None, mup3d_m=None, nper=39, from_shp=False):

    if from_shp:
        layers = [1,3,5]
        cellid = get_wel_coords(gwf, name  = "wellout")
        coords_out = [(lay, cellid[0], cellid[1]) for lay in layers]
        print(coords_out)
    else:
        coords_out = [ # (row, col, layer) 
            (1, 9,  9),
            (3, 9,  9),
            (5, 9,  9),
        ]
    init_rates_out  = [-300,  -30,  -30]                # 3 negatives
    fini_rates_out  = [-400,  -40,  -40]

    init_sp = range(0, 35)   # stress periods 0 – 35
    fini_sp = range(35, nper)  # stress periods 36 – 38
    all_sp  = (*init_sp, *fini_sp)


    def make_rows(coords, rates, add_conc=True):
        if len(coords) != len(rates):
            raise ValueError("Coordinate and rate lists must be the same length")
        return [
            ([cell, q] if add_conc else [cell, q])
            for cell, q in zip(coords, rates)
        ]

    # Time‑invariant blocks for each phase
    if from_shp:
        wellout_init = {sp: make_rows(coords_out, init_rates_out) for sp in all_sp}
        wellout_fini = {sp: make_rows(coords_out, fini_rates_out, add_conc=True) 
                        for sp in all_sp}
        wellout_sp_data = {sp: (wellout_init[sp] if sp in init_sp else wellout_fini[sp])
                    for sp in all_sp}
    else:
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
                                            pname = 'welout',
                                            filename=f'{gwf.name}.welout')
        
        wel_out.set_all_data_external()
    return wel_out

def make_chd(gwf, one_compound = None, mup3d_m=None):
    
    l_hd= 0
    nlay = gwf.dis.nlay.get_data()
    nrow = gwf.dis.nrow.get_data()
    ncol = gwf.dis.ncol.get_data()

    if one_compound is not None:
        df_inj = pd.read_csv(os.path.join(datadir, "ic_aq_chem.csv"), index_col=0)
        assert one_compound in df_inj.index, f"compound {one_compound} not in ic_aq_chem csv"
        c_list = [df_inj.loc[one_compound, 'value']]
        aux = one_compound
        # print(c_list)
    else:
        chdchem = mup3d.ChemStress('chdchem')
        sol_spd = [1]
        chdchem.set_spd(sol_spd)
        mup3d_m.set_chem_stress(chdchem)
        c_list = mup3d_m.chdchem.data[0]
        aux=mup3d_m.components

    chdspd = []
    for i in range(nlay):          # layers
        for j in range(nrow):      # rows
            chdspd.append([(i, j, 0), l_hd])           # left boundary
            chdspd.append([(i, j, ncol - 1), l_hd])    # right boundary
    # print(chdspd)
    for i in range(len(chdspd)):
        chdspd[i].extend(c_list)

    # print(chdspd)
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chdspd),
        stress_period_data=chdspd,
        save_flows=True,
        auxiliary=aux,
        pname="CHD",
        filename=f"{gwf.name}.chd",
    )
    chd.set_all_data_external()
    return chdspd

def prep_model_dir(name="model"):
    model_ws = os.path.join("model", name)
    if os.path.exists(model_ws):
        shutil.rmtree(model_ws)

    # Re‑create an empty folder
    os.makedirs(model_ws, exist_ok=True)
    utils.prep_bins(model_ws)
    return os.path.join(model_ws)

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

    # specify the mf6 gw object & add relevant components
    sim = flopy.mf6.MFSimulation(sim_name=model_name, version='mf6', sim_ws='.')
    sim.set_sim_path(ws)

    # specify tdis
    tdis = flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS", 
                                 nper=nper, perioddata=perioddata)
    outer_dvclose = 1e-7
    inner_dvclose = 1e-7
    ims = flopy.mf6.ModflowIms(sim, 
                            #    pname="ims", 
                            complexity="complex",
                            outer_dvclose=outer_dvclose,
                            inner_dvclose=inner_dvclose,
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
    # k = np.zeros((nlay, nrow, ncol))
    # k33 = np.zeros((nlay, nrow, ncol))
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
        # k33overk = True
    )
    npf.set_all_data_external()

    sto = flopy.mf6.ModflowGwfsto(gwf, 
                                  ss=scoeff96/thick96, 
                                  iconvert=0,
                                    # steady_state={0: False},
                                    transient={0: True})
    sto.set_all_data_external()

    chd = make_chd(gwf, one_compound=tracer, mup3d_m=mup3d_m)

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

    return sim

def make_gwt(sim, tracer = 'Cl', mup3d_m=None):

    gwf = sim.get_model("gwf")
    nlay = gwf.dis.nlay.get_data()

    ne = 0.35 # effective porosity (-) constant across all layers
    long_disp = 0.1 # Longitudinal dispersivity (m)constant across all layers
    disp_tr_vert = long_disp*0.01 # Transverse vertical dispersivity (m) constant across all layers
    disp_tr_hor = long_disp*0.1 # Transverse horizontal dispersivity (m) constant across all layers
    diffc = 0 # diffusion coefficient constant across all layers
    pbulk = 1850 # bulk density in constant across all layers (m/L^3) 

    if mup3d_m is not None and tracer is None:
        components = mup3d_m.components
    else:
        components = [tracer]

    for comp in components:
        print(f"Setting transport for {comp}")
        model_name = comp
        gwt = flopy.mf6.MFModel(
            sim,
            model_type="gwt6",
            modelname=model_name,
            model_nam_file=f"{model_name}.nam"
        )
        outer_dvclose = 1e-7
        inner_dvclose = 1e-7
        ims = flopy.mf6.ModflowIms(sim, 
                                #    pname="ims", 
                                complexity="complex",
                                outer_dvclose=outer_dvclose,
                                inner_dvclose=inner_dvclose,
                                filename=f"{model_name}.ims")
        sim.register_ims_package(ims, 
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
        if tracer is not None:
            strt = mup3d_m.sconc[comp]/1000 # to mmol
        else:
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

        sourcerecarray = [
                        ["welin", "aux", model_name],
                        ["welout", "aux", model_name],
                        ["chd", "aux", model_name]
                        ]

        ssm = flopy.mf6.ModflowGwtssm(
                gwt,
                sources=sourcerecarray,
                save_flows=True,
                print_flows=True,
                filename=f"{model_name}.ssm",
            )
        ssm.set_all_data_external()

        if comp=='Tmp':
            # kd = 2*ne/1850
            distcoef = np.ones(shape=(nlay,ncol, nrow))*2.1141E-04
            sorption = "Linear"
            pbulk = 1850
            bulk_density = np.ones(shape=(nlay,ncol, nrow))*pbulk
            ne=0.35
        else:
            distcoef = None
            sorption = None
            bulk_density = None
            ne=0.35
        porosity  = np.ones(shape=(nlay,ncol, nrow))*ne
        
        mst = flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=porosity,
            first_order_decay=None,
            decay = None,
            decay_sorbed=None,
            sorption= sorption,
            bulk_density=bulk_density, 
            distcoef=distcoef, #Kd m3/mg
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

def clean_array_files(ws, files, gwf):
    nrows, ncols = gwf.dis.nrow.get_data(), gwf.dis.ncol.get_data()
    for file in files:
        with open(os.path.join(ws,file),'r') as f:
            lines = []
            for l in f.readlines():
                lines.extend([float(i) for i in l.split()])
        array = np.array(lines).reshape(nrows,ncols)
        np.savetxt(os.path.join(ws,file),array,fmt='%.6e')
    return

# def get_lst_budget(ws='.',start_datetime=None):
#     import flopy
#     lst = flopy.utils.Mf6ListBudget(os.path.join(ws,"pl253.lst"))
#     inc,cum = lst.get_dataframes(diff=True,start_datetime=start_datetime)
#     inc.columns = inc.columns.map(lambda x: x.lower().replace("_","-"))
#     cum.columns = cum.columns.map(lambda x: x.lower().replace("_", "-"))
#     inc.index.name = "time"
#     cum.index.name = "time"
#     inc.to_csv(os.path.join(ws,"inc.csv"))
#     cum.to_csv(os.path.join(ws,"cum.csv"))
#     return inc, cum

def tidy_array_input_files(f, template_ws=os.path.join('pest','pst_template')):
    """Re-writes array input files as a single column."""
    filename=os.path.join(template_ws, f)
    with open(filename, 'r') as ff:
        a = ff.read()
    a = [float(i) for i in a.split()]
    # make a single column
    a = np.array(a).reshape(-1,1)
    # record array to txt file in single column
    np.savetxt(os.path.join(template_ws, f), a, fmt='%.8e')
    return

def get_input_filenames(tag, template_ws=os.path.join('pest','pst_template')):
    files = [f for f in os.listdir(template_ws) if tag in f and f.endswith(".txt")] 
    return files 

def copy_obs_data_files_to_wd(datadir,wd,files=[]):
    for f in files:
        assert os.path.isfile(os.path.join(datadir,f)),f"file {f} not found"
        print(f"Getting {f} from {datadir} to {wd}")
        shutil.copy(os.path.join(datadir,f), os.path.join(wd, f))

def draw_prior_pe(num_reals, pf, template_ws):
    sigma_range=4.0
    if pf.pst.npar < 36000:  #if you have more than about 35K pars, the cov matrix becomes hard to handle
        prior_cov = pf.build_prior(fmt='coo',
                                   filename=os.path.join(template_ws,"prior_cov.jcb"),
                                   sigma_range=sigma_range)
        pf.pst.pestpp_options["parcov"] = "prior_cov.jcb"
        # draw ensemble
        pe = pyemu.ParameterEnsemble.from_gaussian_draw(pf.pst, cov=prior_cov, num_reals=num_reals)
    else:
        # draw parameters from the prior distribution
        pe = pf.draw(num_reals=num_reals, use_specsim=False, sigma_range=sigma_range) 

    ### Fix params in boundary TVMs
    params = list(pe.columns)
    to_fix = [i for i in params if ('boundary' in i) & ('tvm' in i)]
    pe.loc[:, to_fix] = 1
    # enforces parameter bounds
    pe.enforce() 
    # save external file
    pe.to_binary(os.path.join(template_ws,"prior_pe.jcb")) 
    pf.pst.pestpp_options["ies_parameter_ensemble"] = "prior_pe.jcb"
    return pe

def process_sim_conc(wd='.'):
    from flopy.utils.gridintersect import GridIntersect

    sim = flopy.mf6.MFSimulation.load(sim_ws = wd,
                                    sim_name = 'gwf', 
                                    version='mf6',
                                        exe_name='mf6',
                                        verbosity_level=0)
    gwf = sim.get_model("gwf")
    sout = pd.read_csv(os.path.join(wd, "sout.csv"))

    # nlay = gwf.dis.nlay.get_data()
    ncol = gwf.dis.ncol.get_data()
    nrow = gwf.dis.nrow.get_data()

    ix = GridIntersect(gwf.modelgrid)
    obsloc = pd.read_csv(os.path.join(wd, "obs_loc.csv"))

    obs_list=[]

    for obsid in obsloc.obsid.unique():
        x,y = obsloc.loc[obsloc.obsid==obsid,['x','y']].values[0]
        cellid = ix.intersect([(x,y)],shapetype="point").cellids
        if len(cellid)==0:
            continue
        else:
            cellid = cellid[0]
            obs_layer = int(obsloc.loc[obsloc.obsid==obsid,'layer'].values[0])
            obs_list.append((obsid, 'concentration', (obs_layer, cellid[0], cellid[1])))

    wells = pd.DataFrame(obs_list)
    wells.columns = ['obsid', 'n', 'cell']
    wells['flat_index'] = [cell[0] * (nrow * ncol) + cell[1] * ncol + cell[2] for cell in wells.cell]

    obs_to_index = dict(zip(wells['flat_index'], wells['obsid'].str.lower()))

    sout['cell'] = sout['cell'].astype(int)
    sout['obsid'] = sout['cell'].map(obs_to_index)

    obsdata = pd.read_csv(os.path.join(wd, "obs_chem_cleaned.csv"))
    obsdata.rename(columns={'var': 'variable'},  inplace=True)

    missvar = set(obsdata['variable'].unique()) - set(sout.columns)
    obs_ = sout[['time', 'cell']+list(set(obsdata['variable'].unique())  - missvar)].copy()
    obs_ = obs_.melt(id_vars = ['time', 'cell'])
    obs_['obsid'] = obs_['cell'].map(obs_to_index)
    obs_['obsid'].unique()

    obs_in_ = obs_[~obs_['obsid'].isna()].copy()
    variables = obs_in_.variable.unique()
    obsids = obs_in_.obsid.unique()

    obsdata = obsdata[obsdata['obsid'].isin(obsids)].copy()
    obsdata = obsdata[obsdata['variable'].isin(variables)].copy()

    dfmerged = pd.merge(obs_in_[['time','obsid', 'variable', 'value']], obsdata,
                        on=['time','obsid', 'variable'], how='outer')
    dfmerged.rename(columns={'value_x':'sim',
                            'value_y': 'meas'}, inplace=True)

    dfmerged.sort_values(['obsid','time'], inplace=True)
    dfmerged.set_index('time', inplace=True)

    for oid in obsids:
        print(f"Processing obs for: {oid:>5}")
        for var in variables:
            mask=(dfmerged.obsid==oid)&(dfmerged.variable==var)

            tmp = dfmerged.loc[mask].copy()
            tmp.dropna(subset=['sim'], inplace=True)
            if tmp.shape[0]==0:
                continue
            obs_times = dfmerged.loc[mask].index.values
            dfmerged.loc[mask,'sim'] = time_interpolate(tmp.index.values,
                                                        tmp.sim.values,
                                                        obs_times)

    fname = '_obs.conc.simvsmeas.csv'
    dfmerged = dfmerged.reset_index()
    dfmerged.drop_duplicates(subset=['time', 'obsid', 'variable'], inplace=True)
    dfmerged = dfmerged.set_index('time')
    dfmerged.replace(np.nan,1e30).to_csv(os.path.join(wd, fname))
    print(f"Processed conc saved in {wd}/{fname}")

    return fname

def add_std_to_pst(casename="dizon36",
                    template_ws=os.path.join("pest","pst_template"),
                    fraction=1):
    pst = pyemu.Pst(os.path.join(template_ws,f"{casename}.pst"))
    obs = pst.observation_data
    obs['standard_deviation'] = obs['obsval']*fraction
    pst.write(os.path.join(template_ws, f"{casename}.pst"), version=2)

def build_noise_ensemble(casename="dizon36",
                         template_ws=os.path.join("pest","pst_template")):

    pst = pyemu.Pst(os.path.join(template_ws,f"{casename}.pst"))
    obs = pst.observation_data
    cov = pyemu.Cov.from_observation_data(pst)
    # get number of reals 
    pefname = os.path.join(template_ws, pst.pestpp_options["ies_parameter_ensemble"])
    pe = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=pefname)
    num_reals = pe.shape[0]

    # gerenate noise ensemble
    oe = pyemu.ObservationEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals)

    assert oe.columns.isin(obs.obsnme).all()

    #---save ensemble---#
    fname = os.path.join(template_ws, 'noise.jcb')
    oe.to_binary(fname)
    print(f'saving noise ensemble to: {fname}')

    pst.pestpp_options["ies_observation_ensemble"] = 'noise.jcb'
    pst.write(os.path.join(template_ws, f"{casename}.pst"), version=2)
    return oe

def setup_pest(org_d, num_reals=50, 
               template_ws=os.path.join('pest','pst_template'),
               casename = 'dizon36'):
    tmp_d=os.path.join("tmp_d")
    if os.path.exists(tmp_d):
        shutil.rmtree(tmp_d)
    shutil.copytree(org_d,tmp_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_d, verbosity_level=0)
    # load flow model
    gwf = sim.get_model()
    # get spatial reference
    modelgrid = gwf.modelgrid
    # get zone array
    ib = gwf.dis.idomain.get_data()
    ib[ib<1] = 0

    pf = pyemu.utils.PstFrom(original_d=tmp_d, 
                                new_d=template_ws,
                                remove_existing=True, 
                                longnames=True, 
                                spatial_reference=modelgrid, 
                                zero_based=False, 
                                # start_datetime=start_datetime, 
                                echo=False)

    pf.mod_sys_cmds.append("mf6rtm")
    pf.extra_py_imports.append("flopy")
    pf.add_py_function("workflow.py","process_sim_conc()",is_pre_cmd=False)
    pf.add_py_function("workflow.py","time_interpolate()",is_pre_cmd=None)

    copy_obs_data_files_to_wd(datadir,template_ws,files=['obs_chem_cleaned.csv', 'obs_loc.csv'])
    f = process_sim_conc(wd=template_ws)
    obs_df = pf.add_observations(f, 
                            insfile=f+".ins", 
                            index_cols=['time','obsid','variable'], 
                            use_cols=['sim'], 
                            prefix=f"hm") 
    pp_v = pyemu.geostats.ExpVario(contribution=1,
                                   a=75,
                                   anisotropy=4,
                                   bearing=90.0)
    pp_gs = pyemu.geostats.GeoStruct(variograms=pp_v, transform='log')

    tag_list = ["npf_k_", 
                # "npf_k33_",
                "sto_ss_",
                "kinetic_phases.Pyrite.m0.",
                # "equilibrium_phases.Orgmatter.m0.",
                # "exchange_phases.CaX2.m0",
                # "exchange_phases.FeX2.m0",
                # "exchange_phases.KX.m0",
                # "exchange_phases.MgX2.m0",
                # "exchange_phases.NaX.m0",
                ]
    lb=0.1
    ub=10.0
    for tag in tag_list:
    # uub=100.0
    # ulb=0.01
        files = get_input_filenames(tag, template_ws=template_ws)
        clean_array_files(template_ws, files, gwf)
        for f in files:
            try:
                # layer = int(f.split(".")[1].split("_layer")[-1]) -1
                layer = int(f.split(tag)[1].split('.txt')[0].split("layer")[-1])
            except:
                layer=0
            # print(layer)
            base = tag.replace("_",".")+'layer'+str(layer)
            print(base)
            pf.add_parameters(filenames=f,
                                par_type="pilotpoints",
                                par_name_base='pp.'+base,
                                pargp='pp.'+base,
                                # zone_array=ib[layer],
                                # use_pp_zones=True,
                                upper_bound=ub,
                                lower_bound=lb,
                                # ult_ubound=uub,
                                # ult_lbound=ulb,
                                pp_options={"pp_space":2,
                                            "prep_hyperpars":False},
                                geostruct=pp_gs,
                                # apply_order=2
                                )
            pf.add_parameters(f, 
                                # zone_array=ib[layer],
                                par_type="constant",
                                # geostruct=pp_gs,
                                par_name_base="cn."+base,
                                par_style='m',
                                pargp="cn."+base,
                                lower_bound=lb,
                                upper_bound=ub,
                                # ult_ubound=uub,
                                # ult_lbound=ulb
                                )

            pf.add_observations(f,
                                prefix=base,
                                obsgp=base,
                                # zone_array=ib[layer]
                                )

    pst = pf.build_pst()
    pe = draw_prior_pe(num_reals, pf, template_ws)
    if pf.pst.npar < 35000:
        pst.pestpp_options["parcov"] = "prior_cov.jcb"
    pst.pestpp_options["ies_parameter_ensemble"] = "prior_pe.jcb"
    pst.write(os.path.join(template_ws, f'{casename}.pst'), version=2)


def run_pestpp(md=f"master", td="pst_template", casename="isr", 
               noptmax=-1,freeze=False,
               num_reals=100,
               num_workers=10, worker_root=".", 
               pestpp_version="ies",restart=False,
               reuse_master=False, cleanup=True):

    pst = pyemu.Pst(os.path.join(td, f"{casename}.pst"))
    pst.control_data.noptmax = noptmax

    # and a options to reduce the number of lost reals and improve conditioning
    pst.pestpp_options["overdue_giveup_fac"] = 1e30
    pst.pestpp_options["overdue_giveup_minutes"] = 1e30
    pst.pestpp_options["ies_no_noise"] = False 
    pst.pestpp_options["ies_subset_size"] = -10 # the more the merrier
    pst.pestpp_options["ies_bad_phi_sigma"] = 2.0
    pst.pestpp_options["panther_agent_freeze_on_fail"] = freeze
    pst.pestpp_options["ies_num_reals"] = num_reals

    pst.write(os.path.join(td, f"{casename}.pst"), version=2)
    # run
    pyemu.os_utils.start_workers(td,
                                 f'pestpp-{pestpp_version}',
                                 f'{casename}.pst',
                                 num_workers=num_workers,
                                 worker_root=worker_root,
                                 master_dir=md,
                                 reuse_master=reuse_master,
                                 )
    return

def set_obsval_and_weights(casename="dizon36",
                           template_ws=os.path.join("pest","pst_template")):
    
    '''Set observation parval1 and weights for history matching in pst
    '''
    pst = pyemu.Pst(os.path.join(template_ws,f"{casename}.pst"))

    obs = pst.observation_data
    obs.weight = 0.0

    hm_obs = obs.oname == 'hm'
    zero_weight_obs = obs.loc[obs.obsval>=1e30].obsnme
    zero_weight_obs

    obs_hm = obs.loc[hm_obs].copy()
    obs_hm.sort_values(['obsid','time'], inplace=True)
    obs_hm['time'] = obs_hm['time'].astype(float)

    obs_chem = pd.read_csv(os.path.join(template_ws, "_obs.conc.simvsmeas.csv"))
    obs_chem.sort_values(['obsid','time'], inplace=True)
    obs_chem['variable'] = obs_chem['variable'].str.lower()

    assert obs_hm.shape[0] == obs_chem.shape[0]

    obs_chem = pd.merge(obs_hm, obs_chem, on=['time','obsid', 'variable'])
    obs_chem.loc[obs_chem.meas<1e30, 'weight'] = 1.0
    obs_chem['obgnme'] = obs_chem['variable']
    obs.loc[obs_chem.obsnme, 'obsval'] = obs_chem.meas.values
    obs.loc[obs_chem.obsnme, 'weight'] = obs_chem.weight.values
    assert obs.loc[(obs.oname=='hm') & (obs.weight>0)].shape[0] == obs_chem.loc[obs_chem.meas<1e30].shape[0]
    assert obs.loc[(obs.oname=='hm') & (obs.weight>0)].weight.sum() == obs_chem.loc[obs_chem.meas<1e30].shape[0]
    obs.loc[zero_weight_obs, 'weight'] = 0.0
    obs.loc[obs_chem.obsnme, 'obgnme'] = obs_chem.obgnme.values #oname per var

    pst.write(os.path.join(template_ws, f"{casename}.pst"), version=2)
    return pst

def main(prep_obs = True, run_base = True, run_base_struct = False,
         prep_pest = False, run_pest = False):

    if prep_obs:
        clean_obs_chem(datadir = "data",
                        input_path="obs_chem_raw_0.csv", 
                    output_path="obs_chem_cleaned.csv")
    
    if run_base:
        ws = prep_model_dir(name='reactive_demo')
        nlay = 12
        nrow = 10
        ncol = 51
        mup3d_m=initialize_chemistry(ws, nlay, nrow, ncol)
        tracer = None

        sim = make_gwf(ws, tracer=tracer,mup3d_m=mup3d_m)
        sim = make_gwt(sim, tracer=tracer, mup3d_m=mup3d_m)

        pyemu.os_utils.run('mf6rtm', cwd=sim.sim_path)
    if run_base_struct:
        ws = prep_model_dir(name='test')
        sim = make_gwf_structured(ws, tracer='Cl',mup3d_m=None)
        # pyemu.os_utils.run('mf6', cwd=sim.sim_path)
        gwf=sim.get_model("gwf")
        nlay = gwf.dis.nlay.get_data()
        nrow = gwf.dis.nrow.get_data()
        ncol = gwf.dis.ncol.get_data()
        mup3d_m=initialize_chemistry(ws, nlay, nrow, ncol)
        tracer = None
        sim = make_gwf_structured(ws, tracer=tracer,mup3d_m=mup3d_m)
        sim = make_gwt(sim, tracer=tracer, mup3d_m=mup3d_m)
        pyemu.os_utils.run('mf6rtm', cwd=sim.sim_path)
    if prep_pest:
        template_ws=os.path.join('pest','pst_template')
        org_d = os.path.join('model','test')
        setup_pest(org_d, num_reals=100)
        set_obsval_and_weights()
        add_std_to_pst(fraction=0.05)
        build_noise_ensemble()
    if run_pest:
        md=os.path.join('pest','master1')
        run_pestpp(md=md, td=template_ws, casename="dizon36", 
                noptmax=1,freeze=True,
                num_reals=50,
                num_workers=15, worker_root=".", 
                pestpp_version="ies",restart=False,
                reuse_master=False, cleanup=True)

if __name__ == "__main__":
    main(
        prep_obs = False,
        run_base = False,
        run_base_struct = False,
        prep_pest = True,
        run_pest = True
    )