#!/usr/bin/python3
import sys
import ctypes as ct
import numpy as np
import datetime
from bitsea.commons import genUserDateList as DL
from bitsea.commons.Timelist import TimeList
from bitsea.commons.utils import Time_Interpolation
from bitsea.commons.mask import Mask
from bitsea.commons.time_interval import TimeInterval
import pandas as pd
import netCDF4
import argparse
import os
import time

if 'DIIM_PATH' in os.environ:
    HOME_PATH = MODEL_HOME = os.environ["DIIM_PATH"]
else:
    
    print("Missing local variable DIIM_PATH. \nPlease add it with '$:export DIIM_PATH=path/to/diim'.")
    sys.exit()


class oasim_lib:
    def __init__(self, lib_path, config_path, lat, lon):
        self._lib = ct.cdll.LoadLibrary(lib_path)
        self._init_lib = self._lib.py_init_lib
        self._init_lib.restype = ct.c_void_p
        self._init_lib.argtypes = [ct.c_int,
                                   ct.c_int,
                                   ct.c_char_p,
                                   np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                   np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS")]

        self._finalize_lib = self._lib.py_finalize_lib
        self._finalize_lib.restype = None
        self._finalize_lib.argtypes = [ct.c_void_p]

        error = ct.c_bool()
        if lat.shape != lon.shape or len(lat.shape) != 1:
            raise ValueError("The lat and lon mesh are incompatible")

        self._ptr = self._init_lib(len(config_path),
                                   lat.shape[0],
                                   config_path.encode("utf-8"),
                                   lat,
                                   lon)

        if not self._ptr:
            raise FileNotFoundError("Corrupted file: {}".format(config_path))
    

    def __del__(self):
        if self._ptr:
            self._finalize_lib(self._ptr)
        


class calc_unit:
    def __init__(self, p_size, lib):
        self._lib = lib
        self._init_calc = self._lib._lib.py_init_calc
        self._init_calc.restype = ct.c_void_p
        self._init_calc.argtypes = [ct.c_int,
                                    ct.c_void_p]

        self._monrad = self._lib._lib.py_monrad
        self._monrad.restype = ct.c_bool
        self._monrad.argtypes = [ct.c_void_p,
                                 ct.c_int,
                                 ct.c_int,
                                 np.ctypeslib.ndpointer(ct.c_int, flags="F_CONTIGUOUS"),
                                 ct.c_int,
                                 ct.c_int,
                                 ct.c_double,
                                 ct.c_double,
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ct.c_double, flags="F_CONTIGUOUS")]
                                 

        self._finalize_calc = self._lib._lib.py_finalize_calc
        self._finalize_calc.restype = None
        self._finalize_calc.argtypes = [ct.c_void_p]
        
        self._ptr = self._init_calc(p_size, self._lib._ptr)


    def monrad(self, points, iyr, iday, sec_b, sec_e,
               sp, msl, ws10, tco3, t2m, d2m, tcc, tclw, cdrem,
               taua, asymp, ssalb):
        p_size, rows = taua.shape
        edout = np.zeros((p_size, rows), dtype=np.float64, order="F")
        esout = np.zeros((p_size, rows), dtype=np.float64, order="F")

        error = self._monrad(self._ptr, p_size, rows, 
                             np.asfortranarray(points, dtype=np.int32),
                             iyr, iday, sec_b, sec_e,
                             np.asfortranarray(sp, dtype=np.float64), 
                             np.asfortranarray(msl, dtype=np.float64), 
                             np.asfortranarray(ws10, dtype=np.float64), 
                             np.asfortranarray(tco3, dtype=np.float64), 
                             np.asfortranarray(t2m, dtype=np.float64), 
                             np.asfortranarray(d2m, dtype=np.float64), 
                             np.asfortranarray(tcc, dtype=np.float64),
                             np.asfortranarray(tclw, dtype=np.float64), 
                             np.asfortranarray(cdrem, dtype=np.float64),
                             np.asfortranarray(taua, dtype=np.float64), 
                             np.asfortranarray(asymp, dtype=np.float64), 
                             np.asfortranarray(ssalb, dtype=np.float64), 
                             edout, esout)
        
        if error:
            raise ValueError("Error in monrad computation.")

        return edout, esout


    def __del__(self):
        if self._ptr:
            self._finalize_calc(self._ptr)

def Load_Data(DATADIR, TimeList,before,after,prefix,dateformat='%Y%m%d-%H:%M:%S'):
    Before_date17 = TimeList[before].strftime(dateformat)
    After__date17 = TimeList[after ].strftime(dateformat)
    Before_File = prefix + Before_date17 + ".nc"
    After__File = prefix + After__date17 + ".nc"
    ncB = netCDF4.Dataset(DATADIR + '/' + Before_File,'r')
    ncA = netCDF4.Dataset(DATADIR + '/' + After__File,'r')
    return ncA,ncB


def arguments():
    parser = argparse.ArgumentParser(description = '''
    Script to create a timeseries of a map, with the outputs of the OASIM model.
    ''',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('--conf_file', action='store_true', help="Read the configuration values from file. ")
    parser.add_argument('--conf_file_name','-iconf',
                        type = str,
                        required = False,
                        help = '''Name of the conf_file, used to read the configurations values if --conf_file is used.''',
                        default = 'conf_file.csv'
                        )
    
    parser.add_argument('--dateformat', '-df',
                        type = str,
                        required = False,
                        help = '''Date format for reading and writing file names. e.g. prefix.%%Y%%m%%d-%%H:%%M:%%S.nc''',
                        default = '%Y%m%d-%H:%M:%S'
                        )
    parser.add_argument('--atmosphere_datadir', '-iatm',
                        type = str,
                        required = False,
                        help = '''Directory where to read the atmospheric data. Netcdfs with the variables 'sp' (surface pressure [Pa]),'msl'  (air pressure at mean sea level [Pa]), 'u10'  (wind velocity, horizontal component [m/s]), 'v10' (wind velocity, vertical component [m/s]), 'tco3' (total column ozone [kg m^-2]) , 't2m'  (2 meter temperature [K]), 'd2m' (2 meter dewpoint temperature [K]) and 'tclw' (total cloud cover [0-100]).  All with shape (time,lat,lon). The script reads all the netcdfs in the specified path with dates one day before and one day after the time range specified, and interpolates the outputs to have one hour resolution.   (default: ./) ''',
                        default = './'
                        )
    parser.add_argument('--aerosol_datadir', '-iaer',
                        type = str,
                        required = False,
                        help = '''Directory where to read the aerosol data. Netcdfs with the variables 'taua' (aerosol optical thickness), 'asymp' (aerosol asymmetry parameter) and 'ssalb' (aerosol single scattering albedo). All with shapes (time,wavelenghts,lat,lon). The script reads all the netcdfs in the specified path with dates thirty days before and thirty days after the time range specified, and interpolates the outputs to have one hour resolution.   (default: ./)''',
                        default = './'
                        )
    parser.add_argument('--cloud_datadir', '-icld',
                        type = str,
                        required = False,
                        help = '''Directory where to read the cloud data. Netcdfs with the variable 'cdrem' (cloud droplet effective radius [um]).  Shape (time,lat,lon). The script reads all the netcdfs in the specified path with dates thirty days before and thirty days after the time range specified, and interpolates the outputs to have one hour resolution.   (default: ./)''',
                        default = './'
                        )
    parser.add_argument('--atmosphere_prefix', '-patm',
                        type = str,
                        required = False,
                        help = '''Prefix in the name of the atmospheric input data. e.g. prefix.%%Y%%m%%d-%%H:%%M:%%S.nc ''',
                        default = 'atm'
                        )
    parser.add_argument('--aerosol_prefix', '-paer',
                        type = str,
                        required = False,
                        help = '''Prefix in the name of the aerosol input data. e.g. prefix.%%Y%%m%%d-%%H:%%M:%%S.nc ''',
                        default = 'aero'
                        )
    parser.add_argument('--cloud_prefix', '-pcld',
                        type = str,
                        required = False,
                        help = '''Prefix in the name of the cloud input data. e.g. prefix.%%Y%%m%%d-%%H:%%M:%%S.nc ''',
                        default = 'climatm'
                        )
    parser.add_argument('--start_date', '-ds',
                        type = str,
                        required = False,
                        help = '''Date to start the OASIM computation, in the same format as --dateformat.''',
                        default = '20200102-07:00:00'
                        )
    parser.add_argument('--end_date', '-de',
                        type = str,
                        required = False,
                        help = '''Date to end the OASIM computation, in the same format as --dateformat.''',
                        default = '20200102-16:00:00'
                        )
    parser.add_argument('--wavelengths_file', '-iw',
                        type = str,
                        required = False,
                        help = '''File with the wavelenghts to be interpolated.''',
                        default = HOME_PATH + '/extern/OASIM_ATM/test/data/bin.txt'
                        )
    parser.add_argument('--mask_file', '-im',
                        type = str,
                        required = False,
                        help = '''Path of the meshmask corresponding to input data''',
                        default = '/g100_work/OGS_devC/V9C/RUNS_SETUP/PREPROC/MASK/meshmask.nc'
                        )
    parser.add_argument('--location', '-l',
                        type = str,
                        required = False,
                        help = '''map to create work with all the Mediterranean Sea, file to add a sequence of coordenates using the argument --location_file, use format "lat,lon", e.g. "43.367,7.9" for Boussole and "35.49,12.47" for Lampedusa.''',
                        default = 'map'
                        )
    parser.add_argument('--location_file', '-iloc',
                        type = str,
                        required = False,
                        help = '''file with lats and lons. The file has to be composed of two columns, separated with a space, first column with lats, and the second with lons.''',
                        default = HOME_PATH + '/map/location_file.txt'
                        )

    parser.add_argument('--oasimlib_file', '-ilib',
                        type = str,
                        required = False,
                        help = '''file with liboasim-py.so''',
                        default = HOME_PATH + '/map/liboasim-py.so'
                        )
    parser.add_argument('--oasim_config_file', '-iyaml',
                        type = str,
                        required = False,
                        help = '''file with the oasim config.yaml''',
                        default = HOME_PATH + '/map/oasim_config.yaml'
                        )    
    parser.add_argument('--output_prefix', '-pout',
                        type = str,
                        required = False,
                        help = '''prefix to write the output name.''',
                        default = 'oasim_med'
                        )
    parser.add_argument('--output_datadir', '-o',
                        type = str,
                        required = False,
                        help = '''Output file where to store the output.''',
                        default = './'
                        )

                               
    return parser  
    
def read_arguments():
    """
    read the arguments passed with argparser. If a configuration file is passed (use --conf_file), reads the inputs from a configuration file with name --conf_file_name. Parameters parsed have higher priority than conf_file parameters. 
    """
    parser = arguments()
    args = parser.parse_args()
    default_values = {action.dest: action.default for action in parser._actions if action.default is not argparse.SUPPRESS}
    if args.conf_file:
        conf_ = pd.read_csv(args.conf_file_name,sep = ' ',index_col = 0,comment = '#').T.reset_index().rename(columns={'index':'dateformat'})
        conf_ = {name_:conf_[name_][0] for name_ in conf_.keys()}

    else:
        conf_ = default_values

    for action in parser._actions:
        if any([(option in sys.argv) for option in action.option_strings]):
            conf_[action.dest] = vars(args)[action.dest]
    return conf_


def getting_oasim_output(dateformat='%Y%m%d-%H:%M:%S',\
                         atmosphere_datadir='./',\
                         aerosol_datadir='./',\
                         cloud_datadir='./',\
                         atmosphere_prefix='atm',\
                         aerosol_prefix='aero',\
                         cloud_prefix='climatm',\
                         start_date='20200102-07:00:00',\
                         end_date='20200102-16:00:00',\
                         wavelengths_file=HOME_PATH + '/extern/OASIM_ATM/test/data/bin.txt',\
                         mask_file='/g100_work/OGS_devC/V9C/RUNS_SETUP/PREPROC/MASK/meshmask.nc',\
                         location='map',\
                         oasimlib_file=HOME_PATH + '/map/liboasim-py.so',\
                         oasim_config_file=HOME_PATH + '/map/oasim_config.yaml',\
                         lats = [],\
                         lons = [],\
                         TheMask=None
                         ):
    """
    Script to create a timeseries of a map, with the outputs of the OASIM model.
    
    Parameters:

       dateformat:
            Date format for reading and writing file names. e.g. prefix.%Y%m%d-%H:%M:%S.nc (default:
            %Y%m%d-%H:%M:%S)

       atmosphere_datadir:
            Directory where to read the atmospheric data. Netcdfs with the variables 'sp' (surface pressure [Pa]),'msl'  (air pressure at mean sea level [Pa]), 'u10'  (wind velocity, horizontal component [m/s]), 'v10' (wind velocity, vertical component [m/s]), 'tco3' (total column ozone [kg m^-2]) , 't2m'  (2 meter temperature [K]), 'd2m' (2 meter dewpoint temperature [K]) and 'tclw' (total cloud cover [0-100]).  All with shape (time,lat,lon). The script reads all the netcdfs in the specified path with dates one day before and one day after the time range specified, and interpolates the outputs to have one hour resolution.   (default: ./) 

       aerosol_datadir
            Directory where to read the aerosol data. Netcdfs with the variables 'taua' (aerosol optical thickness), 'asymp' (aerosol asymmetry parameter) and 'ssalb' (aerosol single scattering albedo). All with shapes (time,wavelenghts,lat,lon). The script reads all the netcdfs in the specified path with dates thirty days before and thirty days after the time range specified, and interpolates the outputs to have one hour resolution.   (default: ./)

       cloud_datadir
            Directory where to read the cloud data. Netcdfs with the variable 'cdrem' (cloud droplet effective radius [um]).  Shape (time,lat,lon). The script reads all the netcdfs in the specified path with dates thirty days before and thirty days after the time range specified, and interpolates the outputs to have one hour resolution.   (default: ./)

       atmosphere_prefix
            Prefix in the name of the atmospheric input data. e.g. prefix.%Y%m%d-%H:%M:%S.nc (default:
            atm)

       aerosol_prefix
            Prefix in the name of the aerosol input data. e.g. prefix.%Y%m%d-%H:%M:%S.nc (default: aero)

       cloud_prefix
            Prefix in the name of the cloud input data. e.g. prefix.%Y%m%d-%H:%M:%S.nc (default: climatm)

       start_date
            Date to start the OASIM computation, in the same format as --dateformat. (default:
            20200102-07:00:00)

       end_date
            Date to end the OASIM computation, in the same format as --dateformat. (default:
            20200102-16:00:00)

       wavelengths_file
            File with the wavelenghts to be interpolated. (default:
            /g100_work/OGS23_PRACE_IT/csoto/DIIM/extern/OASIM_ATM/test/data/bin.txt)

       mask_file
            Path of the meshmask corresponding to input data (default:
            /g100_work/OGS_devC/V9C/RUNS_SETUP/PREPROC/MASK/meshmask.nc)

       location
            map to create work with all the Mediterranean Sea, if not, use format "lat,lon", e.g.
            "43.367,7.9" for Boussole and "35.49,12.47" for Lampedusa. (default: map)

       oasimlib_file
            file with liboasim-py.so (default: /g100_work/OGS23_PRACE_IT/csoto/DIIM/map/liboasim-py.so)

       oasim_config_file
            file with the oasim config.yaml (default:
            /g100_work/OGS23_PRACE_IT/csoto/DIIM/map/oasim_config.yaml)

    Returns:

        A dictionary with the following atributes:
                times
                    list with the datetimes of each point interpolated
                wl
                    wavelenghts
                lat
                    latitues
                lon
                    longitudes

            Interpolated atmospheric inputs of the oasim model:
                sp
                    surface pressure [Pa] (shape: (time_steps,lat,lon))
                msl
                    air pressure at mean sea level [Pa] (shape: (time_steps,lat,lon))
                ws10
                    wind Speed [m/s] (shape: (time_steps,lat,lon))
                tc03
                    total column ozone [kg m^-2] (shape: (time_steps,lat,lon))
                t2m
                    2 metre temperature [K] (shape: (time_steps,lat,lon))
                d2m2
                    metre dewpoint temperature [K] (shape: (time_steps,lat,lon))
                tcc
                    total cloud cover [0-100] (shape: (time_steps,lat,lon))
                tclw
                    total column cloud liquid water [kg m^-2] (shape: (time_steps,lat,lon))

            Interpolated claud inputs of the oasim model:
                cdrem
                    cloud droplet effective radius [um] (shape: (time_steps,lat,lon))

            Interpolated aerosol inputs of the oasim model:
                taua
                    aerosol optical thickness (shape: (time_steps,lat,lon))
                asymp
                    aerosol asymmetry parameter (shape: (time_steps,lat,lon))
                ssalb
                    aerosol single scattering albedo [-] (shape: (time_steps,lat,lon))

        The output of the oasim model:

                edout
                    contains binned direct downward irradiance [W m^-2] (shape: (time_steps,wavelenghts,lat,lon))
                esout
                    contains binned diffuse downward irradiance [W m^-2] (shape: (time_steps,wavelenghts,lat,lon))
    """

    #import timelist from data
    start_date = datetime.datetime.strptime(start_date,dateformat)
    end_date = datetime.datetime.strptime(end_date,dateformat)
    
    TL_ATM = TimeList.fromfilenames(TimeInterval((start_date - datetime.timedelta(days=1)).strftime(dateformat),\
                                                 (end_date + datetime.timedelta(days=1)).strftime(dateformat),\
                                                 dateformat = dateformat),\
                                    atmosphere_datadir , atmosphere_prefix+"*nc",\
                                    prefix= atmosphere_prefix + ".", dateformat = dateformat)

    
    TL_AER = TimeList.fromfilenames(TimeInterval((start_date - datetime.timedelta(days=16)).strftime(dateformat),\
                                                 (end_date + datetime.timedelta(days=16)).strftime(dateformat),\
                                                 dateformat = dateformat),\
                                    aerosol_datadir , aerosol_prefix+"*nc" ,\
                                    prefix= aerosol_prefix + ".", dateformat = dateformat)
    
    TL_CLD = TimeList.fromfilenames(TimeInterval((start_date - datetime.timedelta(days=30)).strftime(dateformat),\
                                                 (end_date + datetime.timedelta(days=30)).strftime(dateformat),\
                                                 dateformat = dateformat),\
                                    cloud_datadir, cloud_prefix+"*nc",\
                                    prefix= cloud_prefix + ".", dateformat = dateformat)
    
    #output time iframes

    dtnew = 60.*60. #time interval in seconds
    OUT_Timelist = DL.getTimeList(start_date,end_date, seconds=dtnew) #datelist to be interpolated

    #import wavelenghts
    wl = pd.read_csv(wavelengths_file, delim_whitespace=True, header=None).to_numpy()
    wl = np.mean(wl,1).astype(int)
    if TheMask is None: TheMask = Mask.from_file(mask_file)
    
    location_str = ("".join(location.strip())).lower()

    if location_str == 'map':
        mesh_lat,mesh_lon = np.meshgrid(TheMask.lat,TheMask.lon)
        mesh_lat,mesh_lon = mesh_lat.T,mesh_lon.T
        len_lat,len_lon = mesh_lat.shape
    
        n_points = len_lat*len_lon
        lat = mesh_lat.reshape((n_points)).astype(np.float64)
        lon = mesh_lon.reshape((n_points)).astype(np.float64)
    else:
        lat = lats
        lon = lons
        len_lat = len(lats)
        len_lon = len(lons)
        n_points = len(lats)

    olib = oasim_lib(oasimlib_file, oasim_config_file, lat, lon)
    cunit = calc_unit(n_points, olib)
    dt = (OUT_Timelist[1]-OUT_Timelist[0]).total_seconds() #seconds
    
    #spatial mash
    ji_sel, jj_sel = TheMask.convert_lon_lat_to_indices(lon=lon, lat=lat)
    points = np.arange(1,n_points+1)
    npoints=n_points
    nframe = len(OUT_Timelist)
    nwavelengths = len(wl)
    
    def init_map(w=False,location_str_='map'):
        if location_str_ == 'map':
            if w == False:
                return np.zeros((nframe,len_lat,len_lon))
            else:
                return np.zeros((nframe,nwavelengths,len_lat,len_lon))
        else:
            if w == False:
                return np.zeros((nframe,len_lat))
            else:
                return np.zeros((nframe,nwavelengths,len_lat))

    sp,msl,u10,v10,ws10,tco3,t2m,d2m,tcc,tclw,cdrem = [init_map(location_str_=location_str) for _ in range(11)]
    #carefull,output and input of cunit.monrad will be np.array.shape = (npoints,nwavelenghts)
    taua,asymp,ssalb,edout,esout = [init_map(w=True,location_str_=location_str) for _ in range(5)]
    
    for it,tt in enumerate(OUT_Timelist[:]):
        ############################################ ATMOSPHERE #############################################
        ##############################################################################################
        before,after,T_interp = Time_Interpolation(tt,TL_ATM.Timelist)

        ncA,ncB = Load_Data(atmosphere_datadir, TL_ATM.Timelist, before, after,atmosphere_prefix+".",dateformat)
        
        variables,variable_names = [sp,msl,u10,v10,tco3,t2m,d2m,tclw],['sp','msl','u10','v10','tco3','t2m','d2m','tclw']
        for variable,variable_name in zip(variables,variable_names):
            if location == 'map':
                Before_DATA, After__DATA = np.array(ncB.variables[variable_name]).astype(np.float32),np.array(ncA.variables[variable_name]).astype(np.float32)
            else:
                Before_DATA, After__DATA = np.diagonal(ncB.variables[variable_name][ji_sel,jj_sel]).astype(np.float32),\
                    np.diagonal(ncA.variables[variable_name][ji_sel,jj_sel]).astype(np.float32)
            variable[it] = (1-T_interp)*Before_DATA + T_interp*After__DATA
        
        ws10[it]  = np.sqrt(u10[it]*u10[it]+v10[it]*v10[it])
        
        if location == 'map':
            Before_DATA, After__DATA = np.array(ncA.variables['tcc']).astype(np.float32),np.array(ncA.variables['tcc']).astype(np.float32)
        else:
            Before_DATA, After__DATA = np.diagonal(ncA.variables['tcc'][ji_sel,jj_sel]).astype(np.float32),\
                np.diagonal(ncA.variables['tcc'][ji_sel,jj_sel]).astype(np.float32)
        tcc[it] = 100.*((1-T_interp)*Before_DATA + T_interp*After__DATA)

        ############################################ CLOUD ######################################
        ###############################################################################################

        before,after,T_interp = Time_Interpolation(tt,TL_CLD.Timelist)

        ncA,ncB = Load_Data(cloud_datadir, TL_CLD.Timelist, before, after,cloud_prefix+".",dateformat)

        if location == 'map':
            Before_DATA, After__DATA = np.array(ncB.variables['cdrem']).astype(np.float32),np.array(ncA.variables['cdrem']).astype(np.float32)
        else:
            Before_DATA, After__DATA = np.diagonal(ncB.variables['cdrem'][ji_sel,jj_sel]).astype(np.float32),\
                np.diagonal(ncA.variables['cdrem'][ji_sel,jj_sel]).astype(np.float32)
        cdrem[it] = (1-T_interp)*Before_DATA + T_interp*After__DATA
        

        ########################################### AEROSOL #####################################
        ###############################################################################################

        before,after,T_interp = Time_Interpolation(tt,TL_AER.Timelist)
        ncA,ncB = Load_Data(aerosol_datadir, TL_AER.Timelist, before, after,aerosol_prefix+".",dateformat)
        
        variables, variable_names = [taua,asymp,ssalb],['taua','asymp','ssalb']
        len_wl = len(wl)
        for variable,variable_name in zip(variables,variable_names):
            if location == 'map':
                Before_DATA, After__DATA = np.array(ncB.variables[variable_name][:len_wl]),np.array(ncA.variables[variable_name][:len_wl])
            else:
                Before_DATA, After__DATA = np.diagonal(ncB.variables[variable_name][:len_wl,ji_sel,jj_sel],axis1=1,axis2=2).astype(np.float32),\
                    np.diagonal(ncA.variables[variable_name][:len_wl,ji_sel,jj_sel],axis1=1,axis2=2)
            variable[it] = (1-T_interp)*Before_DATA + T_interp*After__DATA

        iyr = tt.year
        iday = datetime.date(tt.year,tt.month,tt.day).timetuple().tm_yday 
        sec   = tt.second+tt.minute*60+tt.hour*3600 
        sec_b = sec - dt/2
        sec_e = sec + dt/2
        
        edout_inter, esout_inter = cunit.monrad(points, iyr, iday, sec_b, sec_e,
                                                sp[it].reshape((npoints)), msl[it].reshape((npoints)),\
                                                ws10[it].reshape((npoints)), tco3[it].reshape((npoints)),\
                                                t2m[it].reshape((npoints)), d2m[it].reshape((npoints)),\
                                                tcc[it].reshape((npoints)), tclw[it].reshape((npoints)),\
                                                cdrem[it].reshape((npoints)),\
                                                (taua[it].reshape((nwavelengths,npoints))).T,\
                                                (asymp[it].reshape((nwavelengths,npoints))).T,\
                                                (ssalb[it].reshape((nwavelengths,npoints))).T )

        if location == 'map':
            edout[it],esout[it] = (edout_inter.T).reshape((nwavelengths,len_lat,len_lon)),\
                (esout_inter.T).reshape((nwavelengths,len_lat,len_lon))
        else:
            edout[it],esout[it] = edout_inter.T,esout_inter.T
            
        ncA.close()
        ncB.close()

    return {
        'times':OUT_Timelist,
        'wl':wl,
        'lat':TheMask.lat,
        'lon':TheMask.lon,
        'sp':sp,
        'msl':msl,
        'u10':u10,
        'v10':v10,
        'ws10':ws10,
        'tco3':tco3,
        't2m':t2m,
        'd2m':d2m,
        'tcc':tcc,
        'tclw':tclw,
        'cdrem':cdrem,
        'taua':taua,
        'asymp':asymp,
        'ssalb':ssalb,
        'edout':edout,
        'esout':esout
    }

def creat_oasim_netcdf(times,wl,lat,lon,sp,msl,ws10,tco3,t2m,d2m,tcc,tclw,cdrem,taua,asymp,ssalb,edout,esout,output_name='oasim_map.nc',output_datadir='./',location='map',\
                       mean_value=True,\
                       sp_save = False,\
                       msl_save = False,\
                       ws10_save = False,\
                       tco3_save = False,\
                       t2m_save = False,\
                       d2m_save = False,\
                       tcc_save = False,\
                       tclw_save = False,\
                       cdrem_save = False,\
                       taua_save = False,\
                       asymp_save = False,\
                       ssalb_save = False,\
                       edout_save = True,\
                       esout_save = True,\
                       site_name = 'MED',\
):
    """
    Creates a netcdf with the outputs of the oasim model. 
    
    Parameters:
            times
                List of datetimes for dimension time. 
            wl
                The value of the wavelenghts (nm).
            lat
                Latitues of the map (1 dimensional array).
            lon
                Longitudes of the map (1 dimensional array).
                    
        Interpolated atmospheric inputs of the oasim model:
            sp
                surface pressure [Pa] (shape: (time_steps,lat,lon))
            msl
                air pressure at mean sea level [Pa] (shape: (time_steps,lat,lon))
            ws10
                wind Speed [m/s] (shape: (time_steps,lat,lon))
            tc03
                total column ozone [kg m^-2] (shape: (time_steps,lat,lon))
            t2m
                2 metre temperature [K] (shape: (time_steps,lat,lon))
            d2m2
                metre dewpoint temperature [K] (shape: (time_steps,lat,lon))
            tcc
                total cloud cover [0-100] (shape: (time_steps,lat,lon))
            tclw
                total column cloud liquid water [kg m^-2] (shape: (time_steps,lat,lon))

        Interpolated claud inputs of the oasim model:
            cdrem
                cloud droplet effective radius [um] (shape: (time_steps,lat,lon))

        Interpolated aerosol inputs of the oasim model:
            taua
                aerosol optical thickness (shape: (time_steps,lat,lon))
            asymp
                aerosol asymmetry parameter (shape: (time_steps,lat,lon))
            ssalb
                aerosol single scattering albedo [-] (shape: (time_steps,lat,lon))

        The output of the oasim model:
            edout
                contains binned direct downward irradiance [W m^-2] (shape: (time_steps,wavelenghts,lat,lon))
            esout
                contains binned diffuse downward irradiance [W m^-2] (shape: (time_steps,wavelenghts,lat,lon))

        output_name
            name of the output file as output_name.nc. Default: oasim_map.nc
        output_datadir
            data directory to store the output file. Default: ./
    """
    len_lat = len(lat)
    len_lon = len(lon)
    nwavelengths = len(wl)
    nframe = len(times)
    
    ncfile = netCDF4.Dataset(output_datadir + '/' + output_name, "w", format="NETCDF4")
    ncfile.start_date = times[0].strftime('%Y-%m-%d')
    ncfile.stop_date =  times[-1].strftime('%Y-%m-%d')
    ncfile.creation_date = datetime.datetime.utcnow().strftime('%a %b %d %Y')
    ncfile.creation_time = datetime.datetime.utcnow().strftime('%H:%M:%S UTC')
    if location == 'map':
        ncfile.westernmost_longitude = lon[0]
        ncfile.easternmost_longitude = lon[-1]
        ncfile.southernmost_latitude = lat[0]
        ncfile.northernmost_latitude = lat[-1]
        ncfile.grid_resolution =  'Approximately {} degrees'.format(np.abs(lon[1]-lon[0]))

        
    ncfile.institution = 'OGS'
    ncfile.site_name = site_name
    ncfile.model = 'OASIM'

    if mean_value == True:
        time_dim = ncfile.createDimension('time',size=1)
    else:
        time_dim = ncfile.createDimension('time',size=len(times))
    wavelength_dim = ncfile.createDimension('wavelength',size=nwavelengths) 
    lat_dim = ncfile.createDimension('lat',size=len_lat)
    lon_dim = ncfile.createDimension('lon',size=len_lon)
    if location != 'map':
        lat_lon_dim = ncfile.createDimension('lat,lon',size=len_lat)
        lat_lon_var = ncfile.createVariable('lat,lon','f4',('lat,lon'))
    
    time_var = ncfile.createVariable('time','f4',('time',))
    wavelength_var = ncfile.createVariable('wavelength','f4',('wavelength',))
    lat_var = ncfile.createVariable('lat','f4',('lat',))
    lon_var = ncfile.createVariable('lon','f4',('lon',))
    

    def createVariableIf(variable,location='map',w=False):
        if location == 'map':
            if w == False:
                variable_var = ncfile.createVariable(variable,'f4',('time','lat','lon'))
            else:
                variable_var = ncfile.createVariable(variable,'f4',('time','wavelength','lat','lon'))
        else:
            if w == False:
                variable_var = ncfile.createVariable(variable,'f4',('time','lat,lon'))
            else:
                variable_var = ncfile.createVariable(variable,'f4',('time','wavelength','lat,lon'))
        return variable_var
            
    
    if sp_save: sp_var = createVariableIf('sp',location=location,w=False)
    if msl_save: msl_var = createVariableIf('msl',location=location,w=False)
    if ws10_save: ws10_var = createVariableIf('10ws',location=location,w=False)
    if tco3_save: tco3_var = createVariableIf('tcs3',location=location,w=False)
    if t2m_save: t2m_var = createVariableIf('t2m',location=location,w=False)
    if d2m_save: d2m_var = createVariableIf('d2m',location=location,w=False)
    if tcc_save: tcc_var = createVariableIf('tcc',location=location,w=False)
    if tclw_save: tclw_var = createVariableIf('tclw',location=location,w=False)
    if cdrem_save: cdrem_var = createVariableIf('cdrem',location=location,w=False)
    
    if taua_save: taua_var = createVariableIf('taua',location=location,w=True)
    if asymp_save: asymp_var = createVariableIf('asymp',location=location,w=True)
    if ssalb_save: ssalb_var = createVariableIf('ssalb',location=location,w=True)
    if edout_save: edout_var = createVariableIf('edout',location=location,w=True)
    if esout_save: esout_var = createVariableIf('esout',location=location,w=True)
    
    def fill_atributes(var,long_name,units,variable_CDS,short_name,comments = False):
        var.long_name = long_name
        var.units = units
        var.variable_CDS = variable_CDS
        var.short_name = short_name
        if comments != False: var.comments = comments
        
    if sp_save: fill_atributes(sp_var,'Surface pressure','Pa','surface_pressure','sp')
    if msl_save: fill_atributes(msl_var,'Mean sea level pressure','Pa','mean_sea_level_pressure','msl')
    if ws10_save: fill_atributes(ws10_var,'10 metre wind speed','m s**-1','10 metre Speed wind','10ws','sqrt((10u)**2 + (10v)**2)')
    if tco3_save: fill_atributes(tco3_var,'Total column ozone','kg m**-2','total_column_ozone','tco3')
    if t2m_save: fill_atributes(t2m_var,'2 meters temperature','K','2m_temperature','t2m') 
    if d2m_save: fill_atributes(d2m_var,'2 metre dewpoint temperature','K','2m_dewpoint_temperature','d2m')
    if tcc_save: fill_atributes(tcc_var,'Total cloud cover','(0 - 1)','total_cloud_cover','tcc')
    if tclw_save: fill_atributes(tclw_var,'Total column cloud liquid water','kg m**-2','total_column_cloud_liquid_water','tclw')
    if cdrem_save: fill_atributes(cdrem_var,'cloud droplet effective radius','um','cloud_droplet_effective_radius','cdrem') 
    if taua_save: fill_atributes(taua_var,'aerosol optical thickness','m','aerosol_optical_thickness','taua')
    if asymp_save: fill_atributes(asymp_var,'aerosol asymmetry parameter','-','aerosol_asymmetry_parameter','asymp')
    if ssalb_save: fill_atributes(ssalb_var,'aerosol single scattering albedo','-','aerosol_single_scattering_albedo','ssalb')
    if edout_save: fill_atributes(edout_var,'contains binned direct downward irradiance','W m^-2','direct_downward_irradiance','edout') 
    if esout_save: fill_atributes(esout_var,'contains binned diffuse downward irradiance','W m^-2','diffuse_downward_irradiance','esout')

    time_var.units = "seconds since " +  times[0].strftime('%Y-%m-%d %H:%M:%S')
    time_var.long_name = 'time'
    
    lat_var.long_name = 'latitude'
    lat_var.units = 'degrees_north'
    lon_var.long_name = 'longitude'
    lon_var.units = 'degrees_east'

    if mean_value == True:
        time_var[:] = [(times[-1] - times[0]).seconds/2]
    else:
        time_var[:] = [(times[j] - times[0]).seconds for j in range(len(times))]
    wavelength_var[:] = wl
    lat_var[:] = lat
    lon_var[:] = lon

    def mean_value_if(variable,mean_value=False):
        if mean_value:
            return np.mean(variable,axis=0)
        else:
            return variable
        
    if sp_save: sp_var[:] = mean_value_if(sp,mean_value)
    if msl_save: msl_var[:] = mean_value_if(msl,mean_value)
    if ws10_save: ws10_var[:] = mean_value_if(ws10,mean_value)
    if tco3_save: tco3_var[:] = mean_value_if(tco3,mean_value)
    if t2m_save: t2m_var[:] = mean_value_if(t2m,mean_value)
    if d2m_save: d2m_var[:] = mean_value_if(d2m,mean_value)
    if tcc_save: tcc_var[:] = mean_value_if(tcc,mean_value)
    if tclw_save: tclw_var[:] = mean_value_if(tclw,mean_value)
    if cdrem_save: cdrem_var[:] = mean_value_if(cdrem,mean_value)
    
    if taua_save: taua_var[:] = mean_value_if(taua,mean_value)
    if asymp_save: asymp_var[:] = mean_value_if(asymp,mean_value)
    if ssalb_save: ssalb_var[:] = mean_value_if(ssalb,mean_value)
    
    if edout_save: edout_var[:] = mean_value_if(edout,mean_value)
    if esout_save: esout_var[:] = mean_value_if(esout,mean_value)

    if location != 'map':
        lat_lon_var[:] = np.arange(len_lat)


    del times
    del wl
    del lat
    del lon
    del sp
    del msl
    del ws10
    del tco3
    del t2m
    del d2m
    del tcc
    del tclw
    del cdrem
    del taua
    del asymp
    del ssalb
    del edout
    del esout
    ncfile.close()


if __name__ == '__main__':

    conf_ = read_arguments()
    
    location_str = ("".join(conf_['location'].strip())).lower()
    if location_str == 'map':
        lats = []
        lons = []
    elif location_str == 'file':
        latsAndLons = pd.read_csv(conf_['location_file'],sep=' ',comment = '#',names = ['lats','lons'])
        lats = np.array( latsAndLons['lats'], dtype=np.float64, order='F')
        lons = np.array( latsAndLons['lons'], dtype=np.float64, order='F')
        location_str = 'list'
        
    else:
        lats = np.array([float(conf_['location'].split(',')[0])], dtype=np.float64, order='F')
        lons = np.array([float(conf_['location'].split(',')[1])], dtype=np.float64, order='F')
        location_str = 'list'
        
    oasim_output = getting_oasim_output(dateformat=conf_['dateformat'],\
                                        atmosphere_datadir=conf_['atmosphere_datadir'],\
                                        aerosol_datadir=conf_['aerosol_datadir'],\
                                        cloud_datadir=conf_['cloud_datadir'],\
                                        atmosphere_prefix=conf_['atmosphere_prefix'],\
                                        aerosol_prefix=conf_['aerosol_prefix'],\
                                        cloud_prefix=conf_['cloud_prefix'],\
                                        start_date=conf_['start_date'],\
                                        end_date=conf_['end_date'],\
                                        wavelengths_file=conf_['wavelengths_file'],\
                                        mask_file=conf_['mask_file'],\
                                        location=location_str,\
                                        oasimlib_file=conf_['oasimlib_file'],\
                                        oasim_config_file=conf_['oasim_config_file'],\
                                        lats = lats,\
                                        lons = lons
                                        )
    output_name = conf_['output_prefix'] + \
        '_' +oasim_output['times'][0].strftime('%Y%m%d-%H:%M:%S')+ '-' + \
        oasim_output['times'][-1].strftime('%Y%m%d-%H:%M%S') +'.nc'
    
    creat_oasim_netcdf(times=oasim_output['times'],
                       wl=oasim_output['wl'],
                       lat=oasim_output['lat'],
                       lon=oasim_output['lon'],
                       sp=oasim_output['sp'],
                       msl=oasim_output['msl'],
                       ws10=oasim_output['ws10'],
                       tco3=oasim_output['tco3'],
                       t2m=oasim_output['t2m'],
                       d2m=oasim_output['d2m'],
                       tcc=oasim_output['tcc'],
                       tclw=oasim_output['tclw'],
                       cdrem=oasim_output['cdrem'],
                       taua=oasim_output['taua'],
                       asymp=oasim_output['asymp'],
                       ssalb=oasim_output['ssalb'],
                       edout=oasim_output['edout'],
                       esout=oasim_output['esout'],
                       output_name=output_name,
                       output_datadir=conf_['output_datadir'])
    
