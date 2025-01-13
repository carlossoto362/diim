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
import netCDF4 as NC
import netCDF4
import argparse
import os


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
        print(self._ptr)

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
    ncB = NC.Dataset(DATADIR + '/' + Before_File,'r')
    ncA = NC.Dataset(DATADIR + '/' + After__File,'r')
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
                        help = '''Directory where to read the atmospheric data.''',
                        default = '/g100_work/OGS_devC/NECCTON/OPTICS/2020/01'
                        )
    parser.add_argument('--aerosol_datadir', '-iaer',
                        type = str,
                        required = False,
                        help = '''Directory where to read the aerosol data.''',
                        default = '/g100_work/OGS_devC/NECCTON/OPTICS'
                        )
    parser.add_argument('--cloud_datadir', '-icld',
                        type = str,
                        required = False,
                        help = '''Directory where to read the claud data.''',
                        default = '/g100_work/OGS_devC/NECCTON/OPTICS'
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
                        help = '''map to create work with all the Mediterranean Sea, if not, use format "lat,lon", e.g. "43.367,7.9" for Boussole and "35.49,12.47" for Lampedusa.''',
                        default = 'map'
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
    parser.add_argument('--reflectance_datadir', '-irrs',
                        type = str,
                        required = False,
                        help = '''Directory where to read the remote sensing reflectance data.''',
                        default = HOME_PATH + '/../rrs_data/V11C/SAT/WEEKLY_24'
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
    

if __name__ == '__main__':

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
        
    #import timelist from data
    start_date = datetime.datetime.strptime(conf_['start_date'],conf_['dateformat'])
    end_date = datetime.datetime.strptime(conf_['end_date'],conf_['dateformat'])
    
    TL_ATM = TimeList.fromfilenames(TimeInterval((start_date - datetime.timedelta(days=1)).strftime(conf_['dateformat']),\
                                                 (end_date + datetime.timedelta(days=1)).strftime(conf_['dateformat']),\
                                                 dateformat = conf_['dateformat']),\
                                    conf_['atmosphere_datadir'] , conf_['atmosphere_prefix']+"*nc",\
                                    prefix= conf_['atmosphere_prefix'] + ".", dateformat = conf_['dateformat'])
    
    TL_AER = TimeList.fromfilenames(TimeInterval((start_date - datetime.timedelta(days=30)).strftime(conf_['dateformat']),\
                                                 (end_date + datetime.timedelta(days=30)).strftime(conf_['dateformat']),\
                                                 dateformat = conf_['dateformat']),\
                                    conf_['aerosol_datadir'] , conf_['aerosol_prefix']+"*nc" ,\
                                    prefix= conf_['aerosol_prefix'] + ".", dateformat = conf_['dateformat'])
    
    TL_CLD = TimeList.fromfilenames(TimeInterval((start_date - datetime.timedelta(days=30)).strftime(conf_['dateformat']),\
                                                 (end_date + datetime.timedelta(days=30)).strftime(conf_['dateformat']),\
                                                 dateformat = conf_['dateformat']),\
                                    conf_['cloud_datadir'], conf_['cloud_prefix']+"*nc",\
                                    prefix= conf_['cloud_prefix'] + ".", dateformat = conf_['dateformat'])
    
    #output time iframes
    dtnew = 60.*60. #time interval in seconds
    OUT_Timelist = DL.getTimeList(conf_['start_date'],conf_['end_date'], seconds=dtnew) #datelist to be interpolated
        
    #import wavelenghts
    wl = pd.read_csv(conf_['wavelengths_file'], delim_whitespace=True, header=None).to_numpy()
    wl = np.mean(wl,1).astype(int)
    
    TheMask = Mask(conf_['mask_file'])
    
    location = ("".join(conf_['location'].strip())).lower()
    if location != 'map':
        lat = np.array([int(location.split(','))[0]], dtype=np.float64, order='F')
        lon = np.array([location.split(',')[1]], dtype=np.float64, order='F')
        len_lat = 1
        len_lon = 1
        n_points = 1
    else:
        mesh_lat,mesh_lon = np.meshgrid(TheMask.lat,TheMask.lon)
        mesh_lat,mesh_lon = mesh_lat.T,mesh_lon.T
        len_lat,len_lon = mesh_lat.shape
    
        n_points = len_lat*len_lon
        lat = mesh_lat.reshape((n_points)).astype(np.float64)
        lon = mesh_lon.reshape((n_points)).astype(np.float64)

    olib = oasim_lib(conf_['oasimlib_file'], conf_['oasim_config_file'], lat, lon)
    cunit = calc_unit(n_points, olib)
    dt = (OUT_Timelist[1]-OUT_Timelist[0]).total_seconds() #seconds
    
    #spatial mash
    ji_sel, jj_sel = TheMask.convert_lon_lat_to_indices(lon[0], lat[0])
    points = np.arange(1,n_points+1)
    npoints=n_points
    nframe = len(OUT_Timelist)
    nwavelengths = len(wl)
    def init_map(w=False):
        if w == False:
            return np.zeros((nframe,len_lat,len_lon))
        else:
            return np.zeros((nframe,nwavelengths,len_lat,len_lon))

    sp,msl,u10,v10,ws10,tco3,t2m,d2m,tcc,tclw,cdrem = [init_map() for _ in range(11)]
    #carefull,output and input of cunit.monrad will be np.array.shape = (npoints,nwavelenghts)
    taua,asymp,ssalb,edout,esout = [init_map(w=True) for _ in range(5)] 
    
    for it,tt in enumerate(OUT_Timelist[:]):

        ############################################ ATM #############################################
        ##############################################################################################
        before,after,T_interp = Time_Interpolation(tt,TL_ATM.Timelist)

        ncA,ncB = Load_Data(conf_['atmosphere_datadir'], TL_ATM.Timelist, before, after,conf_['atmosphere_prefix']+".",conf_['dateformat'])
        
        variables,variable_names = [sp,msl,u10,v10,tco3,t2m,d2m,tclw],['sp','msl','u10','v10','tco3','t2m','d2m','tclw']
        for variable,variable_name in zip(variables,variable_names):
            Before_DATA, After__DATA = np.array(ncA.variables[variable_name]).astype(np.float32),np.array(ncA.variables[variable_name]).astype(np.float32)
            if location == 'map':
                variable[it] = (1-T_interp)*Before_DATA + T_interp*After__DATA
            else:
                variable[it] = (1-T_interp)*Before_DATA[jj_sel,ji_sel] + T_interp*After__DATA[jj_sel,ji_sel]
        
        ws10[it]  = np.sqrt(u10[it]*u10[it]+v10[it]*v10[it])
        Before_DATA, After__DATA = np.array(ncA.variables['tcc']).astype(np.float32),np.array(ncA.variables['tcc']).astype(np.float32)
        if location == 'map':
            tcc[it] = 100.*((1-T_interp)*Before_DATA + T_interp*After__DATA)
        else:
            tcc[it] = 100.*((1-T_interp)*Before_DATA[jj_sel,ji_sel] + T_interp*After__DATA[jj_sel,ji_sel])
            
        ############################################ MODIS CLOUD ######################################
        ###############################################################################################

        before,after,T_interp = Time_Interpolation(tt,TL_CLD.Timelist)

        ncA,ncB = Load_Data(conf_['cloud_datadir'], TL_CLD.Timelist, before, after,conf_['cloud_prefix']+".",conf_['dateformat'])
        Before_DATA, After__DATA = np.array(ncA.variables['cdrem']).astype(np.float32),np.array(ncA.variables['cdrem']).astype(np.float32)
        
        if location == 'map':
            cdrem[it] = (1-T_interp)*Before_DATA + T_interp*After__DATA
        else:
            cdrem[it] = (1-T_interp)*Before_DATA[jj_sel,ji_sel] + T_interp*After__DATA[jj_sel,ji_sel]
        
            
        ########################################### MODIS AEROSOL #####################################
        ###############################################################################################
        
        before,after,T_interp = Time_Interpolation(tt,TL_AER.Timelist)
        ncA,ncB = Load_Data(conf_['aerosol_datadir'], TL_AER.Timelist, before, after,conf_['aerosol_prefix']+".",conf_['dateformat'])
        
        variables, variable_names = [taua,asymp,ssalb],['taua','asymp','ssalb']
        for variable,variable_name in zip(variables,variable_names):
            Before_DATA, After__DATA = np.array(ncA.variables[variable_name][:len(wl)]).astype(np.float32),np.array(ncA.variables[variable_name][:len(wl)]).astype(np.float32)
            
            if location == 'map':
                variable[it] = (1-T_interp)*Before_DATA + T_interp*After__DATA
            else:
                variable[it] = (1-T_interp)*Before_DATA[:,:,jj_sel,ji_sel] + T_interp*After__DATA[:,:,jj_sel,ji_sel]

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
        
        edout[it],esout[it] = (edout_inter.T).reshape((nwavelengths,len_lat,len_lon)),\
            (esout_inter.T).reshape((nwavelengths,len_lat,len_lon))
        ncA.close()
        ncB.close()
        
        
    ncfile = netCDF4.Dataset(conf_['output_datadir'] + '/' + conf_['output_prefix'] + \
                             '_' +OUT_Timelist[0].strftime('%Y%m%d_%H:%M:%S')+ '-' + \
                             OUT_Timelist[-1].strftime('%Y%m%d_%H:%M%S') +'.nc', "w", format="NETCDF4")
    ncfile.start_date = OUT_Timelist[0].strftime('%Y-%m-%d')
    ncfile.stop_date =  OUT_Timelist[-1].strftime('%Y-%m-%d')
    ncfile.creation_date = datetime.datetime.utcnow().strftime('%a %b %d %Y')
    ncfile.creation_time = datetime.datetime.utcnow().strftime('%H:%M:%S UTC')
    if location == 'map':
        ncfile.westernmost_longitude = -8.875
        ncfile.easternmost_longitude = 36.291668
        ncfile.southernmost_latitude = 30.1875
        ncfile.northernmost_latitude = 45.979168
        ncfile.grid_resolution =  'Approximately 0.04166 degrees'
    else:
        ncfile.latitude = location.split(',')[0]
        ncfile.longitude = location.split(',')[1]
        
    ncfile.institution = 'OGS'
    ncfile.site_name = 'MED'
    ncfile.model = 'OASIM'
    
    time_dim = ncfile.createDimension('time',size=nframe)
    wavelength_dim = ncfile.createDimension('wavelength',size=nwavelengths) 
    lat_dim = ncfile.createDimension('lat',size=len_lat)
    lon_dim = ncfile.createDimension('lon',size=len_lon)
    
    
    time_var = ncfile.createVariable('time','f4',('time',))
    wavelength_var = ncfile.createVariable('wavelength','f4',('wavelength',))
    lat_var = ncfile.createVariable('lat','f4',('lat',))
    lon_var = ncfile.createVariable('lon','f4',('lon',))
    
    sp_var = ncfile.createVariable('sp','f4',('time','lat','lon'))
    msl_var = ncfile.createVariable('msl','f4',('time','lat','lon'))
    u10_var = ncfile.createVariable('10u','f4',('time','lat','lon'))
    v10_var = ncfile.createVariable('10v','f4',('time','lat','lon'))
    ws10_var = ncfile.createVariable('10ws','f4',('time','lat','lon'))
    tco3_var = ncfile.createVariable('tcs3','f4',('time','lat','lon'))
    t2m_var = ncfile.createVariable('t2m','f4',('time','lat','lon'))
    d2m_var = ncfile.createVariable('d2m','f4',('time','lat','lon'))
    tcc_var = ncfile.createVariable('tcc','f4',('time','lat','lon'))
    tclw_var = ncfile.createVariable('tclw','f4',('time','lat','lon'))
    cdrem_var = ncfile.createVariable('cdrem','f4',('time','lat','lon'))
    
    taua_var = ncfile.createVariable('taua','f4',('time','wavelength','lat','lon'))
    asymp_var = ncfile.createVariable('asymp','f4',('time','wavelength','lat','lon'))
    ssalb_var = ncfile.createVariable('ssalb','f4',('time','wavelength','lat','lon'))
    edout_var = ncfile.createVariable('edout','f4',('time','wavelength','lat','lon'))
    esout_var = ncfile.createVariable('esout','f4',('time','wavelength','lat','lon'))
    
    def fill_atributes(var,long_name,units,variable_CDS,short_name,comments = False):
        var.long_name = long_name
        var.units = units
        var.variable_CDS = variable_CDS
        var.short_name = short_name
        if comments != False: var.comments = comments
        
    fill_atributes(sp_var,'Surface pressure','Pa','surface_pressure','sp')
    fill_atributes(msl_var,'Mean sea level pressure','Pa','mean_sea_level_pressure','msl')
    fill_atributes(u10_var,'10 metre U wind component','m s**-1','10m_u_component_of_wind','10u')
    fill_atributes(v10_var,'10 metre V wind component','m s**-1','10m_v_component_of_wind','10v')
    fill_atributes(ws10_var,'10 metre wind speed','m s**-1','10 metre Speed wind','10ws','sqrt((10u)**2 + (10v)**2)')
    fill_atributes(tco3_var,'Total column ozone','kg m**-2','total_column_ozone','tco3')
    fill_atributes(t2m_var,'2 meters temperature','K','2m_temperature','t2m') 
    fill_atributes(d2m_var,'2 metre dewpoint temperature','K','2m_dewpoint_temperature','d2m')
    fill_atributes(tcc_var,'Total cloud cover','(0 - 1)','total_cloud_cover','tcc')
    fill_atributes(tclw_var,'Total column cloud liquid water','kg m**-2','total_column_cloud_liquid_water','tclw')
    fill_atributes(cdrem_var,'cloud droplet effective radius','um','cloud_droplet_effective_radius','cdrem') 
    fill_atributes(taua_var,'aerosol optical thickness','m','aerosol_optical_thickness','taua')
    fill_atributes(asymp_var,'aerosol asymmetry parameter','-','aerosol_asymmetry_parameter','asymp')
    fill_atributes(ssalb_var,'aerosol single scattering albedo','-','aerosol_single_scattering_albedo','ssalb') 
    fill_atributes(edout_var,'contains binned direct downward irradiance','W m^-2','direct_downward_irradiance','edout') 
    fill_atributes(esout_var,'contains binned diffuse downward irradiance','W m^-2','diffuse_downward_irradiance','esout')

    time_var.units = "seconds since 2020-01-02 00:00:00"
    time_var.long_name = 'time'
    
    lat_var.long_name = 'latitude'
    lat_var.units = 'degrees_north'
    lon_var.long_name = 'longitude'
    lon_var.units = 'degrees_east'
    
    time_var[:] = netCDF4.date2num(OUT_Timelist,time_var.units)
    wavelength_var[:] = wl
    lat_var[:] = TheMask.lat
    lon_var[:] = TheMask.lon
    
    sp_var[:,:,:] = sp
    msl_var[:,:,:] = msl
    u10_var[:,:,:] = u10
    v10_var[:,:,:] = v10
    ws10_var[:,:,:] = ws10
    tco3_var[:,:,:] = tco3
    t2m_var[:,:,:] = t2m
    d2m_var[:,:,:] = d2m
    tcc_var[:,:,:] = tcc
    tclw_var[:,:,:] = tclw
    cdrem_var[:,:,:] = cdrem
    
    taua_var[:,:,:,:] = taua
    asymp_var[:,:,:,:] = asymp
    ssalb_var[:,:,:,:] = ssalb
    edout_var[:,:,:,:] = edout
    esout_var[:,:,:,:] = esout
    
    ncfile.close()
