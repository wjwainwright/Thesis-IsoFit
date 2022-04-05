try:
    runCount += 1
except:
    isoIn = False
    clIn = False
    cataIn = False
    closePlots = False
    resultsIn = False
    clusterList = []
    clusters=[]
    isochrones = []
    isoList = []
    catalogue = []
    runCount = 1

class resultClusterObj:
    def __init__(self,cl):
        import numpy as np
        
        #Automatically populates variables based on those from the cluster it was given, except the data arrays
        global properties
        
        #List of all of the variables defined for the cluster cl, strips out the __functions__
        properties = [a for a in dir(cl) if not a.startswith('_')]
        for prop in properties:
            #Saves all 'number' type variables to the memory of the result cluster object
            if eval(f"type(cl.{prop})") == float or eval(f"type(cl.{prop})") == np.float64 or eval(f"type(cl.{prop})") == int:
                exec(f"self.{prop} = float(cl.{prop})")
            elif eval(f"type(cl.{prop})") == str:
                exec(f"self.{prop} = cl.{prop}")
                
        #Manually defined properties
        self.name = cl.name
        self.clType = cl.clType

class clusterObj:
    def __init__(self,name='genericCluster',basedir='clusters/',brightThreshold=15):
        #Declare instance variables
        self.basedir = basedir
        self.dataPath = self.basedir + f"{name}/data/"
        self.imgPath = self.basedir + f"{name}/plots/"
        self.unfilteredWide = []
        self.unfilteredNarrow = []
        self.filtered = []
        self.mag = []
        self.iso = []
        self.condensed = []
        self.condensed0 = []
        self.condensedInit=[]
        self.unfilteredBright = []
        self.filteredBright = []
        self.brightmag = []
        self.distFiltered = []
        self.binaries = []
        self.stars = []
        self.brightThreshold = brightThreshold
        self.mean_par = 0
        self.stdev_par = 0
        self.mean_ra = 0
        self.mean_dec = 0
        self.stdev_ra = 0
        self.stdev_dec = 0
        self.mean_pmra = 0
        self.stdev_pmra = 0
        self.mean_pmdec = 0
        self.stdev_pmdec = 0
        self.mean_a_g = 0
        self.stdev_a_g = 0
        self.mean_e_bp_rp = 0
        self.stdev_e_bp_rp = 0
        self.mean_par_over_ra = 0
        self.stdev_par_over_ra = 0
        self.dist_mod = 0
        self.turnPoint = 0
        self.reddening = 0
        self.radDist = 0
        self.massLoaded = False
        
        #Catalogued properties
        self.name = name
        self.clType = "None"
        self.pmra_min = -99
        self.pmra_max = -99
        self.pmdec_min = -99
        self.pmdec_max = -99
        self.par_min = -99
        self.par_max = -99
        self.cltpx = -99
        self.cltpy = -99
        self.noise_cutoff = -99
        
        #Check directory locations
        import os
        if not os.path.isdir(self.dataPath):
            os.mkdir(self.dataPath)
        if not os.path.isdir(self.imgPath):
            os.mkdir(self.imgPath)
        if not os.path.isdir(f"{self.imgPath}/png"):
            os.mkdir(f"{self.imgPath}/png")


#Gaia DR2 Implementation
# class starObj:
#     def __init__(self,name,ra,ra_err,dec,dec_err,par,par_err,par_over_err,pmra,pmra_err,pmdec,pmdec_err,ra_dec_corr,ra_par_corr,ra_pmra_corr,ra_pmdec_corr,dec_par_corr,dec_pmra_corr,dec_pmdec_corr,par_pmra_corr,par_pmdec_corr,pmra_pmdec_corr,astro_n_obs,astro_n_good_obs,astro_n_bad_obs,astro_gof,astro_chi2,astro_noise,astro_noise_sig,astro_match_obs,astro_sigma5d,match_obs,g_mag,b_mag,r_mag,b_r,b_g,g_r,radvel,radvel_err,variable,teff,a_g,e_bp_rp,lum):
#         #Declare instance variables
#         self.name = name
#         self.ra = float(ra)
#         self.ra_err = float(ra_err)
#         self.dec = float(dec)
#         self.dec_err = float(dec_err)
#         self.par = float(par)
#         self.par_err = float(par_err)
#         self.par_over_err = float(par_over_err)
#         self.pmra = float(pmra)
#         self.pmra_err = float(pmra_err)
#         self.pmdec = float(pmdec)
#         self.pmdec_err = float(pmdec_err)
#         self.ra_dec_corr = float(ra_dec_corr)
#         self.ra_par_corr = float(ra_par_corr)
#         self.ra_pmra_corr = float(ra_pmra_corr)
#         self.ra_pmdec_corr = float(ra_pmdec_corr)
#         self.dec_par_corr = float(dec_par_corr)
#         self.dec_pmra_corr = float(dec_pmra_corr)
#         self.dec_pmdec_corr = float(dec_pmdec_corr)
#         self.par_pmra_corr = float(par_pmra_corr)
#         self.par_pmdec_corr = float(par_pmdec_corr)
#         self.pmra_pmdec_corr = float(pmra_pmdec_corr)
#         self.astro_n_obs = float(astro_n_obs)
#         self.astro_n_good_obs = float(astro_n_good_obs)
#         self.astro_n_bad_obs = float(astro_n_bad_obs)
#         self.astro_gof = float(astro_gof)
#         self.astro_chi2 = float(astro_chi2)
#         self.astro_noise = float(astro_noise)
#         self.astro_noise_sig = float(astro_noise_sig)
#         self.astro_match_obs = float(astro_match_obs)
#         self.astro_sigma5d = float(astro_sigma5d)
#         self.match_obs = float(match_obs)
#         self.g_mag = float(g_mag)
#         self.b_mag = float(b_mag)
#         self.r_mag = float(r_mag)
#         self.b_r = float(b_r)
#         self.b_g = float(b_g)
#         self.g_r = float(g_r)
#         self.radvel = float(radvel)
#         self.radvel_err = float(radvel_err)
#         self.variable = variable
#         self.teff = float(teff)
#         self.a_g = float(a_g)
#         self.e_bp_rp = float(e_bp_rp)
#         self.lum = float(lum)
#         self.member = 0
#         self.binary = 0
#         self.radDist = 0
        
#         self.par_over_ra = float(par)/float(ra)
#         self.par_over_dec = float(par)/float(dec)
#         self.par_over_pmra = float(par)/float(pmra)
#         self.par_over_pmdec = float(par)/float(pmdec)
        
#         self.vosaPoints = []
#         self.excess = 0

#Gaia DR3 implementation
class starObj:
    def __init__(self,name,source_id,ra,ra_err,dec,dec_err,par,par_err,par_over_err,pmra,pmra_err,pmdec,pmdec_err, #Basic astrometrics
                 ra_dec_corr,ra_par_corr,ra_pmra_corr,ra_pmdec_corr,dec_par_corr,dec_pmra_corr,dec_pmdec_corr,par_pmra_corr,par_pmdec_corr,pmra_pmdec_corr, #Correlations
                 astro_n_obs,astro_n_good_obs,astro_n_bad_obs,astro_gof,astro_chi2,astro_noise,astro_noise_sig,astro_nu_eff, #Assorted astrometric properties
                 pseudocolor,pseudocolor_err,ra_pseudocolor_corr,dec_pseudocolor_corr,par_pseudocolor_corr,pmra_pseudoclor_corr,pmdec_pseudocolor_corr, #Pseudocolor
                 astro_sigma5d,duplicated_source, #More assorted properties
                 g_flux,g_flux_err,g_mag, #Gaia_G
                 b_flux,b_flux_err,b_mag, #Gaia_BP
                 r_flux,r_flux_err,r_mag, #Gaia_RP
                 b_over_r_excess,b_r,b_g,g_r, #Color indices and excess
                 radvel,radvel_err,radvel_num_transits,radvel_teff,radvel_feh, #Template Teff and Fe/H used to calculate the radvel
                 l,b,long,lat): #Galactic l and b, ecliptic long and lat
        import numpy as np
        #Declare instance variables
        self.name = name
        self.source_id = source_id
        self.ra = float(ra)
        self.ra_err = float(ra_err)
        self.dec = float(dec)
        self.dec_err = float(dec_err)
        self.par = float(par)
        self.par_err = float(par_err)
        self.par_over_err = float(par_over_err)
        self.pmra = float(pmra)
        self.pmra_err = float(pmra_err)
        self.pmdec = float(pmdec)
        self.pmdec_err = float(pmdec_err)
        
        self.ra_dec_corr = float(ra_dec_corr)
        self.ra_par_corr = float(ra_par_corr)
        self.ra_pmra_corr = float(ra_pmra_corr)
        self.ra_pmdec_corr = float(ra_pmdec_corr)
        self.dec_par_corr = float(dec_par_corr)
        self.dec_pmra_corr = float(dec_pmra_corr)
        self.dec_pmdec_corr = float(dec_pmdec_corr)
        self.par_pmra_corr = float(par_pmra_corr)
        self.par_pmdec_corr = float(par_pmdec_corr)
        self.pmra_pmdec_corr = float(pmra_pmdec_corr)
        
        self.astro_n_obs = float(astro_n_obs)
        self.astro_n_good_obs = float(astro_n_good_obs)
        self.astro_n_bad_obs = float(astro_n_bad_obs)
        self.astro_gof = float(astro_gof)
        self.astro_chi2 = float(astro_chi2)
        self.astro_noise = float(astro_noise)
        self.astro_noise_sig = float(astro_noise_sig)
        self.astro_nu_eff = float(astro_nu_eff)
        
        self.astro_sigma5d = float(astro_sigma5d)
        self.duplicated_source = bool(duplicated_source)
        
        self.g_flux = float(g_flux)
        self.g_flux_err = float(g_flux_err)
        self.g_mag = float(g_mag)
        
        self.b_flux = float(b_flux)
        self.b_flux_err = float(b_flux_err)
        self.b_mag = float(b_mag)
        
        self.r_flux = float(r_flux)
        self.r_flux_err = float(r_flux_err)
        self.r_mag = float(r_mag)
        
        self.b_over_r_excess = float(b_over_r_excess)
        self.b_r = float(b_r)
        self.b_g = float(b_g)
        self.g_r = float(g_r)
        
        self.radvel = float(radvel)
        self.radvel_err = float(radvel_err)
        self.radvel_num_transits=float(radvel_num_transits)
        self.radvel_teff = float(radvel_teff)
        self.radvel_feh = float(radvel_feh)
        
        self.l = float(l)
        self.b = float(b)
        self.long = float(long)
        self.lat = float(lat)
        
        self.member = 0
        self.binary = 0
        self.radDist = 0
        
        self.par_over_ra = float(par)/float(ra)
        self.par_over_dec = float(par)/float(dec)
        self.par_over_pmra = float(par)/float(pmra)
        self.par_over_pmdec = float(par)/float(pmdec)
        
        self.normRA = self.ra*np.cos(self.dec*np.pi/180)
        
        self.vosaPoints = []
        self.excess = 0



class isochroneObj:
    def __init__(self,age=404,feh=404,afe=404,y=404,basedir='isochrones/',subdir='processed',isodir=''):
        #Declare instance variables
        self.basedir = basedir
        self.subdir = subdir
        self.isodir = isodir
        self.starList = []
        self.age = age
        self.feh = feh
        self.afe = afe
        self.y = y
        self.name = f"feh_{feh}_afe_{afe}_age_{age}_y_{y}"
        self.distance = 0
        self.coeff = []
        self.g = []
        self.br = []


class fakeStarObj:
    def __init__(self,g_mag,b_mag,r_mag):
        #Declare instance variables
        self.g_mag = g_mag
        self.b_mag = b_mag
        self.r_mag = r_mag
        self.b_r = self.b_mag-self.r_mag
        self.b_g = self.b_mag-self.g_mag
        self.g_r = self.g_mag-self.r_mag
        self.score = 0

class mistStar:
    def __init__(self,properties):
        #Declare instance variables
        
        for prop,val in properties:
            if "inf" in str(val):
                val = 50
            exec(f"self.{prop} = {val}")


class condensedPoint:
    def __init__(self,b_r,g_mag,weight):
        self.b_r = b_r
        self.g_mag = g_mag
        self.weight = weight


class vosaPoint:
    def __init__(self,filterID,wavelength,obs_flux,obs_error,flux,flux_error,excess):
        self.filterID = filterID
        self.wavelength = wavelength
        self.obs_flux = obs_flux
        self.obs_error = obs_error
        self.flux = flux
        self.flux_error = flux_error
        self.excess = excess


class cataloguedCluster():
    def __init__(self,name,clType,pmra_min,pmra_max,pmdec_min,pmdec_max,par_min,par_max,cltpx,cltpy,noise_cutoff):
        #Catalogued properties
        self.name = str(name)
        self.clType = str(clType)
        self.pmra_min = float(pmra_min)
        self.pmra_max = float(pmra_max)
        self.pmdec_min = float(pmdec_min)
        self.pmdec_max = float(pmdec_max)
        self.par_min = float(par_min)
        self.par_max = float(par_max)
        self.cltpx = float(cltpx)
        self.cltpy = float(cltpy)
        self.noise_cutoff = float(noise_cutoff)


def clusterCatalogue(types='all'):
    import numpy as np
    import pandas as pd
    global data
    global catalogue
    global cataIn
    
    data = pd.read_csv("catalogue.csv",sep=',',dtype=str)
    data = data.to_numpy(dtype=str)
    cata = []
    for row in data:
        cata.append(cataloguedCluster(*row))
    
    if types == 'all':
        catalogue = cata
    
    cataIn = True
    return
    


def readClusters(cList=["M67"],basedir="clusters/",smRad=0.35):
    #Imports
    import numpy as np
    import pandas as pd
    global clusterList
    global clusters
    global stars
    global clIn
    global catalogue
    
    try:
        if clIn and len(clusterList) > 0:
            for clname in cList:
                if clname in clusters:
                    unloadClusters([clname])
    except:
        clusterList=[]
    
    #Check the cluster catalogue to load the catalogued properties
    if not cataIn:
        clusterCatalogue()
    
    #Loop through clusters
    for clname in cList:
        #Create cluster objects
        cluster = clusterObj(name=clname,basedir=basedir)
        
        reference = None
        
        for cl in catalogue:
            if str(cl.name) == str(clname):
                reference = cl
                print(f"Catalogue match for {clname} found")
                break
        if reference == None:
            print(f"Catalogue match for {clname} was not found, please create one")
            continue

        #Filter all of the methods out of the properties list
        properties = [a for a in dir(reference) if not a.startswith('_')]
        print(properties)
        #exec(f"print(reference.{properties[1]})")
        #print(properties)
        
        #Now we have a list of all the attributes assigned to the catalogue (the self.variables)
        for p in properties:
            prop =  getattr(reference,p)
            #print(prop)
            exec(f"cluster.{p} = prop")
            try:
                if prop <= -98:
                    print(f"{clname} does not have a specified catalogue value for {p}")
            except:
                continue
        

        # if cluster.name == 'NGC752' or cluster.name == 'NGC188':
        #     cluster.brightThreshold=18
        
        # if "M67" in clname:
        #     cluster.type = "open"
        # if "M35" in clname:
        #     cluster.type = "open"
        # if "NGC188" in clname:
        #     cluster.type = "open"
        # if "NGC752" in clname:
        #     cluster.type = "open"
        # if "IC4651" in clname:
        #     cluster.type = "open"
        # if "NGC2451" in clname:
        #     cluster.type = "open"
        # if "AlphaPer" in clname:
        #     cluster.type = "open"
        # if "M12" in clname:
        #     cluster.type = "globular"
        # if "M3" in clname:
        #     cluster.type = "globular"
        # if "M5" in clname:
        #     cluster.type = "globular"
        # if "M15" in clname:
        #     cluster.type = "globular"
        # if "M53" in clname:
        #     cluster.type = "globular"
        # if "NGC6426" in clname:
        #     cluster.type = "globular"
        # if "NGC6934" in clname:
        #     cluster.type = "globular"
        
        """
        #Generate wide-field star list
        starlist = np.genfromtxt(cluster.dataPath+"narrow.csv", delimiter=",", skip_header=1, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
        starlist = preFilter(starlist)
        for s in starlist:
            star = starObj(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11],s[12],s[13],s[14],s[15],s[16],s[17])
            cluster.unfilteredNarrow.append(star) 
        """
        
        #Generate narrow-field star list
        starlist = pd.read_csv(cluster.dataPath+"wide.csv",sep=',',dtype=str)
        stars = pd.read_csv(cluster.dataPath+"wide.csv",sep=',',dtype=str)
        starlist = starlist.to_numpy(dtype=str)
        #starlist = np.genfromtxt(cluster.dataPath+"wide.csv", delimiter=",", skip_header=1)
        print(f"{clname} initial length: {len(starlist)}")
        starlist = preFilter(starlist)
        print(f"{clname} post-prefiltered length: {len(starlist)}")
        
        ramean = np.mean([float(x) for x in starlist[:,1]])
        decmean = np.mean([float(x) for x in starlist[:,3]])
        
        
        for s in starlist:
            star = starObj(*s)
            cluster.unfilteredWide.append(star)
            
            if np.less_equal(star.g_mag,cluster.brightThreshold):
                cluster.unfilteredBright.append(star)
            
            # if np.less_equal(np.sqrt(((star.ra-ramean)*np.cos(np.pi/180*star.dec))**2+(star.dec-decmean)**2),smRad):
            #     cluster.unfilteredNarrow.append(star)
        clusterList.append(cluster)
        calcStats(cluster,mode='narrow')
    
    if not 'YSO' in clname:
        rmOutliers()
    clIn = True
    toDict()


def pad(string, pads):
    spl = string.split(',')
    return '\n'.join([','.join(spl[i:i+pads]) for i in range(0,len(spl),pads)])


def readIso(basedir='isochrones/',subdir='MIST_raw/'):
    #Important note: The ages are rounded to a few decimal places in the Gyr range
    #This has the effect of making it such that a few dozen isochrones in the kyr range 
    #are overwritten because they all round to the same value. I found this to be an issue
    #worth overlooking given that a cluster of that age hasn't been identified yet
    
    
    #Imports
    import os
    import re
    
    global isochrone_headers
    global isoList
    global isoIn
    
    path = basedir + subdir
    
    isoList = []
    
    for fn in os.listdir(path):
        
        #Read in file
        main = open(path+fn).read()
        main = main.split("\n")
        
        #Relevant variables from headers
        N_iso = int(main[7].split("=")[1])
        index = 13
        
        varList = re.sub("\s+", ",", main[5].strip()).split(",")
        afe = varList[4]
        feh = varList[3]
        y = varList[1]
        z = varList[2]
        v_vcrit = varList[5]
        
        #Column labels
        #Replace any number of spaces with a single comma, then replace a few problematic phrases and split the list by commas
        isochrone_headers = re.sub("\s+", ",", main[12].replace("2MASS","TwoMASS").replace("[Fe/H]","feh").strip()).split(",")[1:]
        
        for idx in range(0,N_iso):
            N_stars = int(re.sub("\s+", "," , main[index-3].split("=")[1]).split(",")[1])
            
            #print(f"Iso = {idx}   N_stars = {N_stars}")
            
            #Populate a single isochrone
            stars = []
            for i in range(index,index+N_stars):
                #Send the header and values to the mistStar object
                #print(f"i = {i}")
                values = [float(a) for a in re.sub("\s+", "," , main[i].strip()).split(",")]
                properties = zip(isochrone_headers,values)
                stars.append(mistStar(properties))
            #Create the isochrone from the list of stars
            age = round(10**values[1]/1e9,3)
            iso = isochroneObj(age,feh,afe,y)
            iso.starList = stars
            iso.br = [star.Gaia_BP_EDR3-star.Gaia_RP_EDR3 for star in stars]
            iso.g = [star.Gaia_G_EDR3 for star in stars]
            isoList.append(iso)
                
            index += N_stars + 5
        
    isoIn = True
    toDict()
    


def checkIsoDupes():
    global isochrones
    global isoList
    
    names = []
    for iso in isoList:
        if iso.name in names:
            print(iso.name)
        else:
            names.append(iso.name)


def processIso(basedir='isochrones/',subdir='raw/'):
    #Imports
    import os
    import re
    
    path = basedir + subdir
    
    for fn in os.listdir(path):
        main = open(path+fn).read()
        part = main.split('\n\n\n')
        part[0] = part[0].split('#----------------------------------------------------')[3].split('\n',1)[1]
         
        for a in range(len(part)):
            temp = part[a].split('#AGE=')[1].split(' EEPS=')[0]
            age = temp.strip()
            
            out = part[a].split('\n',2)[2]
            out = re.sub("\s+", ",", out.strip())
            out = pad(out,8)
            
            filename = f"{basedir}processed/"+fn.split('.')[0]+'/'+age+".csv"
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)    
            with open(filename,"w") as f:
                f.write(out)


def readIsochrones(basedir='isochrones/',subdir='processed/'):
    #Imports
    import os
    import numpy as np
    global isoList
    global isoIn
    
    isoList=[]
    
    for folder in os.listdir(basedir+subdir):
        for fn in os.listdir(basedir+subdir+folder):
            
            #Get the age and metallicities of the isochrones
            ageStr = fn.split('.csv')[0]
            fehStr = folder.split('feh')[1].split('afe')[0]
            afeStr = folder.split('afe')[1].split('y')[0]
            if 'y' in folder:
                yStr = folder.split('y')[1]
            else:
                yStr = '0'
            
            feh = float(fehStr[1]+fehStr[2])/10
            afe = float(afeStr[1])/10
            age = float(ageStr)
            y = int(yStr)
            
            if fehStr[0] == 'm':
                feh = feh*-1
            if afeStr[0] == 'm':
                afe = afe*-1
            
            #Debug
            #print(f"folder:{folder}   fn:{fn}   fehStr:{fehStr}   feh:{feh}   afeStr:{afeStr}   afe:{afe}   ageStr:{ageStr}   age:{age}")
            
            #Create isochone object
            iso = isochroneObj(age=age,feh=feh,afe=afe,y=y,basedir=basedir,subdir=subdir,isodir=folder+'/')
            
            isoArr = np.genfromtxt(basedir+subdir+folder+"/"+fn, delimiter=",")
            for s in isoArr:
                star = fakeStarObj(s[5],s[6],s[7])
                iso.starList.append(star)
                iso.br.append(s[6]-s[7])
                iso.g.append(s[5])
                
            isoList.append(iso)
    isoIn = True
    toDict()

def preFilter(starList):
    #Imports
    import numpy as np
    
    final = []
    #Columns to be checked for NaN values. If an NaN is present in this column, the entry(star) is discarded from the "unfiltered" list
    #2-12 is the astrometry
    #42,45,48 are the g,bp,rp magnitudes
    #50-52 are the color indices
    cols = list(range(2,13))+[42]+[45]+[48]+list(range(50,53))
    
    #Filters out NaN values except for the last two columns
    for n,s in enumerate(starList):
        dump = False
        for c in cols:
            if np.isnan(float(s[c])):
                dump = True
        if not dump:
            final.append(starList[n])
    
    #Reshapes array     
    final = np.array(final)
    
    return final

def rmOutliers():
    #Imports
    global clusterList
    import numpy as np
    
    for cluster in clusterList:
        
        if cluster.clType.lower() == "globular":
            scale = 4
        else:
            scale = 1.5
        
        #Variables
        pmthreshold = 5
        pmpthreshold = 50
        parthreshold = 5
        posthreshold = 5
        toRemove=[]
        
        #print(cluster.mean_pmra,cluster.mean_pmdec,cluster.stdev_pmra,cluster.stdev_pmdec)
        #print(len(cluster.unfilteredWide))
        
        #Classifies outliers
        for star in cluster.unfilteredWide:
            if cluster.name == "NGC188":
                if star.ra > 100:
                    toRemove.append(star)
            #print(np.sqrt(((star.pmra-cluster.mean_pmra)*np.cos(np.pi/180*star.pmdec))**2+(star.pmdec-cluster.mean_pmdec)**2),star.pmra,star.pmdec)
            if np.greater(np.sqrt(((star.pmra-cluster.mean_pmra)*np.cos(np.pi/180*star.pmdec))**2+(star.pmdec-cluster.mean_pmdec)**2),pmthreshold) or np.greater(np.sqrt(((star.ra-cluster.mean_ra)*np.cos(np.pi/180*star.dec))**2+(star.dec-cluster.mean_dec)**2),posthreshold) or np.greater(abs(star.par),parthreshold):
            #if np.greater(np.sqrt((star.pmra-cluster.mean_pmra)**2+(star.pmdec-cluster.mean_pmdec)**2),threshold):
                toRemove.append(star)
        
        #Removes the outliers from the array
        for rm in toRemove:
            cluster.unfilteredWide.remove(rm)
            try:
                cluster.unfilteredNarrow.remove(rm)
            except ValueError:
                pass
        
        #print(len(cluster.unfilteredWide))

def calcStats(cluster,mode='filtered'):
    #Imports
    import numpy as np
    
    #Reads in all the values for a cluster
    par=[]
    par_err=[]
    ra=[]
    dec=[]
    pmra=[]
    pmdec=[]
    gmag = []
    br = []
    # a_g=[]
    # e_bp_rp=[]
    
    loopList=[]
    
    if mode == 'bright':
        loopList = cluster.filteredBright
    elif mode == 'narrow':
        loopList = cluster.unfilteredNarrow
    elif mode == 'filtered':
        loopList = cluster.filtered
    
    for star in loopList:
        par.append(star.par)
        par_err.append(star.par_err)
        pmra.append(star.pmra)
        pmdec.append(star.pmdec)
        ra.append(star.ra)
        dec.append(star.dec)
        gmag.append(star.g_mag)
        br.append(star.b_r)
        
        # if not np.isnan(star.a_g) and not star.a_g == 0:
        #     a_g.append(star.a_g)
        # if not np.isnan(star.e_bp_rp) and not star.e_bp_rp == 0:
        #     e_bp_rp.append(star.e_bp_rp)
            
    #Calculate the statistics
    cluster.mean_par = np.mean(par[:])
    cluster.mean_ra = np.mean(ra[:])
    cluster.mean_dec = np.mean(dec[:])
    cluster.stdev_ra = np.std(ra[:])
    cluster.stdev_dec = np.std(dec[:])
    cluster.stdev_par = np.std(par[:])
    cluster.mean_pmra = np.mean(pmra[:])
    cluster.stdev_pmra = np.std(pmra[:])
    cluster.mean_pmdec = np.mean(pmdec[:])
    cluster.stdev_pmdec = np.std(pmdec[:])
    # cluster.mean_a_g = np.mean(a_g[:])
    # cluster.stdev_a_g = np.std(a_g[:])
    # cluster.mean_e_bp_rp = np.mean(e_bp_rp[:])
    # cluster.stdev_e_bp_rp = np.std(e_bp_rp[:])
    cluster.mean_par_over_ra = np.mean([x/y for x,y in zip(par,ra)])
    cluster.stdev_par_over_ra = np.std([x/y for x,y in zip(par,ra)])
    cluster.mean_par_err = np.mean(par_err[:])
    
    cluster.dist_mod = 5*np.log10(1000/cluster.mean_par)-5
    
    for star in loopList:
        star.radDist = np.sqrt((star.ra-cluster.mean_ra)**2+(star.dec-cluster.mean_dec)**2)


def saveClusters(cList):
    #Imports
    import dill
    
    saveResults(cList)
    #Creates a pickle file with all of the saved instances
    for cl in cList:
        cluster = clusters[cl]
        #print(cluster.name,id(cluster))
        with open(f"{cluster.dataPath}filtered.pk1", 'wb') as output:
            dill.dump(cluster, output)


def saveIsochrones():
    #Imports
    import dill
    global clusterList
    
    #Creates a pickle file with all of the saved instances
    for iso in isoList:
       with open(f"{iso.basedir}pickled/{iso.name}.pk1", 'wb') as output:
           dill.dump(iso, output)

    
def loadClusters(clusterNames=["M67"],basedir='clusters/'):
    #Imports
    import dill
    global clusterList
    global clusters
    global clIn
    
    for clusterName in clusterNames:
        if clusterName in clusters:
            unloadClusters([clusterName])
        #Reads in instances from the saved pickle file
        with open(f"{basedir}{clusterName}/data/filtered.pk1",'rb') as input:
            cluster = dill.load(input)
            clusterList.append(cluster)
    clIn = True
    toDict()


def loadIsochrones(basedir='isochrones/'):
    #Imports
    import dill
    import os
    global isoList
    global isoIn
    
    isoList=[]
    
    for fn in os.listdir(basedir+"pickled/"):
        #Reads in instances from the saved pickle file
        with open(f"{basedir}pickled/{fn}",'rb') as input:
            iso = dill.load(input)
            isoList.append(iso)
    isoIn = True
    toDict()


def unloadClusters(cList=['all']):
    #Imports
    global clusterList
    global clusters
    
    if 'all' in cList:
        cList = [cluster.name for cluster in clusterList]
    
    for cl in cList:
        cluster = clusters[cl]
        
        clusterList.remove(cluster)
        clusters.pop(cl)
        del cluster
        

def dataProcess(cList,load=False,fit=True,unload=True,plotting=True,member=True,save=True,close=True):
    #This method is largely intended for re-processing a bulk batch of clusters that have already been processed before,
    #meaning they already have condensed point lists or you are already aware of their fitting quality
    
    #Imports
    import matplotlib.pyplot as plt
    global clusterList
    global clusters
    global closePlots
    
    if not isoIn:
        loadIsochrones()
    
    
    loadList = ["M15","M39","M46","M67","NGC188","NGC2355","NGC2158","IC4651","NGC6791","NGC2360","NGC2204"]
    
    for cl in cList:
        
        if cl in loadList:
            condensing = "load"
        else:
            condensing = "auto"
        
        if load:
            loadClusters([cl])
        else:
            readClusters([cl])
            turboFilter([cl])
            
            if close:
                plt.close('all') 
                
        
        if fit:
            turboFit([cl],condensing=condensing)
        if plotting:
            plot([cl],['pos','pm','cmd','quiver','iso'])
            if close:
                plt.close('all') 
        
        if member:
            proxyMatch([cl])
            boundedStats([cl],saveCl=False,unloadCl=False)
            membership(cl,mode='filtered')
            membership(cl,mode='bounded',N=75)
            plt.close('all')
        
        if save:
            saveClusters([cl])
            saveResults([cl])
        if unload:
            unloadClusters([cl])
    



def turboFilter(cl=["all"]):
    #Imports
    global clusterList
    
    cList = checkLoaded(cl)
        
    for clus in cList:
        cluster = clusters[clus]
        
        cluster.filteredBright,cluster.brightmag = pmFilter(cluster.unfilteredBright,cluster.name)
        print(f"==========================={cluster.name}===========================")
        print(f"bright unf/pm fil: {len(cluster.unfilteredBright)}   /   {len(cluster.filteredBright)}")
        calcStats(cluster,mode='bright')
        distFilter(cluster)
        print(f"dist(all): {len(cluster.distFiltered)}")
        cluster.filtered,cluster.mag = pmFilter(cluster.distFiltered,cluster.name)
        
        
        #Manual filtering of extraneous points
        cluster.filtered,cluster.mag = manualFilter(cluster)
        
        
        print(f"pm(all): {len(cluster.filtered)}")
        
        customPlot('b_r','g_mag',cluster.name,'filtered',iso=True,square=False,color='astro_sigma5d')
        
        magnitude = cutNoise(cluster)
        print(f"noise cutoff: mag {magnitude}   length {len(cluster.filtered)}")
        
        customPlot('b_r','g_mag',cluster.name,'filtered',iso=True,square=False,color='astro_sigma5d')
        
        """
        for i in range(10):
            print(f"{cluster.filtered[i].b_r}   {cluster.mag[i,0]}")
        """
        
        calcStats(cluster,mode='filtered')
        setFlag()


def manualFilter(cluster):
    #This exists to remove any points that may or may not be relevant to the cluster but are prohibiting the fit from happening
    
    if "M35" in cluster.name:
        filtered = [star for star in cluster.filtered if star.g_mag > 9 or star.b_r < 1]
        return filtered,magList(filtered)
    else:
        return cluster.filtered,cluster.mag

def magList(filtered):
    import numpy as np
    
    mag = np.empty((0,2))
    
    for star in filtered:
        mag = np.r_[mag,[[star.b_r,star.g_mag]]]


def pmFilter(starList,name):
    #Imports
    import numpy as np
    
    filtered = []
    mag = np.empty((0,2))
    cluster = clusters[name]
    assert cluster.name == name
    
    #Apply an elliptical filter to the proper motion space 
    pmra_width = (cluster.pmra_max-cluster.pmra_min)/2
    pmdec_width = (cluster.pmdec_max-cluster.pmdec_min)/2
    pmra_center = cluster.pmra_min+pmra_width
    pmdec_center = cluster.pmdec_min+pmdec_width
    
    print(pmra_center,pmdec_center)
    
    for star in starList:
        if (star.pmra-pmra_center)**2/pmra_width**2 + (star.pmdec-pmdec_center)**2/pmdec_width**2 <= 1:
            filtered.append(star)
            mag = np.r_[mag,[[star.b_r,star.g_mag]]]
    
    assert len(filtered) > 1
    print(len(filtered))
    
    return filtered,mag


def distFilter(cluster):
    #Imports
    import numpy as np
    
    
    if cluster.par_min == 0 or cluster.par_max == 0:
        threshold = 1.5*cluster.mean_par
        
        print(f"{cluster.name} filtered using mean parallax")
        for star in cluster.unfilteredWide:
            if not np.greater(np.abs(star.par-cluster.mean_par),threshold*cluster.stdev_par):
                cluster.distFiltered.append(star)
    else:
        print(f"{cluster.name} filtered using min & max parallax values")
        for star in cluster.unfilteredWide:
            if star.par > cluster.par_min and star.par < cluster.par_max:
                cluster.distFiltered.append(star)



def cutNoise(cluster):
    #Imports
    import numpy as np
    
    stars = sorted(cluster.filtered,key=lambda x: x.g_mag)
    new = []
    newMag = np.empty((0,2))
    
    if cluster.noise_cutoff <= -98:
        threshold = 1
        print(f"{cluster.name} noise cutoff undefined, using default")
    else:
        threshold = cluster.noise_cutoff
    
    bad = 0
    badCut = 5
    for i,s in enumerate(stars):
        if s.astro_sigma5d > threshold:
            bad += 1
            if bad >= badCut:
                break
        else:
            new.append(s)
            newMag = np.r_[newMag,[[s.b_r,s.g_mag]]]
        
    cluster.filtered = new
    cluster.mag = newMag
    return s.g_mag


def turboFit(cl=["all"],condensing='auto',weighting='pos',tp="catalogue",minScore=0.001):
    #Typical use cases are auto, pos, catalogue --OR-- manual, equal, catalogue
    #Imports
    import time
    global clusterList
    
    cList = checkLoaded(cl)
    
    print("=========================Fitting=========================")
    t0 = time.time()
    
    status = condense(cList,condensing,weighting,tp,minScore)
    if status == "Suspended":
        return
    
    for cluster in cList:
        redFitting(cluster,minScore,weighting)
        
    
    t1 = time.time()
    
    print(f"Total {cluster.name} fit runtime: {t1-t0} seconds")
            


def redFitting(cluster,minScore,weighting):
    #Imports
    import numpy as np
    import math
    from sys import stdout
    from time import sleep
    global clusterList
    
    if type(cluster) == str:
        cluster = clusters[cluster]
    
    cluster.iso = []
        
    redMin = 0
    redMax = 0.7
    step = 0.05
    
    redList = [round(x,2) for x in np.arange(redMin,redMax+step,step)]
    
    for reddening in redList:
        stdout.write(f"\rCurrent reddening value for {cluster.name}: {reddening:.2f} / ({redList[0]:.2f}->{redList[-1]:.2f})")
        shapeFit(cluster,reddening,minScore,weighting)
        stdout.flush()
        sleep(0.1)
    
    cluster.iso = sorted(cluster.iso,key=lambda x: x[1])
    best = float(cluster.iso[0][2])
    
    print(f"\nCoarse-step reddening for {cluster.name}: {best}")
    
    subMin = best - 0.05
    subMax = best + 0.05
    substep = 0.01
    
    if subMin < 0:
        subMin = 0
    
    subList = [round(x,2) for x in np.arange(subMin,subMax+substep,substep) if not round(x,2) in redList and round(x,2) > subMin and round(x,2) < subMax]
    
    for reddening in subList:
        stdout.write(f"\rCurrent fine-step reddening value for {cluster.name}: {reddening:.2f} / ({subList[0]:.2f}->{subList[-1]:.2f})")
        shapeFit(cluster,reddening,minScore,weighting)
        stdout.flush()
        sleep(0.1)
    
    cluster.iso = sorted(cluster.iso,key=lambda x: x[1])
    
    cluster.reddening = float(cluster.iso[0][2])
    cluster.fit_age = float(isochrones[cluster.iso[0][0]].age)
    cluster.fit_feh = float(isochrones[cluster.iso[0][0]].feh)
    cluster.fit_afe = float(isochrones[cluster.iso[0][0]].afe)
    cluster.fit_y = float(isochrones[cluster.iso[0][0]].y)
    
    #Unrelated properties but I needed somewhere to assign them
    setattr(cluster,'meanDist',1000/cluster.mean_par)
    
    meanL = np.mean([a.l*np.pi/180 for a in cluster.filtered])
    galDist = 8000 #pc
    gd = cluster.meanDist**2 + galDist**2 - 2*cluster.meanDist*galDist*np.cos(meanL)
    setattr(cluster,'meanGalacticDist',gd**0.5)
    
    print(f"\nReddening for {cluster.name}: {best}")


def shapeFit(cluster,reddening,minScore,weighting):
    #Imports
    import numpy as np
    import shapely.geometry as geom
    global isoList
    
    
    conversion = 2.1
    
    isoFitList = np.empty((0,3))
    for iso in isoList:
        isoLine = geom.LineString(tuple(zip([x+reddening for x in iso.br],[x+cluster.dist_mod+conversion*reddening for x in iso.g])))
        dist = []
        for star in cluster.condensed:
            starPt = geom.Point(star.b_r,star.g_mag)
            #print(starPt.distance(isoLine))
            pointDist = np.abs(starPt.distance(isoLine))*star.weight
            if pointDist < minScore*star.weight:
                pointDist = minScore*star.weight
            dist.append(pointDist**2)
        isoScore = np.sum(dist[:])
        #print(isoScore,dist)
        #print(list(geom.shape(isoLine).coords))
        isoFitList = np.r_[isoFitList,[[iso.name,float(isoScore),float(reddening)]]]
        #compareInstances(iso,cluster.iso[-1][0])
        #print(isoScore)
    cluster.iso.extend(isoFitList)
    #best = cluster.iso[1][0]
    #specificPlot(cluster.name,best.name,reddening)
    #print(f"\nFirst point of best fit: {best.br[0]+reddening},{best.g[0]+conversion*reddening+cluster.dist_mod}")

    
def onclick(x,y,fig,ax,cluster,minScore,weighting,newList):
    def func(event):
        import matplotlib.pyplot as plt
        global coords
        
        ix, iy = event.xdata, event.ydata
        
        if str(event.button) == "MouseButton.RIGHT":
            for i,(cx,cy) in enumerate(coords):
                if abs(ix-cx) <= 0.075 and abs(iy-cy) <= 0.25:
                    coords.pop(i)
            ax.clear()
            ax.scatter(x,y,s=0.5,color='darkgray')
            ax.invert_yaxis()
            ax.scatter([a[0] for a in coords],[a[1] for a in coords],c='red',s=10)
            plt.gcf().canvas.draw_idle()
        
        if str(event.button) == "MouseButton.LEFT":
            coords.append((ix, iy))
            ax.scatter(ix,iy,c='red',s=10)
            plt.gcf().canvas.draw_idle()
        
        if str(event.button) == "MouseButton.MIDDLE":
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
            updateCondensed(cluster,minScore,weighting,newList)
    
        if len(coords) >= 100:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
            updateCondensed(cluster,minScore,weighting,newList)
            
    
        return
    return func


def updateCondensed(cluster,minScore,weighting,newList):
    #Imports
    import numpy as np
    global coords
    
    condensed = []
    for point in coords:
        if cluster.clType.lower() == "globular" or weighting.lower() == "equal":
                weight = 1
        else:
            #Automatic weighting scheme currently unsupported for manual condensed point definition,
            #but the framework is here to be able to insert it without having to worry about it being
            #passed around from function to function
            weight = 1
        condensed.append(condensedPoint(point[0],point[1],weight))
    
    if cluster.reddening == 0:
        cluster.condensed0 = condensed
    cluster.condensed = condensed
    
    np.savetxt(f"{cluster.dataPath}condensed.csv",coords,delimiter=',')
    
    redFitting(cluster,minScore,weighting)
    if len(newList) > 0:
        turboFit(newList,'manual',weighting,'catalogue',minScore)
    return


def find_nearest(array, value):
    #Imports
    import numpy as np
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def testCluster(name='feh_0.00_afe_0.00_age_0.141_y_0.2703'):
    #Imports
    import numpy as np
    global clusterList
    global clIn
    
    iso = isochrones[name]
    test = clusterObj('test')
    filtered = [starObj('fake',0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,a.Gaia_G_EDR3,0,0,0,0,0,0,0,a.Gaia_BP_EDR3-a.Gaia_RP_EDR3,0,0,0,0,0,0,0,0,0,0,0) for a in iso.starList]
    test.filtered = filtered
    
    mag = np.empty((0,2))
    for star in test.filtered:
        mag = np.r_[mag,[[star.b_r,star.g_mag]]]
    test.mag = mag
    
    if not 'test' in clusters:
        clusterList.append(test)
    else:
        idx = clusterList.index(clusters['test'])
        clusterList.pop(idx)
        clusterList.append(test)
    clIn = True
    toDict()

def condense(cList,condensing,weighting,tp,minScore=0.001):
    #Imports
    import numpy as np
    global isoList
    global mag
    
    
    for cluster in cList:
        
        if type(cluster) == str:
            cluster = clusters[cluster]
            cList[cList.index(cluster.name)] = cluster
            
        
        #Creates mag arrays to be used in place of the filtered star objects
        mag = cluster.mag[:,:]
        mag[mag[:,1].argsort()]
        gmag = list(mag[:,1])
        gmin = mag[0,1]
        gmax = mag[-1,1]
        div = 50
        seg = (gmax-gmin)/div
        minpoints = 1
        
        #The array that will become the condensed points list
        condensed = np.empty((0,3))
        turnPoints = []
        
        
        if condensing.lower() == "load":
            global pts
            pts = np.genfromtxt(f"{cluster.dataPath}condensed.csv",delimiter=',')
            condensed = []
            for point in pts:
                #Missing alternate weighting schemes, but can be imlemented *here*
                condensed.append(condensedPoint(point[0],point[1],1))
            cluster.condensed = condensed
            cluster.condensed0 = condensed
            continue
        
        #Manual point definition
        if condensing.lower() == "manual":
            import matplotlib.pyplot as plt
            global cid
            global coords
            coords = []
            
            if len(cList) == 1:
                newList = []
            else:
                newList = cList[cList.index(cluster)+1:]
            
            x,y = mag[:,0],mag[:,1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x,y,s=0.25,color='darkgray')
            ax.invert_yaxis()
            
            hook = onclick(x,y,fig,ax,cluster,minScore,weighting,newList)
            cid = fig.canvas.mpl_connect('button_press_event', hook) 
            
            return "Suspended"
            
        
        
        
        
        
        #Vertically stacked slices in brightness
        for i in range(div):
            sliced = mag[gmag.index(find_nearest(gmag,gmin+i*seg)):gmag.index(find_nearest(gmag,gmin+(i+1)*seg))]
            #print(np.array(sliced).shape)
            
            #Skip forseen problems with empty arrays
            if len(sliced) < minpoints:
                continue
            condensed = np.r_[condensed,[[np.median(sliced[:,0]),np.median(sliced[:,1]),0]]]
        
        condensed = condensed[::-1]
        
        
        
        #Uses defined turning points in the cluster catalogue
        if tp.lower() == "catalogue":
            if cluster.cltpx <= -98 and cluster.cltpy <= -98:
                tp == "auto"
            
        #If no turning point is found, or auto is specified, then this section of code
        #attempts to find the turning point through steep gradient changes in the main sequence
        if tp.lower() == "auto":
            #Criteria for the line that forms the basis of the gradient change method
            start = 4
            end = 11
            theta_crit = 5
            
            #Creates a slope-intercept fit for the lower main sequence
            basex = [a[0] for a in condensed[start:end]]
            basey = [a[1] for a in condensed[start:end]]
            base = np.polyfit(basex,basey,1)
            
            #Travels up the main sequence
            for i,point in enumerate(condensed):
                if i == start:
                    continue
                #Creates a fit line between the start point and the current point
                x = [point[0],condensed[start,0]]
                y = [point[1],condensed[start,1]]
                lin = np.polyfit(x,y,1)
                
                #Calculates an angle between the new line and the lower main sequence
                point[2] = 180/np.pi*np.arctan(abs( (base[0]-lin[0])/(1+base[0]*lin[0]) ))
                
                #If the angle between the two lines is large enough, the point is considered
                #to be a candidate turning point, and is appended to the list of candidates
                if point[2] > theta_crit and i > end:
                    turnPoints.append(point)
                
    
            #Analysis plot showing the theta value for each condensed point
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(condensed[:,0],condensed[:,1],c=condensed[:,2])
            plt.set_cmap('brg')
            plt.gca().invert_yaxis()
            clb = plt.colorbar()
            clb.ax.set_title("Theta")
            plt.savefig(f'condensed_{cluster.name}')
    
            #If no automatic turning point is found, ends the method here
            if len(turnPoints) == 0:
                print("No turning point identified for {cluster.name}")
                return
            else:
                #Identifies the proper turning point as a 5% color offset of the dimmest turning point candidate
                turnPoints = sorted(turnPoints,key=lambda x: x[1])
                tp = turnPoints[-1]
                tp[0] = tp[0] - 0.05*np.abs(tp[0])
                cluster.turnPoint = tp
        
        #Stores the condensed point list
        cl = []
        for point in condensed:
            cl.append(condensedPoint(point[0],point[1],point[2]))
        
        cluster.condensedInit = cl
        #                                     [ B-R , G , Theta ]
        print(f"{cluster.name} Turning Point: {cluster.turnPoint}")
        
        
        
        
        
        
        
        #Assuming the undefined catch for manual would be caught the first time around
        if tp.lower() == "catalogue":
            cluster.turnPoint = [cluster.cltpx,cluster.cltpy]
        
        if cluster.clType.lower() == "open":
            #Recalc with the turnPoint limit enforced - Ignore blue stragglers
            condensed = np.empty((0,3))
            condensed_giant = np.empty((0,3))
            yList = []
            
            #Vertically stacked slices in brightness
            for i in range(div):
                rawSliced = mag[gmag.index(find_nearest(gmag,gmin+i*seg)):gmag.index(find_nearest(gmag,gmin+(i+1)*seg))]
                
                sliced = np.empty((0,2))
                sliced_giant = np.empty((0,2))
                for point in rawSliced:
                    #print(point)
                    if point[0] >= cluster.turnPoint[0]:
                        sliced = np.r_[sliced,[[point[0],point[1]]]]
                    else:
                        sliced_giant = np.r_[sliced_giant,[[point[0],point[1]]]]
                
                #Skip forseen problems with empty arrays
                if len(sliced) > 0:
                    x = np.median(sliced[:,0])
                    y = np.median(sliced[:,1])
                    yList.append(y)
                    condensed = np.r_[condensed,[[x,y,1]]]
                if len(sliced_giant) > 3:
                    xg = np.median(sliced_giant[:,0])
                    yg = np.median(sliced_giant[:,1])
                    condensed_giant = np.r_[condensed_giant,[[xg,yg,1]]]
    
            
            #New turning point found from the reduced data set
            newTP = find_nearest(yList,cluster.turnPoint[1])
            
            index = 0
            
            for i,point in enumerate(condensed):
                if newTP == point[1]:
                    index = i
                    #print(f"{point} found to be TP")
                    break
            assert not index == 0
        
        
            #Binary star list
            tpcut = index + 3
            
            xset = condensed[tpcut:-1,0]
            yset = condensed[tpcut:-1,1]
            #print(cluster.name,yset)
            fit = np.polyfit(xset,yset,1)
            
            #Distance from the main sequence linear fit
            for star in cluster.filtered:            
                x0 = star.b_r
                y0 = star.g_mag
                dist = abs( y0 - fit[0]*x0 - fit[1] ) / np.sqrt(fit[0]**2 + 1)
                star.distance_MS = dist
                
                if dist > 0.05 and y0 < fit[0]*x0+fit[1] and x0 > xset[0] and y0 > condensed[index,1]:
                    cluster.binaries.append(star)
                    star.binary = 1
                else:
                    star.binary = 0
                
            
        
        
            #Fit weight parameters
            N = len(condensed)
            beta = -2
            
            index = index - 7
            
            for i,point in enumerate(condensed):
                #point[2] = 5/(1+np.abs(index-i))
                if weighting.lower() == 'pos':
                    point[2] = np.exp(beta*((i-index)/N)**2)
                
            
            # if cluster.type == "globular":
            #     condensed = np.vstack((condensed,condensed_giant))
        
        condensed = condensed[::-1]
        

        
        cl = []
        coords = []
        for point in condensed:
            cl.append(condensedPoint(point[0],point[1],point[2]))
            coords.append((point[0],point[1]))
        
        np.savetxt(f"{cluster.dataPath}condensed.csv",coords,delimiter=',')
        
        if cluster.reddening == 0:
            cluster.condensed0 = cl
        cluster.condensed = cl
        

# def checkLoaded(cList):
    
#     needsLoading = []
#     loaded = []
    
#     for cl in cList:
#         if not cl in clusters:
#             needsLoading.append(cl)
#         else:
#             loaded.append(cl)
    
#     return loaded,needsLoading()
        


def toDict():
    #Imports
    global clusterList
    global clusters
    global isoList
    global isochrones
    global resultList
    global results
    global clIn
    global isoIn
    global resultsIn
    
    if clIn:
        clName = []
        
        for cluster in clusterList:
            clName.append(cluster.name)
        clusters = dict(zip(clName,clusterList))
    
    if isoIn:
    
        isoName = []
        
        for iso in isoList:
            isoName.append(iso.name)
        isochrones = dict(zip(isoName,isoList))
    
    if resultsIn:
        resName=[]
        
        for res in resultList:
            resName.append(res.name)
        results = dict(zip(resName,resultList))


def plot(cList=['all'],modes=['pos','pm','cmd','quiver','iso'],closePlots=False):
    #Imports
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    import os
    global clusterList
    
    cList = checkLoaded(cList)
    
    for cl in cList:
        
        cluster = clusters[cl]
        
        if not os.path.isdir(f"{cluster.imgPath}/png"):
            os.mkdir(f"{cluster.imgPath}/png")
        
        #Position plots
        if 'pos' in modes:
            
            unfra=[star.ra for star in cluster.unfilteredWide]
            unfdec=[star.dec for star in cluster.unfilteredWide]
            ra=[star.ra for star in cluster.filtered]
            dec=[star.dec for star in cluster.filtered]
            
            unfnormra=[star.ra*np.cos(star.dec*np.pi/180) for star in cluster.unfilteredWide]
            normra=[star.ra*np.cos(star.dec*np.pi/180) for star in cluster.filtered]
            
            #Unfiltered position plot
            plt.figure(f"{cluster.name}_ra_dec_unfiltered")
            plt.xlabel('RA (Deg)')
            plt.ylabel('DEC (Deg)')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter(unfra[:],unfdec[:],s=0.5,c='darkgray')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_unfiltered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_ra_dec_unfiltered.png",dpi=500)
            
            #Filtered position plot
            plt.figure(f"{cluster.name}_ra_dec_filtered")
            plt.xlabel('RA (Deg)')
            plt.ylabel('DEC (Deg)')
            plt.title(f"{cluster.name} Filtered")
            plt.scatter(ra[:],dec[:],s=0.5,c='midnightblue')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_filtered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_ra_dec_filtered.png",dpi=500)
            
            #Position overlay
            plt.figure(f"{cluster.name}_ra_dec_overlay")
            plt.xlabel('RA (Deg)')
            plt.ylabel('DEC (Deg)')
            plt.title(f"{cluster.name} Overlay")
            plt.scatter(unfra[:],unfdec[:],s=0.5,c='darkgray')
            plt.scatter(ra[:],dec[:],s=1,c='midnightblue')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_overlay.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_ra_dec_overlay.png",dpi=500)
            
            
            #Normalized
            #NormRA = RA*cos(DEC)
            
            #Unfiltered normalized position plot
            plt.figure(f"{cluster.name}_ra_dec_unfiltered_normalized")
            plt.xlabel('RA*cos(DEC) (Deg)')
            plt.ylabel('DEC (Deg)')
            plt.title(f"{cluster.name} Unfiltered Normalized")
            plt.scatter(unfnormra[:],unfdec[:],s=0.5,c='darkgray')
            #plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_unfiltered_normalized.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_ra_dec_unfiltered_normalized.png",dpi=500)
            
            #Filtered normalized position plot
            plt.figure(f"{cluster.name}_ra_dec_filtered_normalized")
            plt.xlabel('RA*cos(DEC) (Deg)')
            plt.ylabel('DEC (Deg)')
            plt.title(f"{cluster.name} Filtered Normalized")
            plt.scatter(normra[:],dec[:],s=0.5,c='midnightblue')
            #plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_filtered_normalized.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_ra_dec_filtered_normalized.png",dpi=500)
            
            #Position overlay normalized
            plt.figure(f"{cluster.name}_ra_dec_overlay_normalized")
            plt.xlabel('RA*cos(DEC) (Deg)')
            plt.ylabel('DEC (Deg)')
            plt.title(f"{cluster.name} Overlay Normalized")
            plt.scatter(unfnormra[:],unfdec[:],s=0.5,c='darkgray')
            plt.scatter(normra[:],dec[:],s=1,c='midnightblue')
            #plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_overlay_normalized.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_ra_dec_overlay_normalized.png",dpi=500)
            
        #Proper motion plots
        if 'pm' in modes:
            
            unfpmra=[star.pmra for star in cluster.unfilteredWide]
            unfpmdec=[star.pmdec for star in cluster.unfilteredWide]
            pmra=[star.pmra for star in cluster.filtered]
            pmdec=[star.pmdec for star in cluster.filtered]
            
            unfpara=[star.par for star in cluster.unfilteredWide]
            para=[star.par for star in cluster.filtered]
            
            x0 = cluster.pmra_min
            x1 = cluster.pmra_max
            y0 = cluster.pmdec_min
            y1 = cluster.pmdec_max
            width = x1-x0
            scale = 5
            subscale = 2
            xmin = x0-scale*width
            xmax = x1+scale*width
            ymin = y0-scale*width
            ymax = y1+scale*width
            sxmin = x0-subscale*width
            sxmax = x1+subscale*width
            symin = y0-subscale*width
            symax = y1+subscale*width
            
            
            #Unfiltered proper motion plot
            plt.figure(f"{cluster.name}_pm_unfiltered")
            plt.xlabel(r'PMRA (mas*yr^{-1})')
            plt.ylabel(r'PMDEC (mas*yr^{-1})')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter(unfpmra[:],unfpmdec[:],s=0.5,c='darkgray')
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            # plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_unfiltered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_pm_unfiltered.png",dpi=500)
            plt.xlim([sxmin,sxmax])
            plt.ylim([symin,symax])
            # plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_unfiltered_closeup.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_pm_unfiltered_closeup.png",dpi=500)
            
            #Filtered proper motion plot
            plt.figure(f"{cluster.name}_pm_filtered")
            plt.xlabel(r'PMRA (mas*yr^{-1})')
            plt.ylabel(r'PMDEC (mas*yr^{-1})')
            plt.title(f"{cluster.name} Filtered")
            plt.scatter(pmra[:],pmdec[:],s=0.5,c='midnightblue')
            # plt.xlim([xmin,xmax])
            # plt.ylim([ymin,ymax])
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_filtered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_pm_filtered.png",dpi=500)
            
            #Proper motion overlay
            plt.figure(f"{cluster.name}_pm_overlay")
            plt.xlabel(r'PMRA (mas*yr^{-1})')
            plt.ylabel(r'PMDEC (mas*yr^{-1})')
            plt.title(f"{cluster.name} Overlay")
            plt.scatter(unfpmra[:],unfpmdec[:],s=0.5,c='darkgray')
            plt.scatter(pmra[:],pmdec[:],s=1,c='midnightblue')
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            # plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_overlay.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_pm_overlay.png",dpi=500)
            plt.xlim([sxmin,sxmax])
            plt.ylim([symin,symax])
            # plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_overlay_closeup.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_pm_overlay_closeup.png",dpi=500)
            
            #Unfiltered PM/Parallax
            plt.figure(f"{cluster.name}_pm_over_parallax_unfiltered")
            plt.xlabel('PMRA / Parallax')
            plt.ylabel('PMDEC / Parallax')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter([a/b for a,b in zip(unfpmra,unfpara)],[a/b for a,b in zip(unfpmdec,unfpara)],s=0.5,c='darkgray')
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            # plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_over_parallax_unfiltered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_pm_over_parallax_unfiltered.png",dpi=500)
            
            #Unfiltered PM*Parallax
            plt.figure(f"{cluster.name}_pm_times_parallax_unfiltered")
            plt.xlabel('PMRA * Parallax')
            plt.ylabel('PMDEC * Parallax')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter([a*b for a,b in zip(unfpmra,unfpara)],[a*b for a,b in zip(unfpmdec,unfpara)],s=0.5,c='darkgray')
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            # plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_times_parallax_unfiltered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_pm_times_parallax_unfiltered.png",dpi=500)
        
        
        #CMD plots
        if 'cmd' in modes:
            
            unfgmag=[star.g_mag for star in cluster.unfilteredWide]
            unf_b_r=[star.b_r for star in cluster.unfilteredWide]
            gmag=[star.g_mag for star in cluster.filtered]
            b_r=[star.b_r for star in cluster.filtered]
            
            bright_b_r = [x.b_r for x in cluster.filteredBright]
            bright_gmag = [x.g_mag for x in cluster.filteredBright]
            par_b_r = [x.b_r for x in cluster.distFiltered]
            par_gmag = [x.g_mag for x in cluster.distFiltered]
        
            #Reddening Correction
            plt.figure(f"{cluster.name}_reddening_CMD")
            plt.gca().invert_yaxis()
            plt.xlabel('BP-RP')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Reddening = {cluster.reddening:.2f}")
            plt.scatter(b_r[:],gmag[:],s=0.5,c='darkgray',label='Observed')
            plt.arrow(b_r[int(len(b_r)/2)]-cluster.reddening,gmag[int(len(gmag)/2)]-2.1*cluster.reddening,cluster.reddening,2.1*cluster.reddening,color='red')
            plt.scatter([s-cluster.reddening for s in b_r[:]],[s-2.1*cluster.reddening for s in gmag[:]],s=1,c='midnightblue',label='Corrected')
            plt.legend()
            plt.savefig(f"{cluster.imgPath}{cluster.name}_reddening_CMD.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_reddening_CMD.png",dpi=500)
            
            #Unfiltered CMD plot
            plt.figure(f"{cluster.name}_CMD_unfiltered")
            plt.gca().invert_yaxis()
            plt.xlabel('BP-RP')
            plt.ylabel('Apparent G Mag')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter(unf_b_r[:],unfgmag[:],s=0.5,c='darkgray')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_unfiltered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_CMD_unfiltered.png",dpi=500)
            
            #Filtered CMD plot
            plt.figure(f"{cluster.name}_CMD_filtered")
            plt.gca().invert_yaxis()
            plt.xlabel('BP-RP')
            plt.ylabel('Apparent G Mag')
            plt.title(f"{cluster.name} Parallax & Proper Motion Filtered")
            plt.scatter(b_r[:],gmag[:],s=0.5,c='midnightblue')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_filtered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_CMD_filtered.png",dpi=500)
            
            #CMD overlay
            plt.figure(f"{cluster.name}_CMD_overlay")
            plt.gca().invert_yaxis()
            plt.xlabel('BP-RP')
            plt.ylabel('Apparent G Mag')
            plt.title(f"{cluster.name} Overlay")
            plt.scatter(unf_b_r[:],unfgmag[:],s=0.5,c='darkgray')
            plt.scatter(b_r[:],gmag[:],s=1,c='midnightblue')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_overlay.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_CMD_overlay.png",dpi=500)
            
            #Condensed CMD overlay
            plt.figure(f"{cluster.name}_condensed_CMD_overlay")
            plt.gca().invert_yaxis()
            plt.xlabel('BP-RP')
            plt.ylabel('Apparent G Mag')
            plt.title(f"{cluster.name} Condensed Overlay")
            plt.scatter([s - cluster.reddening for s in b_r],[s - 2.1*cluster.reddening for s in gmag],s=0.5,c='darkgray',label='Data')
            plt.scatter([s.b_r - cluster.reddening for s in cluster.condensed],[s.g_mag - 2.1*cluster.reddening for s in cluster.condensed],s=5,c='midnightblue',label='Proxy Points')
            try:
                plt.axvline(x=cluster.turnPoint[0] - cluster.reddening,linestyle='--',color='midnightblue',linewidth=0.8,label='95% of Turning Point')
            except:
                print(f"No turning point found for {cluster.name}")
            plt.legend()
            plt.savefig(f"{cluster.imgPath}{cluster.name}_condensed_CMD_overlay.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_condensed_CMD_overlay.png",dpi=500)
            
            #Weighted CMD overlay
            plt.figure(f"{cluster.name}_weighted_CMD_overlay")
            plt.gca().invert_yaxis()
            plt.xlabel('BP-RP')
            plt.ylabel('Apparent G Mag')
            plt.title(f"{cluster.name} Weighted Overlay")
            plt.scatter([s - cluster.reddening for s in b_r],[s - 2.1*cluster.reddening for s in gmag],s=0.5,c='darkgray',label='Data')
            plt.scatter([s.b_r - cluster.reddening for s in cluster.condensed],[s.g_mag - 2.1*cluster.reddening for s in cluster.condensed],s=5,c=[s.weight for s in cluster.condensed],label='Proxy Points')
            try:
                plt.axvline(x=cluster.turnPoint[0] - cluster.reddening,linestyle='--',color='midnightblue',linewidth=0.8,label='95% of Turning Point')
            except:
                print(f"No turning point found for {cluster.name}")
            plt.set_cmap('brg')
            clb = plt.colorbar()
            clb.ax.set_title("Weight")
            plt.legend()
            plt.savefig(f"{cluster.imgPath}{cluster.name}_weighted_CMD_overlay.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_weighted_CMD_overlay.png",dpi=500)
            
            
            #Initial Condensed CMD overlay
            plt.figure(f"{cluster.name}_initial_condensed_CMD_overlay")
            plt.gca().invert_yaxis()
            plt.xlabel('BP-RP')
            plt.ylabel('Apparent G Mag')
            plt.title(f"{cluster.name} Initial Condensed Overlay")
            plt.scatter(b_r,gmag,s=0.5,c='darkgray',label='Data')
            plt.scatter([s.b_r for s in cluster.condensedInit],[s.g_mag for s in cluster.condensedInit],s=5,c='midnightblue',label='Proxy Points')
            try:
                plt.axvline(x=cluster.turnPoint[0] - cluster.reddening,linestyle='--',color='midnightblue',linewidth=0.8,label='95% of Turning Point')
            except:
                print(f"No turning point found for {cluster.name}")
            plt.legend()
            plt.savefig(f"{cluster.imgPath}{cluster.name}_initial_condensed_CMD_overlay.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_initial_condensed_CMD_overlay.png",dpi=500)
            
            #Brightness-PM Filtered CMD plot
            plt.figure(f"{cluster.name}_CMD_bright_filtered")
            plt.gca().invert_yaxis()
            plt.xlabel('BP-RP')
            plt.ylabel('Apparent G Mag')
            plt.title(f"{cluster.name} Bright-Only Proper Motion Filtered")
            plt.scatter(bright_b_r[:],bright_gmag[:],s=0.5,c='midnightblue')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_bright_filtered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_CMD_bright_filtered.png",dpi=500)
            
           #Parallax Filtered CMD plot
            plt.figure(f"{cluster.name}_CMD_parallax_filtered")
            plt.gca().invert_yaxis()
            plt.xlabel('BP-RP')
            plt.ylabel('Apparent G Mag')
            plt.title(f"{cluster.name} Parallax Filtered")
            plt.scatter(par_b_r[:],par_gmag[:],s=0.5,c='midnightblue')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_parallax_filtered.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_CMD_parallax_filtered.png",dpi=500)
            
            
        if 'quiver' in modes:
            
            unfra=[star.ra for star in cluster.unfilteredWide]
            unfdec=[star.dec for star in cluster.unfilteredWide]
            unfpmra=[star.pmra for star in cluster.unfilteredWide]
            unfpmdec=[star.pmdec for star in cluster.unfilteredWide]
            
            x0 = min([s.ra for s in cluster.filtered])
            x1 = max([s.ra for s in cluster.filtered])
            y0 = min([s.dec for s in cluster.filtered])
            y1 = max([s.dec for s in cluster.filtered])
            width = x1-x0
            scale = 0.25
            xmin = x0+scale*width
            xmax = x1-scale*width
            ymin = y0+scale*width
            ymax = y1-scale*width
            
            #Unfiltered position quiver plot
            plt.figure(f"{cluster.name}_ra_dec_unfiltered_quiver")
            plt.xlabel('RA (Deg)')
            plt.ylabel('DEC (Deg)')
            plt.title(f"{cluster.name} Unfiltered")
            ax = plt.gca()
            ax.quiver(unfra[:],unfdec[:],unfpmra[:],unfpmdec[:],color='midnightblue',width=0.003,scale=400,scale_units='width')
            plt.axis("square")
            plt.gcf().set_size_inches(10,10)
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_unfiltered_pm_quiver.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_ra_dec_unfiltered_pm_quiver.png",dpi=500)
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_unfiltered_pm_quiver_zoom.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_ra_dec_unfiltered_pm_quiver_zoom.png",dpi=500)
            
            
        #Isochrone plots
        if 'iso' in modes:
            
            gmag=[star.g_mag for star in cluster.filtered]
            b_r=[star.b_r for star in cluster.filtered]
            isochrone = isochrones[cluster.iso[0][0]]
            
            #Isochrone best fit
            plt.figure(f"{cluster.name}_Iso_best")
            plt.gca().invert_yaxis()
            plt.xlabel('Dereddened BP-RP')
            plt.ylabel('Corrected Absolute G Mag')
            plt.title(f"{cluster.name} Isochrone Best Fit")
            plt.scatter([s - cluster.reddening for s in b_r],[s - 2.1*cluster.reddening-cluster.dist_mod for s in gmag],s=0.5,c='darkgray',label='Cluster')
            
            isoLabels = isochrone.name.split('_')
            isoLabel = r"$[\frac{Fe}{H}]$" + "=" + isoLabels[1] + "\n" \
            + r"$[\frac{\alpha}{Fe}]$" + "=" + isoLabels[3] + "\n" \
            + r"$[Y]$" + "=" + isoLabels[7] + "\n" \
            + "Age" + "=" + isoLabels[5] + " Gyr"
            
            plt.plot(isochrone.br,isochrone.g,c='midnightblue',label=isoLabel)
            plt.scatter([s.b_r - cluster.reddening for s in cluster.condensed],[s.g_mag - 2.1*cluster.reddening-cluster.dist_mod for s in cluster.condensed],s=5,c='red',label='Cluster Proxy')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            h,l = plt.gca().get_legend_handles_labels()
            h.insert(0,extra)
            l.insert(0,f"Reddening: {cluster.reddening}")
            plt.legend(h,l)
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_Iso_BestFit.pdf")
            plt.savefig(f"{cluster.imgPath}png/{cluster.name}_CMD_Iso_BestFit.png",dpi=500)
        
        #Membership plots
        if 'membership' in modes:
            proxyMatch([cl])
            boundedStats([cl],saveCl=False,unloadCl=False)
            membership(cl,mode='filtered')
            membership(cl,mode='bounded',N=50)
        
        #3D Position plots
        if '3D' in modes:
            
            A = [a.ra * np.pi/180 for a in cluster.filtered]
            B = [abs(b.dec) * np.pi/180 for b in cluster.filtered]
            C = [1/(1000*c.par) for c in cluster.filtered]
            
            x = [c*np.cos(b)*np.cos(a) for a,b,c in zip(A,B,C)]
            y = [c*np.cos(b)*np.sin(a) for a,b,c in zip(A,B,C)]
            z = [c*np.sin(b) for b,c in zip(B,C)]
            
            r = [np.sqrt(a**2+b**2) for a,b in zip(x,y)]
            theta = [np.arctan(b/a) for a,b in zip(x,y)]
            
            plt.figure(f"{cluster.name}_3D_Position")
            ax = plt.axes(projection='3d')
            ax.scatter3D(x,y,z)
            ax.scatter(0,0,0,color='red')
            scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
        
        if closePlots:
            plt.close('all')



# def Plot3D(cList):
#     #Imports
#     import matplotlib.pyplot as plt
#     import numpy as np
#     global clusterList
    
#     needsLoading=[]
    
#     plt.figure(f"3D_Position_Ensemble")
#     ax = plt.axes(projection='3d')
    
    
#     for cl in cList:
#         if not cl in clusters:
#             needsLoading.append(cl)
    
#     if not len(needsLoading) == 0:
#         loadClusters(needsLoading)
    
#     for cl in cList:
#         cluster = clusters[cl]
    
#         A = [a.ra * np.pi/180 for a in cluster.filtered]
#         B = [abs(b.dec) * np.pi/180 for b in cluster.filtered]
#         C = [1/(0.001*c.par) for c in cluster.filtered]
        
#         #Flatten radially
#         C = [np.mean(C)]*len(C)
        
#         x = [c*np.cos(b)*np.cos(a) for a,b,c in zip(A,B,C)]
#         y = [c*np.cos(b)*np.sin(a) for a,b,c in zip(A,B,C)]
#         z = [c*np.sin(b) for b,c in zip(B,C)]
        
#         #Force Cluster to origin
#         # x = [a-np.mean(x) for a in x]
#         # y = [a-np.mean(y) for a in y]
#         # z = [a-np.mean(z) for a in z]
        
#         ax.scatter3D(x,y,z,label=cluster.name)
        
#     scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
#     ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
#     #ax.scatter(0,0,0,color='black')
#     plt.legend()


def yso_lookup():
    #Imports
    from astroquery.simbad import Simbad
    import numpy as np
    import os
    import re
    
    global names
    global sect
    global results
    global ra
    global dec
    
    main = open("Excess Examples/YSO_object_list.dat").read()
    main = main.split("\n")[:-1]
    
    #Get the names of all of the objects identified
    names = []
    ra = []
    dec = []
    validNames = []
    for row in main:
        sect = re.split('\s+',row)
        if sect[0] == '':
            sect = sect[1:]
        if sect[2] == 'none':
            continue
        
        name = sect[2]
        
        blacklist = ['A','Ab','AB','ABC','B','AaB']
        for entry in sect[3:]:
            if '.' in entry or entry in blacklist:
                break
            name = name + " " + entry
            
        names.append(name)
    
    #Perform a SIMBAD query for the identified objects
    results = []
    for name in names:
        result = Simbad.query_object(name)
        if not type(result) == type(None):
            results.append(result)
            validNames.append(name.replace(' ',''))
            
            ra1 = str(result.columns['RA']).split('\n')[-1]
            ra1 = re.split('\s+',ra1)
            
            if '' in ra1:
                ra.append('---')
            else:
                ra.append(str(round(float(ra1[0])*15+float(ra1[1])/4+float(ra1[2])/240,5)))
            
            dec1 = str(result.columns['DEC']).split('\n')[-1]
            dec1 = re.split('\s+',dec1)
            if '' in dec1:
                dec.append('---')
            else:
                dec.append(str(round(float(dec1[0])+float(dec1[1])/60+float(dec1[2])/3600,5)))
    
    #Create a text file in the VOSA readable format
    VOSAdata = []
    gaiadata = []
    for i in range(len(validNames)):
        line1 = f"{validNames[i]} {ra[i]} {dec[i]} --- --- --- --- --- --- ---"
        line2 =  f"{ra[i]} {dec[i]}"
        VOSAdata.append(line1)
        if '-' in line2:
            continue
        gaiadata.append(line2)
    np.savetxt("Excess Examples/yso_vosa_output.txt",VOSAdata,fmt="%s")
    np.savetxt("Excess Examples/yso_gaia_output.txt",gaiadata,fmt="%s")
    


def exportVOSA(cl):
    #Imports
    import numpy as np
    
    if not cl in clusters:
        loadClusters([cl])
    
    cluster = clusters[cl]
    
    #objname  RA   DEC     DIS Av  Filter          Flux               Error             PntOpts ObjOpts
    data = []
    for star in cluster.filtered:
        name = star.name.replace(" ","")
        line = f"{name} {star.ra} {star.dec} {1000/star.par} --- --- --- --- --- ---"
        data.append(line)
    np.savetxt(f"{cluster.dataPath}{cluster.name}_VOSA.txt",data,fmt="%s")


def readSED(cList=['all']):
    #imports
    import numpy as np
    import re
    import os
    
    cList = checkLoaded(cList)
    
    for cl in cList:

        cluster = clusters[cl]
        
        objPath = cluster.dataPath + "vosa_results/objects/"
        
        names = []
        for star in cluster.filtered:
            flat = star.name.replace(" ","")
            names.append(flat)
            star.flatName = flat
        cluster.stars = dict(zip(names,cluster.filtered))
        
        idx = 0
        
        #Each star in a cluster has its own folder, and each folder contains several data sets
        for folder in os.listdir(objPath):
            main = open(objPath+folder+"/sed/"+folder+".sed.dat").read()
            main = main.split("\n")
            data = main[10:-1]
            
            #Create a list of measurement object pointers to attach to the stars later
            measurements = []
            
            #Convert every line of the data set into a vosaPoint object
            for row in data:
                sect = re.split('\s+',row)[1:-1]
                measurements.append(vosaPoint(str(sect[0]),float(sect[1]),float(sect[2]),float(sect[3]),float(sect[4]),float(sect[5]),float(sect[6])))
            
            cluster.stars[folder].vosaPoints = measurements
            idx += 1
            

def excessIR(cl,plot=True):
    #Imports
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    checkLoaded([cl])
    
    cluster = clusters[cl]
    
    if not os.path.isdir(f"{cluster.imgPath}excessIR/"):
        os.mkdir(f"{cluster.imgPath}excessIR/")
    
    
    for star in cluster.filtered:
        
        excess = False
        
        for vp in star.vosaPoints:
            
            if vp.excess > 0:
                excess = True
        
        if excess:
            
            #print(f"{star.name} has {len(star.vosaPoints)} VOSA points")
            
            star.excess = 1
            
            if plot:
                plt.figure(f'{cluster.name} - {star.name}')
                plt.title(f'{cluster.name} : {star.name}')
                
                ax = plt.gca()
                ax.set_yscale('log')
                ax.set_xscale('log')
                plt.ylabel(r'Flux ($ergs^{-1}cm^{-2}\AA^{-1}$)')
                plt.xlabel(r'Wavelength ($\AA$)')
                
                plt.scatter([a.wavelength for a in star.vosaPoints],[a.flux for a in star.vosaPoints])
                
                plt.savefig(f"{cluster.imgPath}excessIR/{star.name}.pdf")
                plt.savefig(f"{cluster.imgPath}excessIR/{star.name}.png",dpi=500)




def proxyMatch(cList,plot=False):
    #Imports
    import matplotlib.pyplot as plt
    import numpy as np
    
    checkLoaded(cList)   
    
    for cl in cList:
        cluster = clusters[cl]
        
        iso = isochrones[cluster.iso[0][0]]
        isoPoints = []
        for pt in iso.starList:
            isoPoints.append(pt)
            # if pt.Gaia_G_EDR3+cluster.dist_mod > cluster.turnPoint[1]:
            #     isoPoints.append(pt)
        
        for star in cluster.filtered:
            minDist = 0.2
            smallestDist = 10
            vertCutoff = 1
            minPoint = None
            for point in isoPoints:
                dist = abs(point.Gaia_BP_EDR3-point.Gaia_RP_EDR3-star.b_r+cluster.reddening)
                if dist < minDist:
                    if abs(point.Gaia_G_EDR3+cluster.dist_mod - star.g_mag + 2.1*cluster.reddening) < vertCutoff:
                        minDist = dist
                        minPoint = point
                    elif dist < smallestDist:
                        smallestDist = dist
            try:
                assert minDist < 0.2
            except:
                print(f"[{cluster.name}] Star too distant from isochrone to make a good proxy: BP-RP: {star.b_r} | G: {star.g_mag} | Dist: {smallestDist}")
                star.proxyMass = 0
                star.proxyLogTemp = 0
                star.proxyFeH = 0
                star.proxyLogAge = 0
                star.proxy = None
                continue
            
            #print(minDist)
            star.proxyMass = minPoint.star_mass
            star.proxyLogTemp = minPoint.log_Teff
            star.proxyFeH = minPoint.feh
            star.proxyLogAge = minPoint.log10_isochrone_age_yr
            star.proxy = minPoint
        
        cluster.massLoaded = True
        cluster.meanProxyMass = np.mean([a.proxyMass for a in cluster.filtered])
        cluster.totalProxyMass = np.sum([a.proxyMass for a in cluster.filtered])
        
        cluster.min_g_mag = min([a.g_mag for a in cluster.filtered])
        cluster.max_g_mag = max([a.g_mag for a in cluster.filtered])
        cluster.min_b_r = min([a.b_r for a in cluster.filtered])
        cluster.max_b_r = max([a.b_r for a in cluster.filtered])
        # if plot:
        #     plt.figure(f"{cluster.name}_proxy_fit")
            
    



def variableHistogram(cl,var):
    #Imports
    import numpy as np
    import matplotlib.pyplot as plt
    
    checkLoaded([cl])
    
    cluster = clusters[cl]
    
    plt.figure()
    plt.title(f"{cluster.name} Histogram of {var}")
    plt.xlabel(f"{var}")
    plt.ylabel("Count")
    plt.hist([eval(f"a.{var}") for a in cluster.filtered],bins='auto')


def varHist2D(cl,var1,var2,color='default',listType='filtered'):
    #Imports
    import numpy as np
    import matplotlib.pyplot as plt
    
    checkLoaded([cl])
    
    
    #Check allowed entries
    allowedTypes = ['filtered','unfilteredWide','unfilteredBright,filteredBright,binaries']
    if not listType in allowedTypes:
        print(f"{listType} is not a valid list type, defaulting to filtered")
        listType = "filtered"
    
    
    cluster = clusters[cl]
    
    plt.figure(figsize=(8,8))
    
    #Axis size and spacing
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    
    x = [eval(f"a.{var1}") for a in eval(f"cluster.{listType}")]
    y = [eval(f"a.{var2}") for a in eval(f"cluster.{listType}")]
    
    if color == 'default':
        ax_scatter.scatter(x, y, s=5)
    else:
        colorMap = plt.get_cmap('coolwarm')#.reversed()
        ax_scatter.scatter(x, y, s=5, c=[eval(f"a.{color}") for a in eval(f"cluster.{listType}")], cmap = colorMap)
        # clb = plt.colorbar(ax_scatter)
        # clb.ax.set_title(f"{color}")
    
    ax_histx.hist(x,bins='auto')
    ax_histy.hist(y,bins='auto',orientation='horizontal')
    
    ax_histx.set_title(f"Histogram of {listType} {cluster.name} in {var1} and {var2}")
    ax_scatter.set_xlabel(f"{var1}")
    ax_scatter.set_ylabel(f"{var2}")
    





def Plot3D(cList=['all'],showEarth=True):
    #Imports
    import plotly.express as px
    import plotly.io as pio
    import numpy as np
    global clusterList
    
    pio.renderers.default='browser'
    
    fig = px.scatter_3d()
    
    if showEarth:
        fig.add_scatter3d(x=[0],y=[0],z=[0],marker=dict(color='black'),name="Earth")
    
    cList = checkLoaded([cList])
    
    big = []
    
    for cl in cList:
        cluster = clusters[cl]
    
        A = [a.ra * np.pi/180 for a in cluster.filtered]
        B = [abs(b.dec) * np.pi/180 for b in cluster.filtered]
        C = [1/(0.001*c.par) for c in cluster.filtered]
        
        #Flatten radially
        C = [np.mean(C)]*len(C)
        
        x = [c*np.cos(b)*np.cos(a) for a,b,c in zip(A,B,C)]
        y = [c*np.cos(b)*np.sin(a) for a,b,c in zip(A,B,C)]
        z = [c*np.sin(b) for b,c in zip(B,C)]
        
        #Force Cluster to origin
        # x = [a-np.mean(x) for a in x]
        # y = [a-np.mean(y) for a in y]
        # z = [a-np.mean(z) for a in z]
        
        fig.add_scatter3d(x=x,y=y,z=z,name=cl,mode="markers",marker=dict(size=2))
        
        big.append(np.amax(x))
        big.append(np.amax(y))
        big.append(np.amax(z))
        

    #fig.layout.scene = dict(aspectmode="manual",aspectratio=dict(x=1,y=1,z=1))
    #fig.update_layout(scene=dict(aspectmode="cube",xaxis=dict(showbackground=False,range=[-1*np.amax(big),np.amax(big)]),yaxis=dict(showbackground=False,range=[-1*np.amax(big),np.amax(big)]),zaxis=dict(showbackground=False,range=[-1*np.amax(big),np.amax(big)])))
    fig.update_layout(scene=dict(aspectmode="cube",xaxis=dict(showbackground=False),yaxis=dict(showbackground=False),zaxis=dict(showbackground=False,visible=False)))
    
    fig.show()


def specificPlot(cl,iso,reddening,score):
    #Imports
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import os
    
    checkLoaded([cl])
    
    cluster = clusters[f"{cl}"]
    isochrone = isochrones[f"{iso}"]
    
    #These are displayed on the plot
    # score = 0
    reddening = float(reddening)
    
    #Directory for saving plot outputs
    if not os.path.isdir("SpecificPlots/pdf/"):
        os.makedirs("SpecificPlots/pdf/")
    if not os.path.isdir("SpecificPlots/png/"):
        os.makedirs("SpecificPlots/png/")
    
    # #Find the score of the associated isochrone
    # for chrone in cluster.iso:
    #     if chrone[0] == iso and chrone[2] == reddening:
    #         score = chrone[1]
    #         break
    
    #Plots the CMD and the isochrone, with all of the points adjusted to reddening, extinction, and distance modulus
    plt.figure()
    plt.gca().invert_yaxis()
    plt.xlabel('B-R')
    plt.ylabel('G Mag')
    plt.title(f"{cl} {iso}")
    plt.scatter([s.b_r for s in cluster.filtered],[s.g_mag for s in cluster.filtered],s=0.05,c='darkgray',label='Cluster')
    plt.plot([x + reddening for x in isochrone.br],[x+cluster.dist_mod+2.1*reddening for x in isochrone.g],c='midnightblue',label=f"Score: {float(score):.7f}")
    plt.scatter([s.b_r for s in cluster.condensed],[s.g_mag for s in cluster.condensed],s=5,c=[s.weight for s in cluster.condensed],label='Cluster Proxy')
    
    #Colors the points by their fitting weight
    plt.set_cmap('brg')
    clb = plt.colorbar()
    clb.ax.set_title("Weight")
    
    #Label for the reddening
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    h,l = plt.gca().get_legend_handles_labels()
    h.insert(0,extra)
    l.insert(0,f"Reddening: {reddening}")
    plt.legend(h,l)
    
    #Save figure output to disk
    plt.savefig(f"SpecificPlots/pdf/Requested_Plot_{cl}_{iso}_Reddening_{reddening}.pdf")
    plt.savefig(f"SpecificPlots/png/Requested_Plot_{cl}_{iso}_Reddening_{reddening}.png",dpi=500)


def plotRange(cl,a,b):
    global clusters
    
    checkLoaded([cl])
    
    #Plots the top fitting isochrones over the range a to b for a given cluster
    #Does this by calling the specificPlot() method for each isochrone over the range
    for isochrone in clusters[f"{cl}"].iso[a:b]:
        specificPlot(cl,isochrones[isochrone[0]].name,isochrone[2],isochrone[1])

def getIsoScore(cl,iso,red,output=True):
    #Return the score for a given cluster's isochrone fit
    for i in cl.iso:
        if i[0] == iso.name and float(i[2]) == red:
            return i[1]
    if output:
        print(f"No score found for {cl.name} | {iso.name} | {red}")
    return 0


def onkey(x,y,cx,cy,fig,ax,cluster,iso,reddening):
    global curIso
    global curReddening
    curIso = iso
    curReddening = reddening
    
    def func(event):
        import matplotlib.patches as patches
        global curIso
        global curReddening
        global isochrones
        
        key = str(event.key)
        #print(key)
        
        ageSorted = [a for a in sorted(isoList,key=lambda x: float(x.age)) if a.feh == curIso.feh]
        fehSorted = [a for a in sorted(isoList,key=lambda x: float(x.feh)) if a.age == curIso.age]
        
        age_index = ageSorted.index(curIso)
        feh_index = fehSorted.index(curIso)
        
        #Move up or down in the desired variable space, with wrap-around at the ends of the lists
        if key == "w":
            #Increase metallicity
            try:
                curIso = fehSorted[feh_index+1]
                feh_index = feh_index+1
            except:
                curIso = fehSorted[0]
                feh_index = 0
        if key == "s":
            #Decrease metallicity
            curIso = fehSorted[feh_index-1]
            feh_index = feh_index-1
            if feh_index < 0:
                feh_index = len(fehSorted)+feh_index
        if key == "a":
            #Increase age
            curIso = ageSorted[age_index-1]
            age_index = age_index-1
            if age_index < 0:
                age_index = len(ageSorted)+age_index
        if key == "d":
            #Decrease age
            try:
                curIso = ageSorted[age_index+1]
                age_index = age_index+1
            except:
                curIso = ageSorted[0]
                age_index = 0
        if key == "q":
            #Decrease metallicity
            curReddening = round(curReddening-0.01,2)
        if key == "e":
            #Increase metalicity
            curReddening = round(curReddening+0.01,2)
        if key == "r":
            #Reset to originally requested isochrone
            curIso = iso
            ageSorted = [a for a in sorted(isoList,key=lambda x: float(x.age)) if a.feh == curIso.feh]
            fehSorted = [a for a in sorted(isoList,key=lambda x: float(x.feh)) if a.age == curIso.age]
            age_index = ageSorted.index(curIso)
            feh_index = fehSorted.index(curIso)
        if key == " ":
            #Print currently highlighted isochrone to console
            score = getIsoScore(cluster,curIso,curReddening)
            fig.savefig(f"Jamboree Images/frames/{curIso.name}.png",dpi=500)
            print(f"{curIso.name} | {curReddening} | {score}")
        
        score = getIsoScore(cluster,curIso,curReddening,output=False)
        
        #Replots everything with the updated isochrone
        ax.clear()
        ax.scatter(x,y,s=0.25,color='darkgray')
        ax.scatter(cx,cy,s=4,color='red')
        ax.plot([a.Gaia_BP_EDR3-a.Gaia_RP_EDR3+curReddening for a in curIso.starList],[a.Gaia_G_EDR3+cluster.dist_mod+2.1*curReddening for a in curIso.starList],color='darkblue')
        ax.set_title(f"{curIso.name}\n {curReddening}\n {score}")
        ax.set_xlabel("Apparent BP-RP")
        ax.set_ylabel("Apparent G Mag")
        ax.invert_yaxis()
        
        
        #Progress bar indicators for the interactive plot
        
        #Sets the dimensons of the boxes
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        margin = 0.01
        width = 0.05     * (x1-x0)
        height = 0.6     * (y1-y0)
        xmargin = margin * (x1-x0)
        ymargin = margin * (y1-y0)
        
        
        #The two main progress bars
        rect1 = patches.Rectangle((x1-width-xmargin,y0+ymargin),width,height,linewidth=1,edgecolor='black',facecolor='none',alpha=0.5)
        rect2 = patches.Rectangle((x1-2*width-2*xmargin,y0+ymargin),width,height,linewidth=1,edgecolor='black',facecolor='none',alpha=0.5)
        #rect3 = patches.Rectangle((x1-3*width-3*xmargin,y0+ymargin),width,height,linewidth=1,edgecolor='black',facecolor='none',alpha=0.5)
        
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        #ax.add_patch(rect3)
        
        #The segments filling up the progress bars
        n = len(ageSorted)
        #Adds cells bottom to top
        for i in range(n):
            offset = i*height/n
            alpha = 0.25
            if i == age_index:
                color = 'red'
            else:
                color = 'black'
            #Age progress bar
            ax.add_patch(patches.Rectangle((x1-2*width-2*xmargin,y0+ymargin+offset),width,height/n,linewidth=0.01,edgecolor='black',facecolor=color,alpha=alpha))
        n = len(fehSorted)
        for i in range(n):
            offset = i*height/n
            alpha = 0.25
            if i == feh_index:
                color = 'red'
            else:
                color = 'black'
            #Metallicity progress bar
            ax.add_patch(patches.Rectangle((x1-1*width-1*xmargin,y0+ymargin+offset),width,height/n,linewidth=0.01,edgecolor='black',facecolor=color,alpha=alpha))
        
        fig.canvas.draw_idle()
        
            
    return func

def interactivePlot(cl,iso=0,reddening="auto"):
    #Imports
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    global clusters
    global isochrones
    global kid
    
    checkLoaded([cl])
    
    cluster = clusters[f"{cl}"]
    
    #Select the starting isochrone based on user input
    if type(iso) == str:
        isochrone = isochrones[f"{iso}"]
    elif type(iso) == int:
        assert iso >= 0
        isochrone = isochrones[cluster.iso[iso][0]]
    else:
        print("Invalid declaration of 'iso'")
        return
    name = isochrone.name
    
    #Get the reddening if not manually defined
    if reddening == "auto":
        reddening = cluster.reddening
    assert type(reddening) == float or type(reddening) == int
    
    score = getIsoScore(cluster,isochrone,reddening)
    
    # #Sorted and secondary-sorted isochrone lists
    # ageSorted = sorted(isoList,key=lambda x: (x.age,x.feh))
    # fehSorted = sorted(isoList,key=lambda x: (x.feh,x.age))
    ageSorted = [a for a in sorted(isoList,key=lambda x: float(x.age)) if a.feh == isochrone.feh]
    fehSorted = [a for a in sorted(isoList,key=lambda x: float(x.feh)) if a.age == isochrone.age]
    age_index = ageSorted.index(isochrone)
    feh_index = fehSorted.index(isochrone)
    
    
    #Coordinate lists to plot in addition to the isochrones
    x,y = cluster.mag[:,0],cluster.mag[:,1]
    cx,cy = [s.b_r for s in cluster.condensed],[s.g_mag for s in cluster.condensed]
    
    
    #Systematically remove some of the conflicting default keymaps in Pyplot
    letters = ['w','s','a','d','q','e','r']
    for letter in letters:
        #Finds all keymap references in the rcParams
        for param in [key for key in plt.rcParams if key.startswith("keymap") ]:
            try:
                plt.rcParams[param].remove(letter)
            except:
                continue
    
    
    #Initialize the plot that will be updated every time
    fig = plt.figure(f"Interactive plot of {cl}")
    ax = fig.add_subplot(111)
    ax.scatter(x,y,s=0.25,color='darkgray')
    ax.scatter(cx,cy,s=4,color='red')
    ax.plot([a.Gaia_BP_EDR3-a.Gaia_RP_EDR3+reddening for a in isochrone.starList],[a.Gaia_G_EDR3+cluster.dist_mod+2.1*reddening for a in isochrone.starList],color='darkblue')
    ax.set_title(f"{name}\n {reddening}\n {score}")
    ax.set_xlabel("Apparent BP-RP")
    ax.set_ylabel("Apparent G Mag")
    ax.invert_yaxis()
    
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    margin = 0.01
    width = 0.05     * (x1-x0)
    height = 0.6     * (y1-y0)
    xmargin = margin * (x1-x0)
    ymargin = margin * (y1-y0)
    
    
    rect1 = patches.Rectangle((x1-width-xmargin,y0+ymargin),width,height,linewidth=1,edgecolor='black',facecolor='none',alpha=0.5)
    rect2 = patches.Rectangle((x1-2*width-2*xmargin,y0+ymargin),width,height,linewidth=1,edgecolor='black',facecolor='none',alpha=0.5)
    #rect3 = patches.Rectangle((x1-3*width-3*xmargin,y0+ymargin),width,height,linewidth=1,edgecolor='black',facecolor='none',alpha=0.5)
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    #ax.add_patch(rect3)
    
    n = len(ageSorted)
    #Adds cells bottom to top
    for i in range(n):
        offset = i*height/n
        alpha = 0.25
        if i == age_index:
            color = 'red'
        else:
            color = 'black'
        ax.add_patch(patches.Rectangle((x1-2*width-2*xmargin,y0+ymargin+offset),width,height/n,linewidth=0.01,edgecolor='black',facecolor=color,alpha=alpha))
    n = len(fehSorted)
    for i in range(n):
        offset = i*height/n
        alpha = 0.25
        if i == feh_index:
            color = 'red'
        else:
            color = 'black'
        ax.add_patch(patches.Rectangle((x1-1*width-1*xmargin,y0+ymargin+offset),width,height/n,linewidth=0.01,edgecolor='black',facecolor=color,alpha=alpha))
    
    #Launch the key_press listener
    hook = onkey(x,y,cx,cy,fig,ax,cluster,isochrone,reddening)
    kid = fig.canvas.mpl_connect('key_press_event',hook)


def printList(cList,varList):
    
    cList = checkLoaded(cList)
        
    for cl in cList:
        cluster = clusters[cl]
        for a in varList:
            clStr = f"[{cl}] {a} ="
            exec(f"print(clStr,cluster.{a})")

def statRange(cl,a,b):
    import numpy as np
    global clusters
    
    checkLoaded([cl])
    if not isoIn:
        loadIsochrones()
    
    ages = []
    fehs = []
    ys = []
    reds = []
    
    #Computes the mean age, metallicity, and reddening for the top fitting isochrones over the range a to b for a given cluster
    #For example, a=0, b=10 will average the top 10 isochrone fits
    for isochrone in clusters[cl].iso[a:b]:
        iso = isochrones[isochrone[0]]
        print(f"{iso.name}   Reddening:{isochrone[2]}")
        ages.append(float(iso.age))
        fehs.append(float(iso.feh))
        ys.append(float(iso.y))
        reds.append(float(isochrone[2]))
        
        
    print(f"[{cl}] Mean age= {np.mean(ages)}   Mean feh= {np.mean(fehs)}   Mean y= {np.mean(ys)}   Mean Reddening= {np.mean(reds)}")
        

        
def setFlag():
    #Imports
    global clusterlist
    
    #Goes back and sets membership flags for all of the clusters loaded in memory to ensure that this tag can be used later
    #This takes place automatically after running turboFilter()
    #Example use case for this variable is in the customPlot() method
    for cluster in clusterList:
        for star in cluster.filtered:
            for unfStar in cluster.unfilteredWide:
                if star == unfStar:
                    unfStar.member = 1
        
def customPlot(var1,var2,clname,mode='filtered',iso=False,square=True,color='default',title='default',close=False,save=True):
    #Imports
    import matplotlib.pyplot as plt
    global closePlots
    
    #Load the cluster if it isn't yet
    checkLoaded([clname])
    cluster = clusters[f"{clname}"]
    
    
    #Set the list of stars to be used for the given cluster
    #Using a mode not specified will return a referenced before assignment error
    if mode == 'filtered':
        starlist = cluster.filtered
    elif mode == 'unfiltered':
        starlist = cluster.unfilteredWide
    elif mode == 'bright_filtered':
        starlist = cluster.filteredBright
    elif mode == 'dist_filtered':
        starlist = cluster.distFiltered
    elif mode == 'bright_unfiltered':
        starlist = cluster.unfilteredBright
    elif mode == 'duo':
        starlist = cluster.unfilteredWide 
        starlistF = cluster.filtered
    elif mode == 'binary':
        starlist = cluster.binaries
    elif mode == 'duoBinary':
        starlist = cluster.filtered
        starlistF = cluster.binaries
    elif mode == 'duoBright':
        starlist = cluster.unfilteredBright
        starlistF = cluster.filteredBright
    elif mode == 'duoDist':
        starlist = cluster.distFiltered
        starlistF = cluster.filtered
    elif mode == 'condensed':
        starlist = cluster.condensed
    elif mode == 'duoCondensed':
        starlist = cluster.filtered
        starlistF = cluster.condensed
    elif mode == 'bounded':
        starlist = cluster.bounded
    elif mode == 'duoBounded':
        starlist = cluster.filtered
        starlistF = cluster.bounded
    else:
        print("No preset star list configuration found with that alias")
        return
    
    #Basic plot features with axis labels and a title
    plt.figure()
    if title == 'default':
        plt.title(f"{clname} {mode} | {var1} vs {var2} | {color} color")
    else:
        plt.title(f"{title}")
    plt.xlabel(f"{var1}".upper())
    plt.ylabel(f"{var2}".upper())
    
    #Plots differently depending on the mode
    #The color tag can be used to add distinction of a third variable while limited to two axes
    #If unspecified, filtered starlist with midnight blue coloring will be the result
    if iso:
        plt.gca().invert_yaxis()
    if 'duo' in mode:
         #plt.scatter([eval(f"x.{var1}") for x in starlist],[eval(f"y.{var2}") for y in starlist],s=[0.1+a.member*1.4 for a in starlist],c=[list(('lightgray',eval('z.par')))[z.member] for z in starlist])
         plt.scatter([eval(f"x.{var1}") for x in starlist],[eval(f"y.{var2}") for y in starlist],s=2,c='gray')
         if color == 'default':    
              plt.scatter([eval(f"x.{var1}") for x in starlistF],[eval(f"y.{var2}") for y in starlistF],s=2.5,c='red')
         else:
            plt.scatter([eval(f"x.{var1}") for x in starlistF],[eval(f"y.{var2}") for y in starlistF],s=2.5,c=[eval(f"z.{color}") for z in starlistF])
            plt.set_cmap('brg')
            clb = plt.colorbar()
            clb.ax.set_title(f"{color}")
    else:
        if color == 'default':    
            plt.scatter([eval(f"x.{var1}") for x in starlist],[eval(f"y.{var2}") for y in starlist],s=1,c='midnightblue')
        else:
            plt.scatter([eval(f"x.{var1}") for x in starlist],[eval(f"y.{var2}") for y in starlist],s=2,c=[eval(f"z.{color}") for z in starlist])
            plt.set_cmap('cool')
            clb = plt.colorbar()
            clb.ax.set_title(f"{color}")
    
    #By default, squares the axes to avoid misinformation from stretched axes
    #Turn this off and iso to true for a color magnitude diagram
    if square:
        plt.axis("square")
    
    if save:
        plt.savefig(f"SpecificPlots/pdf/{clname}_{mode}_{var1}_{var2}.pdf")
        plt.savefig(f"SpecificPlots/png/{clname}_{mode}_{var1}_{var2}.png",dpi=500)
    
    if close or closePlots:
        plt.close()
        if save:
            print(f"Custom Plot {clname}_{mode}_{var1}_{var2} saved and closed")
        else:
            print(f"Custom Plot {clname}_{mode}_{var1}_{var2} closed")

def splitMS(clname='M67',slope=3,offset=12.2):
    #Imports
    import numpy as np
    import matplotlib.pyplot as plt
    
    checkLoaded([clname])
    cluster = clusters[clname]
    
    xlist = [s.b_r for s in cluster.filtered]
    ylist = [s.g_mag for s in cluster.filtered]
    
    x = np.linspace(1,2,100)
    
    #Create a diagram showing the lower edge and upper edge of the main sequence, which in theory are separated by 0.75mag
    plt.figure()
    plt.title('Main and Binary Sequences')
    plt.xlabel('B-R')
    plt.ylabel('Apparent G Mag')
    plt.scatter(xlist,ylist,s=0.5,label='Filtered Star Data')
    plt.plot(x,[slope*a + offset for a in x],color='r',label='Main Sequence')
    plt.plot(x,[slope*a + offset - 0.75 for a in x],'--',color='r',label='MS shifted 0.75 mag')
    plt.xlim(0.6,2.2)
    plt.ylim(13,19)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig(f"SpecificPlots/png/{clname}_MS_Spread.png",dpi=500)
    plt.savefig(f"SpecificPlots/pdf/{clname}_MS_Spread.pdf")


def kingProfile(r,K,R):
    
    return K*(1+r**2/R**2)**(-1)

def kingError(r,K,R,dK,dR):
    import numpy as np
    
    dfdK = (1+r**2/R**2)**(-1)
    dfdR = 2*K*r**2*R*(r**2+R**2)**(-2)
    return np.sqrt((dfdK*dK)**2 + (dfdR*dR)**2)

def densityProfile(r,K,R):
    import numpy as np
    
    #The exponential that is fit for the membership profile
    #R is a characteristic radius, typically negative but the absolute value is used for comparison
    #K is a scalar constant
    return K*np.exp(-1*r/R)

def densityError(r,K,R,dK,dR):
    import numpy as np
    
    dfdK = abs(np.exp(-1*r/R))
    dfdR = abs(K*r/(R**2)*np.exp(-1*r/R))
    return np.sqrt((dfdK*dK)**2 + (dfdR*dR)**2)
    

def toIntensity(mag):
    msun = -26.74 #apparent magnitude
    Isun = 1360 #w/m^)
    
    return Isun*10**( 0.4*(msun-mag) )


def membership(clname='M67',N=100,mode='filtered',numPercentileBins=5,percentile=0.2,delta=5):
    #Imports
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import scipy.optimize as so
    import scipy.stats as st
    import math
    
    global volume
    
    checkLoaded([clname])
    cluster = clusters[clname]
    
    mode = mode.lower()
    
    #Default mode is filtered, but unfiltered data can be processed
    if mode == "filtered":
        starList = cluster.filtered
    elif mode == "bounded":
        starList = cluster.bounded
    else:
        starList = cluster.unfilteredWide
    
    #Load mass estimates from isochrone fitting
    if not cluster.massLoaded:
        proxyMatch([cluster.name])
    assert cluster.massLoaded
    assert len(starList) > 0
    
    #Determine bounds of the field of view (post-filtering)
    xmax = sorted(starList,key=lambda x : x.ra)[-1].ra
    ymax = sorted(starList,key=lambda x : x.dec)[-1].dec
    x0 = cluster.mean_ra
    y0 = cluster.mean_dec
    newN = N
    
    #Determine radius of the field of view
    rx = xmax-x0
    ry = ymax-y0
    #r = np.mean([rx,ry])
    radiusFOV = ry
    #Using the mean ra and dec radius caused problems with clusters
    #like NGC188, which are close to the celestial pole and have
    #a very stretched mapping to the RA DEC space
    
    ringBins = list(np.linspace(0,radiusFOV,N))
    
    #The bins are divided up such that 50% of the bins are located in the inner 25% of the cluster radius
    #The remaining 50% of the bins are divided from 25% to 100% of the radius
    rings = list(np.linspace(0,radiusFOV/4,math.ceil(N/2)))
    ring2 = list(np.linspace(radiusFOV/4,radiusFOV,math.floor(N/2)+1))
    ring2 = ring2[1:-1]
    rings.extend(ring2)
    
    x=rings[:-1]
    # for i in range(0,len(rings[:-1])):
    #     x.append((rings[i+1]+rings[i])/2)
    counts = list(np.zeros(N-1,dtype=int))
    masses = list(np.zeros(N-1,dtype=int))
    
    rads=[]
    for star in starList:
        #Radial distance from the mean RA and Dec of the cluster
        rads.append(np.sqrt((star.ra-x0)**2+(star.dec-y0)**2))
        #Find the nearest ring to the star
        r = find_nearest(rings, rads[-1])
        i = rings.index(r)
        #Check bounds
        if i < len(counts):
            #If outside last ring, add to that count
            if r > rads[-1]:
                counts[i-1] += 1
                masses [i-1] += star.proxyMass
            else:
                counts[i] += 1
                masses [i] += star.proxyMass
    #Worth noting here that the way that this is set up, the rings don't actually mark the bounds of the bins but rather the midpoints.
    #There is no check to see if you are exterior or interior to the nearest ring, but rather what ring you are nearest to,
    #so the rings mark the midpoints of their bins not the boundaries
    
    
    #Histogram of the counts in each radial bin
    plt.figure(f"{clname}_membership_{mode}")
    plt.hist(rads,bins=ringBins)
    plt.xlabel("Radius (deg)")
    plt.ylabel("Number of Stars")
    plt.title(f"{clname} Membership")
    plt.savefig(f"{cluster.imgPath}{clname}_membership_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_membership_{mode}.png",dpi=500)

    #Calculates the volume of each region bounded by two concentric rings and the number density of the stars counted in those regions
    volume = []
    for i in range(0,len(rings[:-1])):
        volume.append(np.pi*(rings[i+1]**2-rings[i]**2))
    numDensity = [a/b for a,b in zip(counts,volume)]
    massDensity = [a/b for a,b in zip(masses,volume)]
    error_num = [np.sqrt(a)/b for a,b in zip(counts,volume)]
    error_mass = [np.sqrt(a)/b for a,b in zip(masses,volume)]
    
    for i in range(0,len(error_num)):
        if error_num[i] < 0.1:
            error_num[i] = 0.1

    #Cut out the inner 5% because overbinning in the center of a circle doesn't help
    x = x[math.ceil(N/20):-1]
    counts = counts[math.ceil(N/20):-1]
    numDensity = numDensity[math.ceil(N/20):-1]
    massDensity = massDensity[math.ceil(N/20):-1]
    error_num = error_num[math.ceil(N/20):-1]
    error_mass = error_mass[math.ceil(N/20):-1]

    #Further filter the data based on outliers, either extremely low density or extremely big jumps in density from bin to bin
    i = 0
    numSmall = 0
    numGrad = 0
    while i < len(x)-1:
        if numDensity[i] < 0.5 or numDensity[i] < numDensity[i+1]/delta or massDensity[i] < 0.1:
            x.pop(i)
            counts.pop(i)
            numDensity.pop(i)
            massDensity.pop(i)
            error_num.pop(i)
            error_mass.pop(i)
            numSmall += 1
            newN -= 1
        elif abs(numDensity[i]) > abs(numDensity[i+1])*delta:# or abs(numDensity[i]) < abs(numDensity[i-1])/3:
            x.pop(i)
            counts.pop(i)
            numDensity.pop(i)
            massDensity.pop(i)
            error_num.pop(i)
            error_mass.pop(i)
            numGrad += 1
            newN -= 1
        else:
            i += 1
    if numDensity[-1] < 0.01 or massDensity[-1] < 0.01:
        x.pop(-1)
        counts.pop(-1)
        numDensity.pop(-1)
        massDensity.pop(-1)
        error_num.pop(-1)
        error_mass.pop(-1)
        numSmall += 1
        newN -= 1
    
    
    print(f"[{cluster.name}] Removed {numSmall} points with too small of a density and {numGrad} points with too extreme of a delta")



    #========= Number Density =========
    
    #Number density vs radial bin plot
    plt.figure(f"{clname}_density_{mode}")
    plt.errorbar(x,numDensity,yerr=error_num,ls='None')
    plt.scatter(x,numDensity)
    plt.xlabel("Radius (deg)")
    plt.ylabel(r"Surface Number Density ($deg^{-2}$)")
    plt.title(f"{clname} {mode.capitalize()} Number Density")
    
    #Fit an exponential curve to the density plot based on the densityProfile function defined above
    
    if "NGC2355" in cluster.name:
        p0=[5000,0.1]
    else:
        p0=[5000,0.1]
    
    #print([b/a for a,b in zip(numDensity,error_num)])
    
    fit,var = so.curve_fit(kingProfile,x,numDensity,p0,maxfev=1000)
    
    #Std. Dev. from variance
    err = np.sqrt(var[1][1])
    err_coeff = np.sqrt(var[0][0])
    
    scale = np.abs(fit[1]*3600/206265)/(cluster.mean_par/1000)
    #scaleVar = (3600/206265)*(err/(cluster.mean_par/1000) ) + 2*fit[1]/(cluster.mean_par_err/1000)
    scaleVar = np.abs(scale*np.sqrt((var[1][1]/fit[1])**2 + (cluster.mean_par_err/cluster.mean_par)**2))
    
    #Scale radius from count in parsecs
    setattr(cluster,f"scaleRad_{mode}",scale)
    setattr(cluster,f"scaleRad_err_{mode}",scaleVar)
    #Scale radius from count in degrees
    setattr(cluster,f"scaleAngle_{mode}",abs(fit[1]))
    setattr(cluster,f"scaleAngle_err_{mode}",err)
    setattr(cluster,f"numDensity_coeff_{mode}",fit[0])
    setattr(cluster,f"numDensity_coeff_err_{mode}",err_coeff)

    
    #Plot the curve fit    
    numLabel = ( f"N={newN} ({mode.capitalize()})"+"\n" 
    + fr"K={fit[0]:.3f} $\pm$ {err_coeff:.3f}" + "\n" 
    + fr"$\rho$={np.abs(fit[1]):.3f}$\degree$ $\pm$ {err:.3f}$\degree$"+ "\n" 
    + fr"R={scale:.3f}pc $\pm$ {scaleVar:.3f}pc" )
    
    plt.plot(x,[kingProfile(a,*fit) for a in x],color='red',label=numLabel)
    plt.fill_between(x,[kingProfile(a,*fit)-kingError(a,fit[0],fit[1],err_coeff,err) for a in x],[kingProfile(a,*fit)+kingError(a,fit[0],fit[1],err_coeff,err) for a in x],label=r'$1\sigma$',edgecolor='none',alpha=0.8,facecolor='salmon')
    plt.legend(fontsize=8,loc='upper right')
    plt.savefig(f"{cluster.imgPath}{clname}_numDensity_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_numDensity_{mode}.png",dpi=500)
    plt.yscale('log')
    plt.savefig(f"{cluster.imgPath}{clname}_numDensity_log_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_numDensity_log_{mode}.png",dpi=500)
    
    
    #Double plot for bounded regions
    if mode == "bounded":
        plt.figure(f"{clname}_density_filtered")
        
        plt.title(f"{clname} Overlaid Number Density")
        plt.errorbar(x,numDensity,yerr=error_num,ls='None',color='midnightblue')
        plt.scatter(x,numDensity,color='midnightblue')
        plt.plot(x,[kingProfile(a,*fit) for a in x],color='darkred',label=numLabel)
        plt.fill_between(x,[kingProfile(a,*fit)-kingError(a,fit[0],fit[1],err_coeff,err) for a in x],[kingProfile(a,*fit)+kingError(a,fit[0],fit[1],err_coeff,err) for a in x],edgecolor='none',alpha=0.8,facecolor='salmon')
        plt.legend(fontsize=8,loc='upper right')
        plt.yscale('linear')
        plt.savefig(f"{cluster.imgPath}{clname}_numDensity_overlay.pdf")
        plt.savefig(f"{cluster.imgPath}png/{clname}_numDensity_overlay.png",dpi=500)
        plt.yscale('log')
        plt.savefig(f"{cluster.imgPath}{clname}_numDensity_log_overlay.pdf")
        plt.savefig(f"{cluster.imgPath}png/{clname}_numDensity_log_overlay.png",dpi=500)
    
    #========= Mass Density =========
    
    #Mass density vs radial bin plot
    plt.figure(f"{clname}_mass_density_{mode}")
    plt.errorbar(x,massDensity,yerr=error_mass,ls='None')
    plt.scatter(x,massDensity)
    plt.xlabel("Radius (deg)")
    plt.ylabel(r"Surface Mass Density ($M_{\odot}*deg^{-2}$)")
    plt.title(f"{clname} {mode.capitalize()} Mass Density")
    
    #Fit an exponential curve to the density plot based on the densityProfile function defined above
    fit_mass,var_mass = so.curve_fit(kingProfile,x,massDensity,p0,maxfev=1000)
    
    #Std. Dev. from variance
    err_mass = np.sqrt(var[1][1])
    err_mass_coeff = np.sqrt(var[0][0])
    
    scale_mass = np.abs(fit_mass[1]*3600/206265)/(cluster.mean_par/1000)
    #scaleVar_mass = (3600/206265)*(err_mass/(cluster.mean_par/1000) ) + 2*fit_mass[1]/(cluster.mean_par_err/1000)
    scaleVar_mass = np.abs(scale_mass*np.sqrt((var_mass[1][1]/fit_mass[1])**2 + (cluster.mean_par_err/cluster.mean_par)**2))
    
    #Scale radius from mass in parsecs
    setattr(cluster,f"scaleRad_mass_{mode}",scale_mass)
    setattr(cluster,f"scaleRad_mass_err_{mode}",scaleVar_mass)
    #Scale radius from mass in degrees
    setattr(cluster,f"scaleAngle_mass_{mode}",abs(fit_mass[1]))
    setattr(cluster,f"scaleAngle_mass_err_{mode}",err_mass)
    setattr(cluster,f"massDensity_coeff_{mode}",fit_mass[0])
    setattr(cluster,f"massDensity_coeff_err_{mode}",err_mass_coeff)
    
    #Plot the curve fit
    massLabel = ( f"N={newN} ({mode.capitalize()})"+"\n" 
    + fr"K={fit_mass[0]:.3f} $\pm$ {err_mass_coeff:.3f}" + "\n" 
    + fr"$\rho$={np.abs(fit_mass[1]):.3f}$\degree$ $\pm$ {err_mass:.3f}$\degree$"+ "\n" 
    + fr"R={scale_mass:.3f}pc $\pm$ {scaleVar_mass:.3f}pc" )
    
    plt.plot(x,[kingProfile(a,*fit_mass) for a in x],color='red',label=massLabel)
    plt.fill_between(x,[kingProfile(a,*fit_mass)-kingError(a,fit_mass[0],fit_mass[1],err_mass_coeff,err_mass) for a in x],[kingProfile(a,*fit_mass)+kingError(a,fit_mass[0],fit_mass[1],err_mass_coeff,err_mass) for a in x],label=r'$1\sigma$',edgecolor='none',alpha=0.8,facecolor='salmon')
    plt.legend(fontsize=8,loc='upper right')
    plt.savefig(f"{cluster.imgPath}{clname}_massDensity_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_massDensity_{mode}.png",dpi=500)
    plt.yscale('log')
    plt.savefig(f"{cluster.imgPath}{clname}_massDensity_log_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_massDensity_log_{mode}.png",dpi=500)
    
    #Double plot for bounded regions
    if mode == "bounded":
        plt.figure(f"{clname}_mass_density_filtered")
        
        plt.title(f"{clname} Overlaid Mass Density")
        plt.errorbar(x,massDensity,yerr=error_mass,ls='None',color='midnightblue')
        plt.scatter(x,massDensity,color='midnightblue')
        plt.plot(x,[kingProfile(a,*fit_mass) for a in x],color='darkred',label=massLabel)
        plt.fill_between(x,[kingProfile(a,*fit_mass)-kingError(a,fit_mass[0],fit_mass[1],err_mass_coeff,err_mass) for a in x],[kingProfile(a,*fit_mass)+kingError(a,fit_mass[0],fit_mass[1],err_mass_coeff,err_mass) for a in x],edgecolor='none',alpha=0.8,facecolor='salmon')
        plt.legend(fontsize=8,loc='upper right')
        plt.yscale('linear')
        plt.savefig(f"{cluster.imgPath}{clname}_massDensity_overlay.pdf")
        plt.savefig(f"{cluster.imgPath}png/{clname}_massDensity_overlay.png",dpi=500)
        plt.yscale('log')
        plt.savefig(f"{cluster.imgPath}{clname}_massDensity_log_overlay.pdf")
        plt.savefig(f"{cluster.imgPath}png/{clname}_massDensity_log_overlay.png",dpi=500)
        
    
    #========= Average Mass =========
    
    averageMass = [a/b for a,b in zip(massDensity,numDensity)]
    
    xDist = [np.abs(a*3600/206265)/(cluster.mean_par/1000) for a in x]
    
    #Average Mass plot
    plt.figure(f"{clname}_average_mass_{mode}")
    plt.scatter(xDist,averageMass,label=fr"N={newN} ({mode.capitalize()})"+"\n"+f"{numPercentileBins} Percentile Bins")
    plt.xlabel("Distance from Center (pc)")
    plt.ylabel(r"Average Stellar Mass ($M_{\odot}$)")
    plt.title(f"{clname} {mode.capitalize()} Average Mass")
    
    
    #Split average mass data into numPercentileBins number of bins
    if mode == "filtered":
        cluster.pMin = xDist[0]
        cluster.pMax = xDist[-1]
        
    pBins = np.linspace(cluster.pMin,cluster.pMax,numPercentileBins+1)
    xBins = []
    for i in range(len(pBins)-1):
        xBins.append((pBins[i]+pBins[i+1])/2)
    pBins = np.delete(pBins,0)
    pBins = np.delete(pBins,-1)
    for b in pBins:
        plt.axvline(x=b,color='black',linestyle='--')
    
    binned = []
    for n in range(numPercentileBins):
        binned.append([])
    
    #Assign the average mass data points to the bins
    for i in range(len(xDist)):
        #Finds the nearest xBin to each x value and sorts the corresponding averageMass into that bin
        val = find_nearest(xBins,xDist[i])
        idx = xBins.index(val)
        binned[idx].append(averageMass[i])
    
    #Creates arrays that are numPercentileBins long that store the standard and quantile means of the points in those bins
    quantileMean = []
    binMean = []
    meanBins = []
    for b in binned:
        if len(b) == 0:
            continue
        binSorted = sorted(b)
        #Finds the index of the lower percentile marker (ex. 20%)
        lower = binSorted.index(find_nearest(binSorted, np.quantile(b,percentile)))
        #Finds the index of the upper percentile marker (ex. 80%)
        upper = binSorted.index(find_nearest(binSorted, np.quantile(b,1-percentile)))
        #Means between lower and upper percentile markers
        quantileMean.append(np.mean(binSorted[lower:upper+1]))
        #Standard Mean
        binMean.append(np.mean(b))
        #Bins
        meanBins.append(xBins[binned.index(b)])
        
    try:
        fit, var = so.curve_fit(kingProfile,xDist,[kingProfile(a,*fit_mass)/kingProfile(a,*fit) for a in x])
        residual_coeff, residual_scaleAngle = fit[0],fit[1]
    except:
        print(f"Unable to fit the residuals for {cluster.name}")
        residual_coeff, residual_scaleAngle = -99, -99
    
    massFit = st.linregress(meanBins,quantileMean)
    fitslope, intercept, rval, pval, fitslope_err, intercept_err = massFit.slope, massFit.intercept, massFit.rvalue, massFit.pvalue, massFit.stderr, massFit.intercept_stderr
    residual_scaleRad = np.abs(residual_scaleAngle*3600/206265)/(cluster.mean_par/1000)
    
    setattr(cluster,f"residual_coeff_{mode}",residual_coeff)
    setattr(cluster,f"residual_scaleAngle_{mode}",residual_scaleAngle)
    setattr(cluster,f"residual_scaleRad_{mode}",residual_scaleRad)
    
    setattr(cluster,f"mass_slope_{mode}",fitslope)
    setattr(cluster,f"mass_slope_err_{mode}",fitslope_err)
    setattr(cluster,f"mass_intercept_{mode}",intercept)
    setattr(cluster,f"mass_intercept_err_{mode}",intercept_err)
    setattr(cluster,f"mass_fit_r2_{mode}",rval**2)
    setattr(cluster,f"mass_fit_p_{mode}",pval)
    
    fitLabel = ( fr"Slope = {fitslope:.3f} $\pm$ {fitslope_err:.3f}" + "\n" 
    + fr"Intercept = {intercept:.3f} $\pm$ {intercept_err:.3f}" + "\n" 
    + fr"$r^2$ = {rval**2:.3f} ({mode.capitalize()})" )
    
    #Plot the quantile and standard means on the existing average mass plot
    plt.scatter(meanBins,quantileMean,color='red',label=f'Interquartile Mean ({mode.capitalize()})')
    plt.plot(xDist,[fitslope*a+intercept for a in xDist],color='red',label=fitLabel)
    #plt.scatter(meanBins,binMean,color='darkgray',label=f'{mode.capitalize()} Standard Mean')
    plt.legend(fontsize=8,loc='upper right')
    plt.savefig(f"{cluster.imgPath}{clname}_averageMass_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_averageMass_{mode}.png",dpi=500)
    
    
    #Double plot for bounded regions
    if mode == "bounded":
        plt.figure(f"{clname}_average_mass_filtered")
        
        plt.title(f"{clname} Overlaid Average Mass")
        plt.scatter(xDist,averageMass,color='midnightblue',label=fr"N={newN} ({mode.capitalize()})"+"\n"+f"{numPercentileBins} Percentile Bins")
        plt.plot(xDist,[fitslope*a+intercept for a in xDist],color='darkred',label=fitLabel)
        plt.scatter(meanBins,quantileMean,color='darkred',label=f'Interquartile Mean ({mode.capitalize()})')
        #plt.scatter(meanBins,binMean,color='black',label=f'{mode.capitalize()} Standard Mean')
        plt.legend(fontsize=8,loc='upper right')
        plt.savefig(f"{cluster.imgPath}{clname}_averageMass_overlay.pdf")
        plt.savefig(f"{cluster.imgPath}png/{clname}_averageMass_overlay.png",dpi=500)
    
    #========= Radius Plot =========
    plt.figure(f"{clname}_characteristic_radius_{mode}")
    plt.scatter([star.ra for star in cluster.unfilteredWide],[star.dec for star in cluster.unfilteredWide],s=0.5,c='lightgray',label='Unfiltered')
    plt.scatter([star.ra for star in cluster.filtered],[star.dec for star in cluster.filtered],s=1,c='midnightblue',label='Filtered')
    outline1 = Circle([x0,y0],1*abs(getattr(cluster,f"scaleAngle_{mode}")),color='red',fill=False,ls='--',label=fr"$\rho$={1*abs(fit[1]):0.3f}$\degree$",alpha=0.7)
    outline2 = Circle([x0,y0],5*abs(getattr(cluster,f"scaleAngle_{mode}")),color='red',fill=False,ls='--',label=fr"5$\rho$={2*abs(fit[1]):0.3f}$\degree$",alpha=0.7)
    #outline3 = Circle([x0,y0],10*abs(getattr(cluster,f"scaleAngle_{mode}")),color='red',fill=False,ls='--',label=fr"10$\rho$={3*abs(fit[1]):0.3f}$\degree$",alpha=0.7)
    plt.gca().add_patch(outline1)
    plt.gca().add_patch(outline2)
    #plt.gca().add_patch(outline3)
    plt.legend(fontsize=10,loc='upper right')
    plt.axis('square')
    plt.xlabel("RA (Deg)")
    plt.ylabel("DEC (Deg)")
    plt.title(f"{clname} {mode.capitalize()} Characteristic Radius")
    plt.gcf().set_size_inches(8,8)
    plt.savefig(f"{cluster.imgPath}{clname}_radialMembership_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_radialMembership_{mode}.png",dpi=500)
    
    if "M67" in clname and mode == "filtered":
        plt.figure(f"{clname}_rings_{mode}")
        plt.scatter([star.ra for star in cluster.unfilteredWide],[star.dec for star in cluster.unfilteredWide],s=0.5,c='lightgray',label='Unfiltered')
        plt.scatter([star.ra for star in cluster.filtered],[star.dec for star in cluster.filtered],s=1,c='midnightblue',label='Filtered')
        
        for i in range(0,len(rings)):
            outline = Circle([x0,y0],rings[i],color='red',fill=False)
            plt.gca().add_patch(outline)
        
        plt.legend(fontsize=10,loc='upper right')
        plt.axis('square')
        plt.xlabel("RA (Deg)")
        plt.ylabel("DEC (Deg)")
        plt.title(f"{clname} Radial Bins")
        plt.gcf().set_size_inches(8,8)
        plt.savefig(f"SpecificPlots/pdf/{clname}_radialBins.pdf")
        plt.savefig(f"SpecificPlots/png/{clname}_radialBins.png",dpi=500)
        plt.xlim(132.7,133)
        plt.ylim(11.67,11.97)
        plt.savefig(f"SpecificPlots/pdf/{clname}_radialBins_center.pdf")
        plt.savefig(f"SpecificPlots/png/{clname}_radialBins_center.png",dpi=500)
    
    
    #========= Stars by Mass =========
    massList = []
    innerMassList = []
    for star in starList:
        massList.append(star.proxyMass)
        if np.sqrt((star.ra-x0)**2+(star.dec-y0)**2) <= getattr(cluster,f"scaleAngle_{mode}"):
            innerMassList.append(star.proxyMass)
    
    mBins = np.arange(min(massList),max(massList)+0.1,0.1)
    inBins = np.arange(min(innerMassList),max(innerMassList)+0.1,0.1)
    plt.figure(f"{clname}_mass_frequency_{mode}")
    plt.xlabel(r"Stellar Mass ($M_{\odot}$)")
    plt.ylabel("Number of Stars")
    plt.title(f"{clname} {mode.capitalize()} Mass Frequency")
    plt.hist(massList,bins=mBins,label=f"Total {mode.capitalize()}")
    plt.hist(innerMassList,bins=inBins,color='midnightblue',label=f'Inside Core Radius ({mode.capitalize()})')
    plt.legend(fontsize=10,loc='upper right')
    plt.savefig(f"{cluster.imgPath}{clname}_massFrequency_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_massFrequency_{mode}.png",dpi=500)

    #Double plot for bounded regions
    if mode == "bounded":
        plt.figure(f"{clname}_mass_frequency_filtered")
        plt.title(f"{clname} Overlaid Mass Frequency")
        plt.hist(massList,bins=mBins,label=f"Total {mode.capitalize()}",color='red')
        plt.hist(innerMassList,bins=inBins,color='darkred',label=f'Inside Core Radius ({mode.capitalize()})')
        plt.legend(fontsize=10,loc='upper right')
        plt.savefig(f"{cluster.imgPath}{clname}_massFrequency_overlay.pdf")
        plt.savefig(f"{cluster.imgPath}png/{clname}_massFrequency_overlay.png",dpi=500)
    
    
    #========= Stars by Magnitude =========
    magList = []
    innerMagList = []
    for star in starList:
        magList.append(star.g_mag-2.1*cluster.reddening-cluster.dist_mod)
        if np.sqrt((star.ra-x0)**2+(star.dec-y0)**2) <= getattr(cluster,f"scaleAngle_{mode}"):
            innerMagList.append(star.g_mag-2.1*cluster.reddening-cluster.dist_mod)
    
    mBins = np.arange(min(magList),max(magList)+0.1,0.1)
    inBins = np.arange(min(innerMagList),max(innerMagList)+0.1,0.1)
    plt.figure(f"{clname}_mag_frequency_{mode}")
    plt.xlabel(r"Absolute G Mag")
    plt.ylabel("Number of Stars")
    plt.title(f"{clname} {mode.capitalize()} Absolute Magnitude Frequency")
    plt.hist(magList,bins=mBins,label=f"Total {mode.capitalize()}")
    plt.hist(innerMagList,bins=inBins,color='midnightblue',label=f'Inside Core Radius ({mode.capitalize()})')
    plt.legend(fontsize=10,loc='upper right')
    plt.savefig(f"{cluster.imgPath}{clname}_magFrequency_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_magFrequency_{mode}.png",dpi=500)

    #Double plot for bounded regions
    if mode == "bounded":
        plt.figure(f"{clname}_mag_frequency_filtered")
        plt.title(f"{clname} Overlaid Absolute Magnitude Frequency")
        plt.hist(magList,bins=mBins,label=f"Total {mode.capitalize()}",color='red')
        plt.hist(innerMagList,bins=inBins,color='darkred',label=f'Inside Core Radius ({mode.capitalize()})')
        plt.legend(fontsize=10,loc='upper right')
        plt.savefig(f"{cluster.imgPath}{clname}_magFrequency_overlay.pdf")
        plt.savefig(f"{cluster.imgPath}png/{clname}_magFrequency_overlay.png",dpi=500)
    
    #========= Stars by Color =========
    colorList = []
    innerColorList = []
    for star in starList:
        colorList.append(star.b_r-cluster.reddening)
        if np.sqrt((star.ra-x0)**2+(star.dec-y0)**2) <= getattr(cluster,f"scaleAngle_{mode}"):
            innerColorList.append(star.b_r-cluster.reddening)
    
    mBins = np.arange(min(colorList),max(colorList)+0.1,0.1)
    inBins = np.arange(min(innerColorList),max(innerColorList)+0.1,0.1)
    plt.figure(f"{clname}_color_frequency_{mode}")
    plt.xlabel(r"Dereddened BP-RP")
    plt.ylabel("Number of Stars")
    plt.title(f"{clname} {mode.capitalize()} Dereddened Color Index Frequency")
    plt.hist(colorList,bins=mBins,label=f"Total {mode.capitalize()}")
    plt.hist(innerColorList,bins=inBins,color='midnightblue',label=f'Inside Core Radius ({mode.capitalize()})')
    plt.legend(fontsize=10,loc='upper right')
    plt.savefig(f"{cluster.imgPath}{clname}_colorFrequency_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_colorFrequency_{mode}.png",dpi=500)

    #Double plot for bounded regions
    if mode == "bounded":
        plt.figure(f"{clname}_color_frequency_filtered")
        plt.title(f"{clname} Overlaid Dereddened Color Index Frequency")
        plt.hist(colorList,bins=mBins,label=f"Total {mode.capitalize()}",color='red')
        plt.hist(innerColorList,bins=inBins,color='darkred',label=f'Inside Core Radius ({mode.capitalize()})')
        plt.legend(fontsize=10,loc='upper right')
        plt.savefig(f"{cluster.imgPath}{clname}_colorFrequency_overlay.pdf")
        plt.savefig(f"{cluster.imgPath}png/{clname}_colorFrequency_overlay.png",dpi=500)
    
    
    
    #========= Other Radii =========    
    massSum = np.sum([star.proxyMass for star in starList])
    intensitySum = np.sum([toIntensity(star.g_mag) for star in starList])
    
    setattr(cluster,f"medianRad_{mode}",np.median([np.abs(star.radDist*3600/206265)/(cluster.mean_par/1000) for star in starList]))
    setattr(cluster,f"medianAngle_{mode}",np.median([star.radDist for star in starList]))
    radialStarList = sorted(starList,key=lambda x: x.radDist)
    
    curMassSum = 0
    curIntSum = 0
    massFound = False
    intFound = False
    
    for star in radialStarList:
        curMassSum += star.proxyMass
        curIntSum += toIntensity(star.g_mag)
        
        if curMassSum > massSum/2 and not massFound:
            setattr(cluster,f"halfMassRad_{mode}",np.abs(star.radDist*3600/206265)/(cluster.mean_par/1000))
            setattr(cluster,f"halfMassAngle_{mode}",star.radDist)
            massFound = True
        if curIntSum > intensitySum/2 and not intFound:
            setattr(cluster,f"halfLightRad_{mode}",np.abs(star.radDist*3600/206265)/(cluster.mean_par/1000))
            setattr(cluster,f"halfLightAngle_{mode}",star.radDist)
            intFound = True
        if massFound and intFound:
            break
    
    plt.figure(f"{clname}_other_radii_{mode}")
    plt.scatter([star.ra for star in cluster.unfilteredWide],[star.dec for star in cluster.unfilteredWide],s=0.5,c='lightgray',label='Unfiltered')
    plt.scatter([star.ra for star in cluster.filtered],[star.dec for star in cluster.filtered],s=1,c='midnightblue',label='Filtered')
    medRad = getattr(cluster,f"medianRad_{mode}")
    medAngle = getattr(cluster,f"medianAngle_{mode}")
    mRad = getattr(cluster,f"halfMassRad_{mode}")
    mAngle = getattr(cluster,f"halfMassAngle_{mode}")
    lRad = getattr(cluster,f"halfLightRad_{mode}")
    lAngle = getattr(cluster,f"halfLightAngle_{mode}")
    print(medAngle)
    outline1 = Circle([x0,y0],medAngle,color='red',fill=False,ls='--',label=fr"Median Star Distance = {medAngle:.3f}$\degree$, {medRad:.3f}pc",alpha=1)
    outline2 = Circle([x0,y0],mAngle,color='darkgreen',fill=False,ls='--',label=fr"Half Mass Radius = {mAngle:.3f}$\degree$, {mRad:.3f}pc",alpha=1)
    outline3 = Circle([x0,y0],lAngle,color='purple',fill=False,ls='--',label=fr"Half Light Radius = {lAngle:.3f}$\degree$, {lRad:.3f}pc",alpha=1)
    plt.gca().add_patch(outline1)
    plt.gca().add_patch(outline2)
    plt.gca().add_patch(outline3)
    plt.legend(fontsize=10,loc='upper right')
    plt.axis('square')
    plt.xlabel("RA (deg)")
    plt.ylabel("DEC (Deg)")
    plt.title(f"{clname} {mode.capitalize()} Various Radii")
    plt.gcf().set_size_inches(8,8)
    plt.savefig(f"{cluster.imgPath}{clname}_otherRadii_{mode}.pdf")
    plt.savefig(f"{cluster.imgPath}png/{clname}_otherRadii_{mode}.png",dpi=500)
    
        

def checkLoaded(cList):
    if 'all' in cList:
        cList = [c.name for c in clusterList]
    else:
        for cl in cList:
            if not cl in clusters:
                loadClusters([cl])
            
    return cList

def saveResults(cList,outdir="results"):
    #Imports
    import numpy as np
    import dill
    import os
    global clusters
    global clusterList
    
    checkLoaded(cList)
    
    #Check and create the relevant directory paths to save/load the results
    if not os.path.isdir(f"{outdir}/"):
        os.mkdir(f"{outdir}/")
    if not os.path.isdir(f"{outdir}/pickled/"):
        os.mkdir(f"{outdir}/pickled/")
    
    else:
        for cl in cList:
            cluster = clusters[cl]
            #Creates a "result cluster" object from the cluster, effectively just stripping away lists
            rCl = resultClusterObj(cluster)
            #Pickle the result cluster object
            with open(f"{outdir}/pickled/{cluster.name}.pk1", 'wb') as output:
                dill.dump(rCl, output)
                
            #Store variables into an array to be printed as csv
            properties = [a for a in dir(rCl) if not a.startswith('_')]
            res = [getattr(rCl,p) for p in properties]
            #Stack into an array of 2 rows with variable names and values
            fin = np.vstack((properties,res))
            np.savetxt(f"{outdir}/{cluster.name}.csv",fin,delimiter=',',fmt='%s')

def loadResults(filter="None",indir="results"):
    #Imports
    import numpy as np
    import dill
    import os
    global resultList
    global resultsIn
    
    assert os.path.isdir("results/")
    resultList = []
    for fn in os.listdir(indir+"/pickled/"):
        #Reads in instances from the saved pickle file
        with open(f"{indir}/pickled/{fn}",'rb') as input:
            res = dill.load(input)
            resultList.append(res)
    resultsIn = True
    toDict()

def refreshProperties(cList=['all']):
    import numpy as np
    global catalogue
    global clusterList
    global clusters
    
    clusterCatalogue()
    checkLoaded(cList)
    
    #Loop through clusters
    for cluster in cList:
        
        reference = None
        
        for cl in catalogue:
            if str(cl.name) == str(cluster.name):
                reference = cl
                print(f"Catalogue match for {cluster.name} found")
                break
        if reference == None:
            print(f"Catalogue match for {cluster.name} was not found, please create one")
            continue

        #Filter all of the methods out of the properties list
        properties = [a for a in dir(reference) if not a.startswith('_')]
        #print(properties)
        #exec(f"print(reference.{properties[1]})")
        #print(properties)
        
        #Now we have a list of all the attributes assigned to the catalogue (the self.variables)
        for p in properties:
            prop =  getattr(reference,p)
            #print(prop)
            exec(f"cluster.{p} = prop")
            try:
                if prop <= -98:
                    print(f"{cluster.name} does not have a specified catalogue value for {p}")
            except:
                continue
        
        #Additional properties that may be useful
        for star in cluster.filtered:
            star.normRA = star.pmra*np.cos(star.dec*np.pi/180)
        
        print(f"{cluster.name} properties refreshed from catalogue")


    

def statPlot(statX,statY,population="open",color="default",square=True,invertY=False,logX=False,logY=False,pointLabels=True,linFit=False,directory='default'):
    #Create plots of stat X vs stat Y across a population of clusters, similar to customPlot()
    #Can be set to use a custom list of clusters, or all clusters of a given type
    #
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import linregress
    global clusters
    global clusterList
    global catalogue
    global resultsIn
    global resultList
    
    
    if not resultsIn:
        loadResults()
    
    #Filter out incorrect inputs
    if type(population) == str:
        population = population.lower()
        try:
            assert population == "open" or population == "globular"
        except:
            print("Specified population type not recognized")
    else:
        try:
            assert type(population) == list
            assert type(population[0]) == str
        except:
            print("Population type given is not valid, must be either a list of cluster name strings or a single string \'open\' or \'closed\'")
            return
        try:
            assert len(population) > 1
        except:
            print("Population statistic plots cannot be made with fewer than 2 clusters given")
            return
        
    
    #Load cluster information from cList
    #This is going to involve using the resultCluster object to read data from each cluster folder in the cList
    cList = []
    banList = ['NGC2204']
    if type(population) == str:
        for res in resultList:
            if res.clType.lower() == population and not res.name in banList:
                cList.append(res)
    else:
        for res in resultList:
            if res.name in population:
                cList.append(res)
    
    if statX.lower() == "b_r" and statY.lower() == "g_mag":
        #Corrected CMD overlay
        
        NUM_COLORS = len(cList)
        cm = plt.get_cmap('nipy_spectral')
        
        
        plt.figure("uncorrected")
        plt.title("Cluster Overlay")
        plt.xlabel("Observed B-R")
        plt.ylabel("Apparent G Mag")
        plt.gca().invert_yaxis()
        plt.gca().set_prop_cycle('color', [cm(1.025*i/NUM_COLORS) for i in range(NUM_COLORS)])
        
        plt.figure("unshifted")
        plt.title("Corrected Cluster Overlay")
        plt.xlabel("Dereddened B-R")
        plt.ylabel("Absolute G Mag")
        plt.gca().invert_yaxis()
        plt.gca().set_prop_cycle('color', [cm(1.025*i/NUM_COLORS) for i in range(NUM_COLORS)])
        
        plt.figure("shifted")
        plt.title("Corrected Cluster Overlay - Offset")
        plt.xlabel("Dereddened B-R")
        plt.ylabel("Absolute G Mag")
        plt.gca().invert_yaxis()
        plt.gca().set_prop_cycle('color', [cm(1.025*i/NUM_COLORS) for i in range(NUM_COLORS)])
        
        index = 0
        offset = 2.5
        for cluster in cList:
            try:
                path = cluster.dataPath
            except:
                path = f"clusters/{cluster.name}/data/"
            
            condensed = np.genfromtxt(f"{path}condensed.csv",delimiter=",")
            cluster.condensed = condensed
            
            #Adjust by cluster.reddening and cluster.dist_mod
            x1 = [a[0] for a in condensed]
            y1 = [a[1] for a in condensed]
            x2 = [a[0]-cluster.reddening for a in condensed]
            y2 = [a[1]-2.1*cluster.reddening-cluster.dist_mod for a in condensed]
            x3 = [a[0]-cluster.reddening for a in condensed]
            y3 = [a[1]-2.1*cluster.reddening-cluster.dist_mod+index*offset for a in condensed]
            
            index += 1
            
            plt.figure("uncorrected")
            plt.scatter(x1,y1,label=f"{cluster.name}")
            
            plt.figure("unshifted")
            plt.axvline(x=1.6,ymax=0.5,color='black',linestyle='--')
            plt.axhline(y=4,xmin=0.59,color='black',linestyle='--')
            plt.scatter(x2,y2,label=f"{cluster.name}")
            
            plt.figure("shifted")
            plt.scatter(x3,y3,label=f"{cluster.name}")
            plt.axvline(x=1.6,color='black',linestyle='--')
            
            # if 'NGC2301' in cluster.name:
            #     for a,b in zip(x2,y2):
            #         print(f"{a},{b}")
        
        
        plt.figure("uncorrected")
        plt.legend(fontsize=10,loc='upper right')
        plt.gcf().set_size_inches(8,6)
        plt.savefig(f"results/plots/pdf/{population}_clusters_stacked_cmd_apparent.pdf")
        plt.savefig(f"results/plots/png/{population}_clusters_stacked_cmd_apparent.png",dpi=500)
        
        plt.figure("unshifted")
        plt.legend(fontsize=10,loc='upper right')
        plt.gcf().set_size_inches(8,6)
        plt.savefig(f"results/plots/pdf/{population}_clusters_stacked_cmd_absolute.pdf")
        plt.savefig(f"results/plots/png/{population}_clusters_stacked_cmd_absolute.png",dpi=500)
        
        plt.figure("shifted")
        plt.legend(fontsize=10,loc='upper right')
        plt.gcf().set_size_inches(8,6)
        plt.savefig(f"results/plots/pdf/{population}_clusters_stacked_cmd_shifted.pdf")
        plt.savefig(f"results/plots/png/{population}_clusters_stacked_cmd_shifted.png",dpi=500)
            
    
    
    else:
        x = [getattr(a, statX) for a in cList]
        y = [getattr(a, statY) for a in cList]
        
        plt.figure()
        plt.xlabel(f"{statX}")
        plt.ylabel(f"{statY}")
        if pointLabels:
            for cluster in cList:
                plt.scatter(getattr(cluster, statX),getattr(cluster, statY),label=cluster.name)
            plt.legend(fontsize="small")
        else:
            plt.scatter(x,y)
        
        if linFit:
            reg = linregress(x,y)
            plt.plot(x,[reg[0]*a+reg[1] for a in x])
        
        plt.savefig(f"SpecificPlots/pdf/{population}_{statX}_{statY}.pdf")
        plt.savefig(f"SpecificPlots/png/{population}_{statX}_{statY}.png",dpi=500)
    
    return

def ageMassFit(t,m0,k):
    import numpy as np
    
    return 1 + m0*np.exp(-1*k*t)

def extinctionLaw(d,M0):
    import numpy as np
    
    return M0 -2.5*np.log10(1/(4*np.pi*d**2))

def resultPlots():
    #Imports
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import linregress
    from scipy.optimize import curve_fit
    global clusters
    global clusterList
    global catalogue
    global resultsIn
    global resultList
    
    
    if not resultsIn:
        loadResults()
    
    #Select open clusters from resultList
    banList = ['NGC2204']
    cList = []
    for res in resultList:
        if res.clType.lower() == "open" and not res.name in banList:
            cList.append(res)
    
    
    #Filtered mass versus age
    fname = "mass_vs_age_filtered"
    plt.figure(fname)
    plt.title(f"{len(cList)} Open Clusters")
    plt.xlabel("Fit Age (Gyr)")
    plt.ylabel(r"Mean Cluster Member Mass ($M_{\odot}$)")
    plt.scatter([c.fit_age for c in cList],[c.meanProxyMass for c in cList])
    plt.savefig(f"results/plots/pdf/{fname}.pdf")
    plt.savefig(f"results/plots/png/{fname}.png",dpi=500)
    
    
    #Bounded mass versus age
    fname = "mass_vs_age_bounded"
    plt.figure(fname)
    plt.title(f"{len(cList)} Open Clusters - BR-RP Limit Enforced")
    plt.xlabel("Fit Age (Gyr)")
    plt.ylabel(r"Mean Cluster Member Mass ($M_{\odot}$)")
    
    x,y = [c.fit_age for c in cList],[c.meanBoundedProxyMass for c in cList]
    plt.scatter(x,y)
    fit,var = curve_fit(ageMassFit,x,y,p0=[8,1],maxfev=1000)
    xr = list(np.linspace(min(x),max(x),101))
    
    fitLabel = fr"$y = 1+{fit[0]:.3f}e^{{-{fit[1]:.3f}t}}$" + "\n" + fr"Uncertainties = $\pm{var[0][0]:.3f}, \pm{var[1][1]:.3f}$"
    
    plt.plot(xr,[ageMassFit(a,fit[0],fit[1]) for a in xr],label=fitLabel)
    plt.legend()
    plt.savefig(f"results/plots/pdf/{fname}.pdf")
    plt.savefig(f"results/plots/png/{fname}.png",dpi=500)
    
    
    #Mass intercept versus age
    fname = "mass_intercept_vs_age_bounded"
    plt.figure(fname)
    plt.title(f"{len(cList)} Open Clusters - BR-RP Limit Enforced")
    plt.xlabel("Fit Age (Gyr)")
    plt.ylabel(r"Mean Stellar Mass in Core ($M_{\odot}$)")
    
    x,y = [c.fit_age for c in cList],[c.mass_intercept_bounded for c in cList]
    plt.scatter(x,y)
    fit,var = curve_fit(ageMassFit,x,y,p0=[8,1],maxfev=1000)
    xr = list(np.linspace(min(x),max(x),101))
    
    fitLabel = fr"$y = 1+{fit[0]:.3f}e^{{-{fit[1]:.3f}t}}$" + "\n" + fr"Uncertainties = $\pm{var[0][0]:.3f}, \pm{var[1][1]:.3f}$"
    
    plt.plot(xr,[ageMassFit(a,fit[0],fit[1]) for a in xr],label=fitLabel)
    plt.legend()
    plt.savefig(f"results/plots/pdf/{fname}.pdf")
    plt.savefig(f"results/plots/png/{fname}.png",dpi=500)
    
    
    #Mass slope versus age
    fname = "mass_slop_vs_age_bounded"
    plt.figure(fname)
    plt.title(f"{len(cList)} Open Clusters - BR-RP Limit Enforced")
    plt.xlabel("Fit Age (Gyr)")
    plt.ylabel(r"IQM Stellar Mass Dropoff ($\frac{M_{\odot}}{pc}$)")
    
    x,y = [c.fit_age for c in cList],[c.mass_slope_bounded for c in cList]
    plt.scatter(x,y)
    plt.savefig(f"results/plots/pdf/{fname}.pdf")
    plt.savefig(f"results/plots/png/{fname}.png",dpi=500)
    
    
    #Magnitude versus distance (Extinction law)
    fname = "mag_vs_dist_bounded"
    plt.figure(fname)
    plt.title(f"{len(cList)} Open Clusters - BR-RP Limit Enforced")
    plt.xlabel("Cluster Distance from Earth (pc)")
    plt.ylabel(r"Mean Apparent G Magnitude")
    
    x,y = [c.meanDist for c in cList],[c.mean_bounded_g_mag for c in cList]
    plt.scatter(x,y)
    fit,var = curve_fit(extinctionLaw,x,y,maxfev=1000)
    xr = list(np.linspace(min(x),max(x),101))
    plt.plot(xr,[extinctionLaw(a,fit[0]) for a in xr],label="Inverse Square Law \n" + fr" $M_0 = {fit[0]:.3f} \pm {var[0][0]:.3f}$")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig(f"results/plots/pdf/{fname}.pdf")
    plt.savefig(f"results/plots/png/{fname}.png",dpi=500)
    
    #Bounded fraction versus distance
    fname = "bounded_fraction_vs_dist"
    plt.figure(fname)
    plt.title(f"{len(cList)} Open Clusters - BR-RP Limit Enforced")
    plt.xlabel("Cluster Distance from Earth (pc)")
    plt.ylabel("Fraction Unaffected by BP-RP Limit")
    
    x,y = [c.meanDist for c in cList],[c.fractionBounded for c in cList]
    plt.scatter(x,y)
    plt.savefig(f"results/plots/pdf/{fname}.pdf")
    plt.savefig(f"results/plots/png/{fname}.png",dpi=500)
    
    
    #Radii
    plt.figure()
    plt.scatter([c.meanGalacticDist for c in cList],[c.halfLightRad_bounded/c.medianRad_bounded for c in cList])


    
def boundedStats(cList,xmax=1.6,saveCl=True,unloadCl=True):
    import numpy as np
    global clusters
    global subList
    for cl in cList:
        checkLoaded([cl])
        cluster = clusters[cl]
        
        subList = [star for star in cluster.filtered if  not (star.b_r-cluster.reddening > xmax and star.g_mag > cluster.cltpy)]
        
        cluster.bounded = subList
        
        #Windowed properties (over the xmin to xmax range)
        cluster.meanBoundedProxyMass = np.mean([a.proxyMass for a in subList])
        cluster.totalBoundedProxyMass = np.sum([a.proxyMass for a in subList])
        cluster.numBounded = len(subList)
        cluster.fractionBounded = len(subList)/len(cluster.filtered)
        cluster.mean_bounded_b_r = np.mean([a.b_r for a in subList])
        cluster.mean_bounded_g_mag = np.mean([a.g_mag for a in subList])
        
        if saveCl:
            saveClusters([cl])
            saveResults([cl])
        if unloadCl:
            unloadClusters([cl])
        
        
        


def tryFits(fitVar='fit_age'):
    from scipy.stats import linregress
    
    global resultsIn
    global resultList
    global props
    global r2
    
    if not resultsIn:
        loadResults()
    
    cList = []
    for res in resultList:
        if res.clType.lower() == "open":
            cList.append(res)
    
    if 'all' in fitVar:
        #List of plottable variables
        props = dir(cList[0])
        props = [a for a in props if not '__' in a]
        propList = [a for a in props if type(getattr(cList[0],a)) == float]
        propList.remove('turnPoint')
        
        
        r2 = []
        
        for pr in propList:
            #List of plottable variables
            props = dir(cList[0])
            props = [a for a in props if not '__' in a]
            props = [a for a in props if type(getattr(cList[0],a)) == float]
            props.remove('turnPoint')
            props.remove(pr)
            
            for prop in props:
                
                x = [getattr(a, pr) for a in cList]
                y = [getattr(a, prop) for a in cList]
                
                reg = linregress(x,y)
                r2.append((pr,prop,reg[2]**2))
        
        r2 = sorted(r2,key = lambda x: x[2],reverse=True)
        
        print("Top 100 r^2 values:")
        for r in r2[:200]:
            print(f"{r[0]} | {r[1]} | {r[2]}")
    
    
    else:
        #List of plottable variables
        props = dir(cList[0])
        props = [a for a in props if not '__' in a]
        props = [a for a in props if type(getattr(cList[0],a)) == float]
        props.remove('turnPoint')
        props.remove(fitVar)
        
        r2 = []
        for prop in props:
            
            x = [getattr(a, fitVar) for a in cList]
            y = [getattr(a, prop) for a in cList]
            
            reg = linregress(x,y)
            r2.append((prop,reg[2]**2))
        
        r2 = sorted(r2,key = lambda x: x[1],reverse=True)
        
        print("Top 20 r^2 values:")
        for r in r2[:20]:
            print(f"{r[0]}   |   {r[1]}")
    
        

def prelimPlot(cl):
    import matplotlib.pyplot as plt
    
    cluster = clusters[cl]
    plt.scatter([a.ra for a in cluster.unfilteredWide],[a.dec for a in cluster.unfilteredWide],s=0.1)
    plt.figure()
    plt.scatter([a.pmra for a in cluster.unfilteredWide],[a.pmdec for a in cluster.unfilteredWide],s=0.1)
    # plt.figure()
    # plt.scatter([a.pmra for a in cluster.unfilteredWide],[a.pmdec for a in cluster.unfilteredWide],s=0.1,c=[a.par for a in cluster.unfilteredWide])
    # plt.set_cmap('cool')
    # clb = plt.colorbar()
    plt.figure()
    plt.scatter([a.b_r for a in cluster.unfilteredWide],[a.g_mag for a in cluster.unfilteredWide],s=0.1)
    plt.gca().invert_yaxis()
    # plt.figure()
    # plt.scatter([a.par for a in cluster.unfilteredWide],[a.par for a in cluster.unfilteredWide],s=0.1,c=[(a.pmra**2 + a.pmdec**2)**0.5 for a in cluster.unfilteredWide])
    # plt.set_cmap('cool')
    
            
