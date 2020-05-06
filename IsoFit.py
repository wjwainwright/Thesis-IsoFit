isoIn = False
clIn = False
try:
    runCount += 1
except:
    clusterList = []
    runCount = 1

class clusterObj:
    def __init__(self,name='genericCluster',basedir='clusters/',brightThreshold=15):
        #Declare instance variables
        self.name = name
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
        
        #Check directory locations
        import os
        if not os.path.isdir(self.dataPath):
            os.mkdir(self.dataPath)
        if not os.path.isdir(self.imgPath):
            os.mkdir(self.imgPath)



class starObj:
    def __init__(self,name,ra,ra_err,dec,dec_err,par,par_err,par_over_err,pmra,pmra_err,pmdec,pmdec_err,ra_dec_corr,ra_par_corr,ra_pmra_corr,ra_pmdec_corr,dec_par_corr,dec_pmra_corr,dec_pmdec_corr,par_pmra_corr,par_pmdec_corr,pmra_pmdec_corr,astro_n_obs,astro_n_good_obs,astro_n_bad_obs,astro_gof,astro_chi2,astro_noise,astro_noise_sig,astro_match_obs,astro_sigma5d,match_obs,g_mag,b_mag,r_mag,b_r,b_g,g_r,radvel,radvel_err,variable,teff,a_g,e_bp_rp,lum):
        #Declare instance variables
        self.name = name
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
        self.astro_match_obs = float(astro_match_obs)
        self.astro_sigma5d = float(astro_sigma5d)
        self.match_obs = float(match_obs)
        self.g_mag = float(g_mag)
        self.b_mag = float(b_mag)
        self.r_mag = float(r_mag)
        self.b_r = float(b_r)
        self.b_g = float(b_g)
        self.g_r = float(g_r)
        self.radvel = float(radvel)
        self.radvel_err = float(radvel_err)
        self.variable = variable
        self.teff = float(teff)
        self.a_g = float(a_g)
        self.e_bp_rp = float(e_bp_rp)
        self.lum = float(lum)
        self.member = 0
        
        self.par_over_ra = float(par)/float(ra)
        self.par_over_dec = float(par)/float(dec)
        self.par_over_pmra = float(par)/float(pmra)
        self.par_over_pmdec = float(par)/float(pmdec)



class isochroneObj:
    def __init__(self,age=404,feh=404,afe=404,y=404,basedir='isochrones/',subdir='processed',isodir=''):
        #Declare instance variables
        self.basedir = basedir
        self.subdir = subdir
        self.isodir = isodir
        self.starList = []
        self.age = age
        self. feh = feh
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


class condensedPoint:
    def __init__(self,b_r,g_mag,weight):
        self.b_r = b_r
        self.g_mag = g_mag
        self.weight = weight



def readClusters(clusters=["M67"],basedir="clusters/",smRad=0.35):
    #Imports
    import numpy as np
    import pandas as pd
    global clusterList
    global stars
    global clIn
    
    clusterList=[]
    
    #Loop through clusters
    for clname in clusters:
        #Create cluster objects
        cluster = clusterObj(name=clname,basedir=basedir)
        
        if cluster.name == 'NGC752' or cluster.name == 'NGC188':
            cluster.brightThreshold=18
        
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
            
            if np.less_equal(np.sqrt(((star.ra-ramean)*np.cos(np.pi/180*star.dec))**2+(star.dec-decmean)**2),smRad):
                cluster.unfilteredNarrow.append(star)
        clusterList.append(cluster)
        calcStats(cluster)
    rmOutliers()
    clIn = True
    toDict()


def pad(string, pads):
    spl = string.split(',')
    return '\n'.join([','.join(spl[i:i+pads]) for i in range(0,len(spl),pads)])


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
    cols = list(range(1,12))+list(range(32,38))
    
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
        
        #Variables
        pmthreshold = 1*np.sqrt(cluster.stdev_pmra**2+cluster.stdev_pmdec**2)
        pmpthreshold = 50
        parthreshold = 3
        toRemove=[]
        
        #print(cluster.mean_pmra,cluster.mean_pmdec,cluster.stdev_pmra,cluster.stdev_pmdec)
        #print(len(cluster.unfilteredWide))
        
        #Classifies outliers
        for star in cluster.unfilteredWide:
            #print(np.sqrt(((star.pmra-cluster.mean_pmra)*np.cos(np.pi/180*star.pmdec))**2+(star.pmdec-cluster.mean_pmdec)**2),star.pmra,star.pmdec)
            if np.greater(np.sqrt(((star.pmra-cluster.mean_pmra)*np.cos(np.pi/180*star.pmdec))**2+(star.pmdec-cluster.mean_pmdec)**2),pmthreshold) or np.greater(np.sqrt((star.pmra/star.par)**2+(star.pmdec/star.par)**2),pmpthreshold) or np.greater(abs(star.par),parthreshold):
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

def calcStats(cluster,bright=False):
    #Imports
    import numpy as np
    
    #Reads in all the values for a cluster
    par=[]
    ra=[]
    dec=[]
    pmra=[]
    pmdec=[]
    a_g=[]
    e_bp_rp=[]
    
    loopList=[]
    
    if bright:
        loopList = cluster.filteredBright
    else:
        loopList = cluster.unfilteredNarrow
    
    for star in loopList:
        par.append(star.par)
        pmra.append(star.pmra)
        pmdec.append(star.pmdec)
        ra.append(star.ra)
        dec.append(star.dec)
        
        if not np.isnan(star.a_g) and not star.a_g == 0:
            a_g.append(star.a_g)
        if not np.isnan(star.e_bp_rp) and not star.e_bp_rp == 0:
            e_bp_rp.append(star.e_bp_rp)
            
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
    cluster.mean_a_g = np.mean(a_g[:])
    cluster.stdev_a_g = np.std(a_g[:])
    cluster.mean_e_bp_rp = np.mean(e_bp_rp[:])
    cluster.stdev_e_bp_rp = np.std(e_bp_rp[:])
    cluster.mean_par_over_ra = np.mean([x/y for x,y in zip(par,ra)])
    cluster.stdev_par_over_ra = np.std([x/y for x,y in zip(par,ra)])
    
    cluster.dist_mod = 5*np.log10(1000/cluster.mean_par)-5


def saveClusters():
    #Imports
    import pickle
    global clusterList
    
    #Creates a pickle file with all of the saved instances
    for cluster in clusterList:
       with open(f"{cluster.dataPath}filtered.pk1", 'wb') as output:
           pickle.dump(cluster, output, pickle.HIGHEST_PROTOCOL)


def saveIsochrones():
    #Imports
    import pickle
    global clusterList
    
    #Creates a pickle file with all of the saved instances
    for iso in isoList:
       with open(f"{iso.basedir}pickled/{iso.name}.pk1", 'wb') as output:
           pickle.dump(iso, output, pickle.HIGHEST_PROTOCOL)

    
def loadClusters(clusterNames=["M67"],basedir='clusters/'):
    #Imports
    import pickle
    global clusterList
    global clIn
    
    clusterList=[]
    
    for clusterName in clusterNames:
        #Reads in instances from the saved pickle file
        with open(f"{basedir}{clusterName}/data/filtered.pk1",'rb') as input:
            cluster = pickle.load(input)
            clusterList.append(cluster)
    clIn = True
    toDict()


def loadIsochrones(basedir='isochrones/'):
    #Imports
    import pickle
    import os
    global isoList
    global isoIn
    
    isoList=[]
    
    for fn in os.listdir(basedir+"pickled/"):
        #Reads in instances from the saved pickle file
        with open(f"{basedir}pickled/{fn}",'rb') as input:
            iso = pickle.load(input)
            isoList.append(iso)
    isoIn = True
    toDict()



def turboFilter():
    #Imports
    global clusterList
    
    for cluster in clusterList:
        
        cluster.filteredBright,cluster.brightmag = pmFilter(cluster.unfilteredBright,cluster.name)
        print(f"==========================={cluster.name}===========================")
        print(f"bright unf/pm fil: {len(cluster.unfilteredBright)}   /   {len(cluster.filteredBright)}")
        calcStats(cluster,bright=True)
        distFilter(cluster)
        print(f"dist(all): {len(cluster.distFiltered)}")
        cluster.filtered,cluster.mag = pmFilter(cluster.distFiltered,cluster.name)
        print(f"pm(all): {len(cluster.filtered)}")
        
        customPlot('b_r','g_mag',cluster.name,'filtered',iso=True,square=False,color='astro_sigma5d')
        
        magnitude = cutNoise(cluster)
        print(f"noise cutoff: mag {magnitude}   length {len(cluster.filtered)}")
        
        customPlot('b_r','g_mag',cluster.name,'filtered',iso=True,square=False,color='astro_sigma5d')
        
        """
        for i in range(10):
            print(f"{cluster.filtered[i].b_r}   {cluster.mag[i,0]}")
        """
        
        setFlag()
    print(f"=========================Fitting=========================")

def pmFilter(starList,name):
    #Imports
    import numpy as np
        
    pmra_center = 0
    pmdec_center = 0
    pm_radius = 0
    
    if name == 'M67':
        #Thresholds
        pmra_center=-11
        pmdec_center=-3
        pm_radius=1.25
    if name == 'M35':
        #Thresholds
        pmra_center = 2.35
        pmdec_center = -2.9
        pm_radius = 0.4
    if name == 'NGC752':
        #Thresholds
        pmra_center = 9.8
        pmdec_center = -11.7
        pm_radius = 0.5
    if name == 'NGC188':
        #Thresholds
        pmra_center = -2.3
        pmdec_center = -1
        pm_radius = 0.4
    
    assert not pm_radius == 0
    
    filtered = []
    mag = np.empty((0,2))
    
    
    for star in starList:
        if np.less_equal(np.sqrt((star.pmra-pmra_center)**2+(star.pmdec-pmdec_center)**2),pm_radius):
            filtered.append(star)
            mag = np.r_[mag,[[star.b_r,star.g_mag]]]
    return filtered,mag


def distFilter(cluster):
    #Imports
    import numpy as np
    
    threshold = 1.5
    
    for star in cluster.unfilteredWide:
        if not np.greater(np.abs(star.par-cluster.mean_par),threshold*cluster.stdev_par):
            cluster.distFiltered.append(star)



def cutNoise(cluster):
    #Imports
    import numpy as np
    
    stars = sorted(cluster.filtered,key=lambda x: x.g_mag)
    new = []
    newMag = np.empty((0,2))
    
    threshold = 0.5
    
    for i,s in enumerate(stars):
        if s.astro_sigma5d > threshold:
            break
        new.append(s)
        newMag = np.r_[newMag,[[s.b_r,s.g_mag]]]
        
    cluster.filtered = new
    cluster.mag = newMag
    return i


def turboFit(method='pos',minScore=0.001):
    #Imports
    import numpy as np
    from sys import stdout
    import time
    from time import sleep
    global clusterList
    
    
    t0 = time.time()
    
    condense(method)

    for cluster in clusterList:
        cluster.iso = []
        
        redCenter = np.nanmedian([x.e_bp_rp for x in cluster.filtered])
        redMin = redCenter - 0.2
        redMax = redCenter + 0.2
        step = 0.01
        
        if redMin < 0:
            redMin = 0
        
        redList = [round(x,2) for x in np.arange(redMin,redMax+step,step)]
        
        for reddening in redList:
            stdout.write(f"\rCurrent reddening value for {cluster.name}: {reddening:.2f} / {redList[-1]:.2f}")
            shapeFit(cluster,reddening,minScore)
            stdout.flush()
            sleep(0.1)
        
        reddening = cluster.iso[0][2]
        
        cluster.reddening = reddening
        print(f"\nReddening for {cluster.name}: {reddening}")
        
        #cluster.mag[:,0] -= reddening
        #cluster.mag[:,1] -= reddening*2.1
    
    t1 = time.time()
    
    print(f"Total {cluster.name} fit runtime: {t1-t0} seconds")
            
            


def shapeFit(cluster,reddening,minScore):
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
        isoFitList = np.r_[isoFitList,[[iso,isoScore,reddening]]]
        #compareInstances(iso,cluster.iso[-1][0])
        #print(isoScore)
    cluster.iso.extend(isoFitList)
    cluster.iso = sorted(cluster.iso,key=lambda x: x[1])
    best = cluster.iso[1][0]
    #specificPlot(cluster.name,best.name,reddening)
    print(f"\nFirst point of best fit: {best.br[0]+reddening},{best.g[0]+conversion*reddening+cluster.dist_mod}")
    
    

def find_nearest(array, value):
    #Imports
    import numpy as np
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def condense(method):
    #Imports
    import numpy as np
    global clusterList
    global isoList
    global mag
    
    
    for cluster in clusterList:
        mag = cluster.mag[:,:]
        mag[mag[:,1].argsort()]
        gmag = list(mag[:,1])
        gmin = mag[0,1]
        gmax = mag[-1,1]
        div = 50
        seg = (gmax-gmin)/div
        minpoints = 1
        
        condensed = np.empty((0,3))
        
        turnPoints = []

        for i in range(div):
            sliced = mag[gmag.index(find_nearest(gmag,gmin+i*seg)):gmag.index(find_nearest(gmag,gmin+(i+1)*seg))]
            #print(np.array(sliced).shape)
            
            #Skip forseen problems with empty arrays
            if len(sliced) < minpoints:
                continue
            condensed = np.r_[condensed,[[np.median(sliced[:,0]),np.median(sliced[:,1]),0]]]
        
        condensed = condensed[::-1]
        
        #Original turning point method
        """
        #Find Turning Point
        counts=[]
        for i,point1 in enumerate(condensed):
            count = 0
            total = 0
            for j,point2 in enumerate(condensed):
                if j > i:
                    total += 1
                    if point2[0] > point1[0]:
                        count += 1
            counts.append(count)
            if not (total == 0) and not (count == 0):
                turnPoints.append([point1[0],point1[1],count])
        """
        
        #New turning point method
        start = 4
        end = 11
        theta_crit = 5
        
        basex = [a[0] for a in condensed[start:end]]
        basey = [a[1] for a in condensed[start:end]]
        base = np.polyfit(basex,basey,1)
        
        for i,point in enumerate(condensed):
            if i == start:
                continue
            x = [point[0],condensed[start,0]]
            y = [point[1],condensed[start,1]]
            lin = np.polyfit(x,y,1)
            
            point[2] = 180/np.pi*np.arctan(abs( (base[0]-lin[0])/(1+base[0]*lin[0]) ))
            
            if point[2] > theta_crit and i > end:
                turnPoints.append(point)
            

        #Analysis Plot
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(condensed[:,0],condensed[:,1],c=condensed[:,2])
        plt.set_cmap('brg')
        plt.gca().invert_yaxis()
        clb = plt.colorbar()
        clb.ax.set_title("Theta")
        plt.savefig(f'condensed_{cluster.name}')

        
        if len(turnPoints) == 0:
            print("No turning point identified for {cluster.name}")
            return
        else:
            turnPoints = sorted(turnPoints,key=lambda x: x[1])
            tp = turnPoints[-1]
            tp[0] = tp[0] - 0.05*np.abs(tp[0])
            cluster.turnPoint = tp
            
            cl = []
            for point in condensed:
                cl.append(condensedPoint(point[0],point[1],point[2]))
            
            cluster.condensedInit = cl
            #                                     [ B-R , G , Theta ]
            print(f"{cluster.name} Turning Point: {cluster.turnPoint}")
        
        
        #Recalc with the turnPoint limit enforced - Ignore blue stragglers
        condensed = np.empty((0,3))
        yList = []
        
        for i in range(div):
            rawSliced = mag[gmag.index(find_nearest(gmag,gmin+i*seg)):gmag.index(find_nearest(gmag,gmin+(i+1)*seg))]
            
            sliced = np.empty((0,2))
            for point in rawSliced:
                #print(point)
                if point[0] >= cluster.turnPoint[0]:
                    sliced = np.r_[sliced,[[point[0],point[1]]]]
            
            #Skip forseen problems with empty arrays
            if len(sliced) == 0:
                continue
            #print(sliced)
            
            x = np.median(sliced[:,0])
            y = np.median(sliced[:,1])
            yList.append(y)
            
            condensed = np.r_[condensed,[[x,y,1]]]
        
        newTP = find_nearest(yList,cluster.turnPoint[1])
        
        index = 0
        
        for i,point in enumerate(condensed):
            if newTP == point[1]:
                index = i
                #print(f"{point} found to be TP")
                break
        assert not index == 0
        
        #Fit weight parameters
        N = len(condensed)
        alpha = 0.5
        beta = 10
        c = alpha + 1
        
        index = index - 7
        
        for i,point in enumerate(condensed):
            #point[2] = 5/(1+np.abs(index-i))
            if method == 'pos':
                point[2] = c/(1+alpha*np.exp(beta*((i-index)/N)**2))
            
        
        condensed = condensed[::-1]
        
        cl = []
        for point in condensed:
            cl.append(condensedPoint(point[0],point[1],point[2]))
        
        condensed = cl
        
        if cluster.reddening == 0:
            cluster.condensed0 = condensed
        cluster.condensed = condensed
        


def toDict():
    #Imports
    global clusterList
    global clusters
    global isoList
    global isochrones
    global clIn
    global isoIn
    
    if clIn:
        clName=[]
        
        for cluster in clusterList:
            clName.append(cluster.name)
        clusters = dict(zip(clName,clusterList))
    
    if isoIn:
    
        isoName=[]
        
        for iso in isoList:
            isoName.append(iso.name)
        isochrones = dict(zip(isoName,isoList))


def plot(cl=clusterList,pos=True,pm=True,cmd=True,iso=True,test=False):
    #Imports
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    global clusterList
    
    if type(cl[0]) is str:
        clarr = []
        for s in cl:
            clarr.append(clusters[s])
        cl = clarr
    
    for cluster in cl:
        
        #Arrays for plotting
        unfra=[star.ra for star in cluster.unfilteredWide]
        unfdec=[star.dec for star in cluster.unfilteredWide]
        ra=[star.ra for star in cluster.filtered]
        dec=[star.dec for star in cluster.filtered]
        
        unfpmra=[star.pmra for star in cluster.unfilteredWide]
        unfpmdec=[star.pmdec for star in cluster.unfilteredWide]
        pmra=[star.pmra for star in cluster.filtered]
        pmdec=[star.pmdec for star in cluster.filtered]
        
        unfpara=[star.par for star in cluster.filtered]
        para=[star.par for star in cluster.unfilteredWide]
        
        unfgmag=[star.g_mag for star in cluster.unfilteredWide]
        unf_b_r=[star.b_r for star in cluster.unfilteredWide]
        gmag=[star.g_mag for star in cluster.filtered]
        b_r=[star.b_r for star in cluster.filtered]

        bright_b_r = [x.b_r for x in cluster.filteredBright]
        bright_gmag = [x.g_mag for x in cluster.filteredBright]
        par_b_r = [x.b_r for x in cluster.distFiltered]
        par_gmag = [x.g_mag for x in cluster.distFiltered]
        
        
                  
        #Position plots
        if pos:
            #Unfiltered position plot
            plt.figure(f"{cluster.name}_ra_dec_unfiltered")
            plt.xlabel('RA')
            plt.ylabel('DEC')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter(unfra[:],unfdec[:],s=0.025,c='olive')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_unfiltered.pdf")
            
            #Filtered position plot
            plt.figure(f"{cluster.name}_ra_dec_filtered")
            plt.xlabel('RA')
            plt.ylabel('DEC')
            plt.title(f"{cluster.name} Filtered")
            plt.scatter(ra[:],dec[:],s=0.025,c='midnightblue')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_filtered.pdf")
            
            #Position overlay
            plt.figure(f"{cluster.name}_ra_dec_overlay")
            plt.xlabel('RA')
            plt.ylabel('DEC')
            plt.title(f"{cluster.name} Overlay")
            plt.scatter(unfra[:],unfdec[:],s=0.025,c='olive')
            plt.scatter(ra[:],dec[:],s=0.075,c='midnightblue')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_overlay.pdf")
            
            
        #Proper motion plots
        if pm:
            #Unfiltered proper motion plot
            plt.figure(f"{cluster.name}_pm_unfiltered")
            plt.xlabel('PMRA')
            plt.ylabel('PMDEC')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter(unfpmra[:],unfpmdec[:],s=0.05,c='olive')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_unfiltered.pdf")
            
            #Filtered proper motion plot
            plt.figure(f"{cluster.name}_pm_filtered")
            plt.xlabel('PMRA')
            plt.ylabel('PMDEC')
            plt.title(f"{cluster.name} Filtered")
            plt.scatter(pmra[:],pmdec[:],s=0.05,c='midnightblue')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_filtered.pdf")
            
            #Proper motion overlay
            plt.figure(f"{cluster.name}_pm_overlay")
            plt.xlabel('PMRA')
            plt.ylabel('PMDEC')
            plt.title(f"{cluster.name} Overlay")
            plt.scatter(unfpmra[:],unfpmdec[:],s=0.05,c='olive')
            plt.scatter(pmra[:],pmdec[:],s=0.05,c='midnightblue')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_overlay.pdf")
            
            #Unfiltered PM/Parallax
            plt.figure(f"{cluster.name}_pm_over_parallax_unfiltered")
            plt.xlabel('PMRA / Parallax')
            plt.ylabel('PMDEC / Parallax')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter([a/b for a,b in zip(unfpmra,unfpara)],[a/b for a,b in zip(unfpmdec,unfpara)],s=0.05,c='olive')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_over_parallax_unfiltered.pdf")
            
            #Unfiltered PM*Parallax
            plt.figure(f"{cluster.name}_pm_times_parallax_unfiltered")
            plt.xlabel('PMRA * Parallax')
            plt.ylabel('PMDEC * Parallax')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter([a*b for a,b in zip(unfpmra,unfpara)],[a*b for a,b in zip(unfpmdec,unfpara)],s=0.05,c='olive')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_pm_times_parallax_unfiltered.pdf")
        
        
        #CMD plots
        if cmd:
            #Reddening Correction
            plt.figure(f"{cluster.name}_reddening_CMD")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Reddening = {cluster.reddening:.2f}")
            plt.scatter(b_r[:],gmag[:],s=0.05,c='olive',label='Original')
            plt.scatter([s-cluster.reddening for s in b_r[:]],[s-cluster.reddening for s in gmag[:]],s=0.05,c='midnightblue',label='Corrected')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_reddening_CMD.pdf")
            
            #Unfiltered CMD plot
            plt.figure(f"{cluster.name}_CMD_unfiltered")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Unfiltered")
            plt.scatter(unf_b_r[:],unfgmag[:],s=0.05,c='olive')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_unfiltered.pdf")
            
            #Filtered CMD plot
            plt.figure(f"{cluster.name}_CMD_filtered")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Parallax & Proper Motion Filtered")
            plt.scatter(b_r[:],gmag[:],s=0.05,c='midnightblue')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_filtered.pdf")
            
            #CMD overlay
            plt.figure(f"{cluster.name}_CMD_overlay")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Overlay")
            plt.scatter(unf_b_r[:],unfgmag[:],s=0.05,c='olive')
            plt.scatter(b_r[:],gmag[:],s=0.05,c='midnightblue')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_overlay.pdf")
            
            #Condensed CMD overlay
            plt.figure(f"{cluster.name}_condensed_CMD_overlay")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Condensed Overlay")
            plt.scatter([s - cluster.reddening for s in b_r],[s - 2.1*cluster.reddening for s in gmag],s=0.05,c='olive',label='Data')
            plt.scatter([s.b_r - cluster.reddening for s in cluster.condensed],[s.g_mag - 2.1*cluster.reddening for s in cluster.condensed],s=5,c='midnightblue',label='Proxy Points')
            plt.axvline(x=cluster.turnPoint[0] - cluster.reddening,linestyle='--',color='midnightblue',linewidth=0.8,label='95% of Turning Point')
            plt.legend()
            plt.savefig(f"{cluster.imgPath}{cluster.name}_condensed_CMD_overlay.pdf")
            
            #Weighted CMD overlay
            plt.figure(f"{cluster.name}_weighted_CMD_overlay")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Weighted Overlay")
            plt.scatter([s - cluster.reddening for s in b_r],[s - 2.1*cluster.reddening for s in gmag],s=0.05,c='olive',label='Data')
            plt.scatter([s.b_r - cluster.reddening for s in cluster.condensed],[s.g_mag - 2.1*cluster.reddening for s in cluster.condensed],s=5,c=[s.weight for s in cluster.condensed],label='Proxy Points')
            plt.axvline(x=cluster.turnPoint[0] - cluster.reddening,linestyle='--',color='midnightblue',linewidth=0.8,label='95% of Turning Point')
            plt.set_cmap('brg')
            clb = plt.colorbar()
            clb.ax.set_title("Weight")
            plt.legend()
            plt.savefig(f"{cluster.imgPath}{cluster.name}_weighted_CMD_overlay.pdf")
            
            
            #Initial Condensed CMD overlay
            plt.figure(f"{cluster.name}_initial_condensed_CMD_overlay")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Initial Condensed Overlay")
            plt.scatter([s - cluster.reddening for s in b_r],[s - 2.1*cluster.reddening for s in gmag],s=0.05,c='olive',label='Data')
            plt.scatter([s.b_r - cluster.reddening for s in cluster.condensedInit],[s.g_mag - 2.1*cluster.reddening for s in cluster.condensedInit],s=5,c='midnightblue',label='Proxy Points')
            plt.axvline(x=cluster.turnPoint[0] - cluster.reddening,linestyle='--',color='midnightblue',linewidth=0.8,label='95% of Turning Point')
            plt.legend()
            plt.savefig(f"{cluster.imgPath}{cluster.name}_initial_condensed_CMD_overlay.pdf")
            
            #Brightness-PM Filtered CMD plot
            plt.figure(f"{cluster.name}_CMD_bright_filtered")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Bright-Only Proper Motion Filtered")
            plt.scatter(bright_b_r[:],bright_gmag[:],s=0.05,c='midnightblue')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_bright_filtered.pdf")
            
           #Parallax Filtered CMD plot
            plt.figure(f"{cluster.name}_CMD_parallax_filtered")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Parallax Filtered")
            plt.scatter(par_b_r[:],par_gmag[:],s=0.05,c='midnightblue')
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_parallax_filtered.pdf")
            
            
        #Isochrone plots
        if iso:
            
            isochrone = cluster.iso[0][0]
            
            #Isochrone best fit
            plt.figure(f"{cluster.name}_Iso_best")
            plt.gca().invert_yaxis()
            plt.xlabel('B-R')
            plt.ylabel('G Mag')
            plt.title(f"{cluster.name} Isochrone Best Fit")
            plt.scatter([s - cluster.reddening for s in b_r],[s - 2.1*cluster.reddening for s in gmag],s=0.05,c='olive',label='Cluster')
            plt.plot(isochrone.br,[x+cluster.dist_mod for x in isochrone.g],c='midnightblue',label=f"{isochrone.name}")
            plt.scatter([s.b_r - cluster.reddening for s in cluster.condensed],[s.g_mag - 2.1*cluster.reddening for s in cluster.condensed],s=5,c='red',label='Cluster Proxy')
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            h,l = plt.gca().get_legend_handles_labels()
            h.insert(0,extra)
            l.insert(0,f"Reddening: {cluster.reddening}")
            plt.legend(h,l)
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_Iso_BestFit.pdf")



def specificPlot(cl,iso,reddening):
    #Imports
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import os
    
    cluster = clusters[f"{cl}"]
    isochrone = isochrones[f"{iso}"]
    
    score = 0
    
    
    if not os.path.isdir("SpecificPlots/pdf/"):
        os.makedirs("SpecificPlots/pdf/")
    if not os.path.isdir("SpecificPlots/png/"):
        os.makedirs("SpecificPlots/png/")
    
    for chrone in cluster.iso:
        if chrone[0].name == iso and chrone[2] == reddening:
            score = chrone[1]
            break
    
    #Isochrone CMD fit
    plt.figure()
    plt.gca().invert_yaxis()
    plt.xlabel('B-R')
    plt.ylabel('G Mag')
    plt.title(f"{cl} {iso}")
    plt.scatter([s.b_r for s in cluster.filtered],[s.g_mag for s in cluster.filtered],s=0.05,c='olive',label='Cluster')
    plt.plot([x + reddening for x in isochrone.br],[x+cluster.dist_mod+2.1*reddening for x in isochrone.g],c='midnightblue',label=f"Score: {score}")
    plt.scatter([s.b_r for s in cluster.condensed],[s.g_mag for s in cluster.condensed],s=5,c=[s.weight for s in cluster.condensed],label='Cluster Proxy')
    
    plt.set_cmap('brg')
    clb = plt.colorbar()
    clb.ax.set_title("Weight")
    
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    h,l = plt.gca().get_legend_handles_labels()
    h.insert(0,extra)
    l.insert(0,f"Reddening: {reddening}")
    plt.legend(h,l)
    
    plt.savefig(f"SpecificPlots/pdf/Requested_Plot_{cl}_{iso}_Reddening_{reddening}.pdf")
    plt.savefig(f"SpecificPlots/png/Requested_Plot_{cl}_{iso}_Reddening_{reddening}.png")


def plotRange(cl,a,b):
    global clusters
    
    for isochrone in clusters[f"{cl}"].iso[a:b]:
        specificPlot(cl,isochrone[0].name,isochrone[2])
        
def setFlag():
    #Imports
    global clusterlist
    
    for cluster in clusterList:
        for star in cluster.filtered:
            for unfStar in cluster.unfilteredWide:
                if star == unfStar:
                    unfStar.member = 1
        
def customPlot(var1,var2,cluster,mode,iso=False,square=True,color='default',title='default'):
    #Imports
    import matplotlib.pyplot as plt
    
    if mode == 'filtered':
        starlist = clusters[f"{cluster}"].filtered
    elif mode == 'unfiltered':
        starlist = clusters[f"{cluster}"].unfilteredWide
    elif mode == 'bright_filtered':
        starlist = clusters[f"{cluster}"].filteredBright
    elif mode == 'bright_unfiltered':
        starlist = clusters[f"{cluster}"].unfilteredBright
    elif mode == 'duo':
        starlist = clusters[f"{cluster}"].unfilteredWide 
        starlistF = clusters[f"{cluster}"].filtered
    
    
    plt.figure()
    if title == 'default':
        plt.title(f"Custom Plot {var1} {var2} {color}")
    else:
        plt.title(f"{title}")
    plt.xlabel(f"{var1}".upper())
    plt.ylabel(f"{var2}".upper())
    if iso:
        plt.gca().invert_yaxis()
    if mode == 'duo':
         #plt.scatter([eval(f"x.{var1}") for x in starlist],[eval(f"y.{var2}") for y in starlist],s=[0.1+a.member*1.4 for a in starlist],c=[list(('lightgray',eval('z.par')))[z.member] for z in starlist])
         plt.scatter([eval(f"x.{var1}") for x in starlist],[eval(f"y.{var2}") for y in starlist],s=0.25,c='lightgray')
         if color == 'default':    
              plt.scatter([eval(f"x.{var1}") for x in starlistF],[eval(f"y.{var2}") for y in starlistF],s=1.5,c='midnightblue')
         else:
            plt.scatter([eval(f"x.{var1}") for x in starlistF],[eval(f"y.{var2}") for y in starlistF],s=1.5,c=[eval(f"z.{color}") for z in starlistF])
            plt.set_cmap('brg')
            clb = plt.colorbar()
            clb.ax.set_title(f"{color}")
    else:
        if color == 'default':    
            plt.scatter([eval(f"x.{var1}") for x in starlist],[eval(f"y.{var2}") for y in starlist],s=0.1,c='midnightblue')
        else:
            plt.scatter([eval(f"x.{var1}") for x in starlist],[eval(f"y.{var2}") for y in starlist],s=0.5,c=[eval(f"z.{color}") for z in starlist])
            plt.set_cmap('brg')
            clb = plt.colorbar()
            clb.ax.set_title(f"{color}")
    if square:
        plt.axis("square")
    plt.savefig(f"SpecificPlots/pdf/Custom_Plot_{var1}_{var2}.pdf")
    plt.savefig(f"SpecificPlots/png/Custom_Plot_{var1}_{var2}.png")
        

def splitMS(clname='M67',slope=3,offset=12.2):
    #Imports
    import numpy as np
    import matplotlib.pyplot as plt
    
    cluster = clusters[clname]
    
    xlist = [s.b_r for s in cluster.filtered]
    ylist = [s.g_mag for s in cluster.filtered]
    
    x = np.linspace(1,2,100)
    
    
    plt.figure()
    plt.title('Main sequence Spread')
    plt.xlabel('B-R')
    plt.ylabel('G Mag')
    plt.scatter(xlist,ylist,s=0.05,c='olive',label='Filtered Star Data')
    plt.plot(x,[slope*a + offset for a in x],color='midnightblue',label='Main Sequence')
    plt.plot(x,[slope*a + offset - 0.75 for a in x],'--',color='midnightblue',label='MS shifted 0.75 mag')
    plt.xlim(0.6,2.2)
    plt.ylim(13,19)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig(f"{clname}_MS_Spread.png")
    plt.savefig(f"{clname}_MS_Spread.pdf")

