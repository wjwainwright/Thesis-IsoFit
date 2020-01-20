isoIn = False
clIn = False

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
    def __init__(self,age=404,feh=404,afe=404,name='genericIsochrone',basedir='isochrones/',subdir='processed',isodir=''):
        #Declare instance variables
        self.name = name
        self.basedir = basedir
        self.subdir = subdir
        self.isodir = isodir
        self.starList = []
        self.age = age
        self. feh = feh
        self.afe = afe
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
        print(len(starlist))
        starlist = preFilter(starlist)
        
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


def testLoad():
    #Imports
    import numpy as np
    import pandas as pd
    global stars
    
    stars = pd.read_csv("clusters/M67/data/wide.csv",sep=',',dtype=str)
    stars = stars.to_numpy(dtype=str)
    #print(stars)


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
            
            feh = float(fehStr[1]+fehStr[2])/10
            afe = float(afeStr[1])/10
            age = float(ageStr)
            
            if fehStr[0] == 'm':
                feh = feh*-1
            if afeStr[0] == 'm':
                afe = afe*-1
            
            #Debug
            #print(f"folder:{folder}   fn:{fn}   fehStr:{fehStr}   feh:{feh}   afeStr:{afeStr}   afe:{afe}   ageStr:{ageStr}   age:{age}")
            
            #Create isochone object
            iso = isochroneObj(age=age,feh=feh,afe=afe,name=f"feh_{feh}_afe_{afe}_age_{age}",basedir=basedir,subdir=subdir,isodir=folder+'/')
            
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


"""
def turboFilter():
    #Imports
    import numpy as np
    global clusterList
    
    iter = 10
    
    for cluster in clusterList:
        starList = np.empty((0,4))
        for star in cluster.unfilteredWide:
            starList.r_[starList,[[star,star.par,star.pmra,star.pmdec]]]
        
        for count in range(iter):
            starList = clumpFilter(cluster,starList)
            starList = distFilter(cluster,starList)
"""

"""
def distFilter():
    #Imports
    import numpy as np
    import matplotlib.pyplot as plt
    global clusterList
    
    for cluster in clusterList:
        parray = np.empty((0,2))
        for star in cluster.unfilteredWide:
            parray = np.r_[parray,[[star,star.par]]]
        parray[parray[:,1].argsort()]
        par = list(parray[:,1])
        pmin = parray[0,1]
        pmax = parray[-1,1]
        div = 500
        seg = (pmax-pmin)/div
        
        slices = []
        
        for i in range(div):
            sliced = parray[par.index(find_nearest(par,pmin+i*seg)):par.index(find_nearest(par,pmin+(i+1)*seg))]
            slices.append(sliced)
        plt.hist([len(x) for x in slices])
"""        



def turboFilter():
    #Imports
    import numpy as np
    global clusterList
    
    for cluster in clusterList:
        
        cluster.filtered,cluster.mag = pmFilter(cluster.unfilteredWide,cluster.name)

        setFlag()

def pmFilter(starList,name):
    #Imports
    import numpy as np
        
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
    
    #Set back to 1.5 after done with capstone paper
    threshold = 100
    
    for star in cluster.unfilteredWide:
        if not np.greater(np.abs(star.par-cluster.mean_par),threshold*cluster.stdev_par):
            cluster.distFiltered.append(star)




def turboFit():
    #Imports
    global clusterList
    
    fitCount = 10
    
    conversion = 2.1
    
    condense()
    
    for cluster in clusterList:
        
        reddening = 0.5
        difference = 0.5
        
        for fit in range(fitCount):
            shapeFit(cluster,reddening + difference)
            upScore = cluster.iso[0][1]
            shapeFit(cluster,reddening - difference)
            downScore = cluster.iso[0][1]
            
            if upScore < downScore:
                reddening = reddening + difference
            else:
                reddening = reddening - difference
            difference = difference/2
        cluster.reddening = reddening
        print(f"Reddening: {reddening}")
        
        cluster.mag[:,0] -= reddening
        cluster.mag[:,1] -= reddening*conversion
    
    condense()
    
    for cluster in clusterList:
        shapeFit(cluster,0)
            
            


def shapeFit(cluster,reddening):
    #Imports
    import numpy as np
    import shapely.geometry as geom
    global isoList
    
    
    conversion = 2.1
    
    cluster.iso = np.empty((0,2))
    for iso in isoList:
        isoLine = geom.LineString(tuple(zip([x+conversion*reddening for x in iso.br],[x+cluster.dist_mod+reddening for x in iso.g])))
        dist = []
        for star in cluster.condensed:
            starPt = geom.Point(star[0],star[1])
            #print(starPt.distance(isoLine))
            dist.append(np.abs(starPt.distance(isoLine)))
        isoScore = np.mean(dist[:])
        #print(list(geom.shape(isoLine).coords))
        cluster.iso = np.r_[cluster.iso,[[iso,isoScore]]]
        #print(isoScore)
    cluster.iso = sorted(cluster.iso,key=lambda x: x[1])
    


"""
def getReddening(iso="isochrones/processed/fehm05afem2/10.000.csv"):
    #Imports
    import numpy as np
    import matplotlib.pyplot as plt
    global isoLine
    global fullIso
    
    fullIso = np.genfromtxt(iso,delimiter=",")
    
    plt.figure("red")
    plt.gca().invert_yaxis()
    plt.scatter(fullIso[:,6]-fullIso[:,7],fullIso[:,5],color='olive')
    
    gmin = 5
    gmax = 8.5
    isoX = []
    isoY = []
    
    for s in fullIso:
        if s[5] >= gmin and s[5] <= gmax:
            isoX.append(s[6]-s[7])
            isoY.append(s[5])
    
    isoLine = np.polyfit(isoX[:],isoY[:],1)
    x=np.linspace(isoX[0],isoX[-1],50)
    y=isoLine[0]*x+isoLine[1]
    plt.plot(x,y,color='midnightblue')
    plt.savefig("reddening.pdf")
"""


def clumpFind(data,count):
    #Imports
    import numpy as np
    from sklearn.neighbors import NearestNeighbors,KDTree
    
    tree = KDTree(data)
    
    return tree.query(tree,k=count,return_distance=False)
    
    


def find_nearest(array, value):
    #Imports
    import numpy as np
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def condense():
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
        
        #Find Turning Point
        for i,point1 in enumerate(condensed):
            count = 0
            total = 0
            for j,point2 in enumerate(condensed):
                if j > i:
                    total += 1
                    if point2[0] > point1[0]:
                        count += 1
            threshold = 0.5
            if not (total == 0) and not (count == 0):
                if count/total >= threshold:
                    turnPoints.append([point1[0],point1[1]])
        
        
        
        if len(turnPoints) == 0:
            print("No turning point identified")
            return
        else:
            turnPoints = sorted(turnPoints,key=lambda x: x[0])
            cluster.turnPoint = turnPoints[0]
            print(f"Turning point: {cluster.turnPoint}")
        
        
        #Recalc with the turnPoint limit enforced - Ignore blue stragglers
        condensed = np.empty((0,3))
        
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
            condensed = np.r_[condensed,[[np.median(sliced[:,0]),np.median(sliced[:,1]),0]]]
        
        condensed = condensed[::-1]
        
        
        distances=[]
        
        #Find average distance to nearby points
        for i,point1 in enumerate(condensed):
            for j,point2 in enumerate(condensed):
                if np.abs(i-j) == 1:
                            distances.append(np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2))
        
        
        medDist = np.median(distances)
        stdevDist = np.std(distances)
        distThreshold = 3
        nearbyThreshold = -1
        
        #Find nearby points
        for i,point1 in enumerate(condensed):
            for j,point2 in enumerate(condensed):
                if not i == j and np.abs(np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)-medDist) < distThreshold*stdevDist:
                    point1[2] += 1
        
        cluster.condensed = np.empty((0,3))
        
        #Remove points 
        for n,pt in enumerate(condensed):
            if pt[0] > cluster.turnPoint[0] and pt[2] > nearbyThreshold:
                cluster.condensed = np.r_[cluster.condensed,[[pt[0],pt[1],pt[2]]]]
        print(f"{len(condensed)-len(cluster.condensed)} points removed")



            
            
"""
def isoSort():
    #Imports
    import numpy as np
    global clusterList
    global isoList
    
    
    for cluster in clusterList:
        
        dist_mod = 5*np.log10(1000/cluster.mean_par)-5
        cluster.iso = []
        smallest = []
        
        for iso in isoList:
            for star in cluster.unfilteredWide:
                for point in iso.starList:
                    
                    point.distance = np.sqrt((star.b_r-(point.b_mag-point.r_mag))**2 + (star.g_mag-(point.g_mag+dist_mod))**2)
                smallest.append(sorted(iso.starList,key=lambda x: x.distance)[0].distance)
            iso.score = np.average(smallest[:])
            print(iso.score,iso.name)
            smallest = []
        cluster.iso.append(sorted(isoList,key=lambda x: x.score)[:20])
"""                    

"""
def polyFit():
    import numpy as np
    import numpy.polynomial.polynomial as poly
    import matplotlib.pyplot as plt
    global isoList
    global clusterList
    
    for cluster in clusterList:
        for iso in isoList[:1]:
            iso.coeff = poly.polyfit(iso.br[:],iso.g[:],9)
            iso.poly = np.polyval(iso.br[:],iso.coeff[:])
            plt.figure(f"{iso}")
            plt.gca().invert_yaxis()
            plt.scatter(iso.br,iso.g,color='olive')
            plt.plot(iso.br,iso.poly,color='midnightblue')
"""

"""
def treeSort():
    #Imports
    import numpy as np
    from sklearn.neighbors import NearestNeighbors,KDTree
    global clusterList
    global isoList
    global isoTree
    global isoArr
    
    
    position = 0
    dist_mod = 5*np.log10(1000/cluster.mean_par)-5

    
    
    
    
    isoArr = np.empty((0,3))
    for point in isoList[0].starList:
        isoArr = np.r_[isoArr,[[point.b_mag-point.r_mag,point.g_mag + dist_mod,position]]]
        position += 1
    
    isoTree=KDTree(isoArr)
    

    for iso in isoList:
        isoArr = []
        for point in iso.starList:
            isoArr.append((point.b_mag-point.r_mag,point.g_mag,position),axis=0)
            position += 1
        
        isoTree=KDTree(isoArr)
"""

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


def plot(pos=True,pm=True,cmd=True,iso=True,test=False):
    #Imports
    import matplotlib.pyplot as plt
    import numpy as np
    global clusterList
    
    for cluster in clusterList:
        
        #Arrays for plotting
        unfra=[]
        unfdec=[]
        ra=[]
        dec=[]
        
        unfpmra=[]
        unfpmdec=[]
        pmra=[]
        pmdec=[]
        
        unfpara = []
        para = []
        
        unfgmag=[]
        unf_b_r=[]
        gmag=[]
        b_r=[]
        ebr_ra = []
        ebr_dec = []
        
        bright_b_r = [x.b_r for x in cluster.filteredBright]
        bright_gmag = [x.g_mag for x in cluster.filteredBright]
        par_b_r = [x.b_r for x in cluster.distFiltered]
        par_gmag = [x.g_mag for x in cluster.distFiltered]
        
        
        
        index=0
        
        #De-tangle star properties
        for star in cluster.unfilteredWide:
            
            #Unfiltered arrays
            unfra.append(star.ra)
            unfdec.append(star.dec)
            unfpmra.append(star.pmra)
            unfpmdec.append(star.pmdec)
            unfgmag.append(star.g_mag)
            unf_b_r.append(star.b_r)
            unfpara.append(star.par)
            
            if not np.isnan(star.e_bp_rp):
                ebr_ra.append(star.ra)
                ebr_dec.append(star.dec)
            
            
            
            if index < len(cluster.filtered):
                #Put info for filtered list too
                if star == cluster.filtered[index]:
                    index += 1
                    ra.append(star.ra)
                    dec.append(star.dec)
                    pmra.append(star.pmra)
                    pmdec.append(star.pmdec)
                    gmag.append(star.g_mag)
                    b_r.append(star.b_r)
                    para.append(star.par)
        
        
            
            
        
        
        #Test plots
        if test:
            #Color Color
            plt.figure(f"{cluster.name}_color_color")
            plt.xlabel('B-R')
            plt.ylabel('B-R')
            plt.title(f"{cluster.name} Color Color")
            plt.scatter(unf_b_r[:],unf_b_r[:],s=0.025,c='olive')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}test/{cluster.name}_color_color.pdf")
            
            #Wack
            plt.figure(f"{cluster.name}_wack")
            plt.xlabel('??')
            plt.ylabel('??')
            plt.title(f"{cluster.name} Wack")
            plt.scatter([a*b for a,b in zip(unfpmra,unfpmdec)],[a*b for a,b in zip(unfra,unfdec)],s=0.025,c='olive')
            #plt.axis("square")
            plt.savefig(f"{cluster.imgPath}test/{cluster.name}_wack.pdf")
            
            #Wack2
            plt.figure(f"{cluster.name}_wack2")
            plt.xlabel('??')
            plt.ylabel('??')
            plt.title(f"{cluster.name} Wack")
            plt.scatter([a*b for a,b in zip(unfpmra,unfpmdec)],unfpara,s=0.025,c='olive')
            #plt.scatter([a*b for a,b in zip(pmra,pmdec)],para,s=0.025,c='midnightblue')
            #plt.axis("square")
            plt.savefig(f"{cluster.imgPath}test/{cluster.name}_wack2.pdf")
            
            #Wack3
            plt.figure(f"{cluster.name}_wack3")
            plt.xlabel('??')
            plt.ylabel('??')
            plt.title(f"{cluster.name} Wack")
            plt.scatter([a*b for a,b in zip(unfpmra,unfpmdec)],unfpara,s=0.025,c='olive')
            #plt.scatter([a*b for a,b in zip(pmra,pmdec)],para,s=0.025,c='midnightblue')
            #plt.axis("square")
            plt.savefig(f"{cluster.imgPath}test/{cluster.name}_wack3.pdf")
        
        
                    
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
            
            #Position reddening overlay
            plt.figure(f"{cluster.name}_ra_dec_reddening_overlay")
            plt.xlabel('RA')
            plt.ylabel('DEC')
            plt.title(f"{cluster.name} Reddening Overlay")
            plt.scatter(unfra[:],unfdec[:],s=0.025,c='olive')
            plt.scatter(ebr_ra[:],ebr_dec[:],s=0.75,c='midnightblue')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_reddening_overlay.pdf")
            
            
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
            plt.title(f"{cluster.name} Reddening")
            plt.scatter(b_r[:],gmag[:],s=0.05,c='olive')
            plt.scatter(b_r[:],gmag[:],s=0.05,c='midnightblue')
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
            plt.scatter([s.b_r - cluster.reddening for s in cluster.filtered],[s.g_mag - 2.1*cluster.reddening for s in cluster.filtered],s=0.05,c='olive',label='Data')
            plt.scatter(cluster.condensed[:,0],cluster.condensed[:,1],s=5,c='midnightblue',label='Proxy Points')
            plt.axvline(x=cluster.turnPoint[0],linestyle='--',color='midnightblue',linewidth=0.8,label='90% of Turning Point')
            plt.legend()
            plt.savefig(f"{cluster.imgPath}{cluster.name}_condensed_CMD_overlay.pdf")
            
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
            plt.scatter([s.b_r - cluster.reddening for s in cluster.filtered],[s.g_mag - 2.1*cluster.reddening for s in cluster.filtered],s=0.05,c='olive',label='Cluster')
            plt.plot(isochrone.br,[x+cluster.dist_mod for x in isochrone.g],c='midnightblue',label=f"{isochrone.name}")
            plt.scatter(cluster.condensed[:,0],cluster.condensed[:,1],s=5,c='red',label='Cluster Proxy')
            plt.legend()
            plt.savefig(f"{cluster.imgPath}{cluster.name}_CMD_Iso_BestFit.pdf")



def specificPlot(cl,iso):
    #Imports
    import matplotlib.pyplot as plt
    import os
    
    cluster = clusters[f"{cl}"]
    isochrone = isochrones[f"{iso}"]
    
    score = 0
    
    if not os.path.isdir("SpecificPlots/pdf/"):
        os.makedirs("SpecificPlots/pdf/")
    if not os.path.isdir("SpecificPlots/png/"):
        os.makedirs("SpecificPlots/png/")
    
    for chrone in cluster.iso:
        if chrone[0].name == iso:
            score = chrone[1]
            break
    
    #Isochrone CMD fit
    plt.figure(f"{cl}_{iso}")
    plt.gca().invert_yaxis()
    plt.xlabel('B-R')
    plt.ylabel('G Mag')
    plt.title(f"{cl} {iso}")
    plt.scatter(cluster.mag[:,0],cluster.mag[:,1],s=0.05,c='olive',label='Cluster')
    plt.plot(isochrone.br,[x+cluster.dist_mod for x in isochrone.g],c='midnightblue',label=f"Score: {score}")
    plt.scatter(cluster.condensed[:,0],cluster.condensed[:,1],s=5,c='red',label='Cluster Proxy')
    plt.legend()
    plt.savefig(f"SpecificPlots/pdf/Requested_Plot_{cl}_{iso}.pdf")
    plt.savefig(f"SpecificPlots/png/Requested_Plot_{cl}_{iso}.png")


def plotRange(cl,a,b):
    global clusters
    
    for isochrone in clusters[f"{cl}"].iso[a:b]:
        specificPlot(cl,isochrone[0].name)
        
def setFlag():
    #Imports
    global clusterlist
    
    for cluster in clusterList:
        for star in cluster.filtered:
            for unfStar in cluster.unfilteredWide:
                if star == unfStar:
                    unfStar.member = 1
        
def customPlot(var1,var2,cluster,mode,iso=False,square=True,color='default'):
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
    
    
    plt.figure(f"customPlot_{var1}_{var2}_{color}")
    plt.title(f"Custom Plot {var1} {var2} {color}")
    plt.xlabel(f"{var1}")
    plt.ylabel(f"{var2}")
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
        

