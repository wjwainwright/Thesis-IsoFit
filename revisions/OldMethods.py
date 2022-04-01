# -*- coding: utf-8 -*-

def testLoad():
    #Imports
    import numpy as np
    import pandas as pd
    global stars
    
    stars = pd.read_csv("clusters/M67/data/wide.csv",sep=',',dtype=str)
    stars = stars.to_numpy(dtype=str)
    #print(stars)





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



def compareInstances(i1,i2):
    print(f"{i1}    {i2}")
    if not len(i1.br) == len(i2.br):
        print(f"{i1.name} and {i2.name} are not identical")



def checkInstances(stage):
    import numpy as np
    
    count = 0
    total = 0
    global positions
    positions = []
    for cli in clusterList[0].iso:
        total = total + 1
        cliso = cli[0]
        iso = isochrones[cliso.name]
        
        
        
        #print(f"Iso: {iso.name} : {iso}    Cliso: {cliso.name} : {cliso}")
        #sim_br = np.mean(np.subtract(iso.br,cliso.br))
        #sim_g = np.mean(np.subtract(iso.g,cliso.g))
        
        if not len(cliso.br) == len(iso.br) :
            #print(f"Iso: {iso.name} {iso}    Cliso: {cliso.name} {cliso} do not match in number of points")
            positions.append(total-1)
            count = count + 1
    print(f"Stage: {stage}    Count: {count}    Total points: {total}    Fraction: {count/total}    Positions: {len(positions)} points over {positions[0]} -> {positions[-1]} ({len(positions)/(positions[-1]-positions[0])*100}%)")



    
def plotPositions():
    import matplotlib.pyplot as plt
    import numpy as np
    global positions
    global isoList
    
    xrange = np.arange(len(isoList))
    yrange = np.zeros(len(isoList))
    
    for x in xrange:
        if x in positions:
            yrange[x] = 1
        else:
            yrange[x] = 0
    
    plt.figure("Instance Positions",figsize=(15,3))
    plt.title("Instance Positions")
    plt.xlabel("Array Position")
    plt.ylabel("Unchanged = 0; Changed = 1")
    plt.scatter(xrange,yrange,s=0.5)
    plt.savefig(f"SpecificPlots/pdf/InstancePositions.pdf")
    plt.savefig(f"SpecificPlots/png/InstancePositions.png")
    
    plt.figure("Instance Positions Line",figsize=(15,3))
    plt.title("Instance Positions")
    plt.xlabel("Array Position")
    plt.ylabel("Unchanged = 0; Changed = 1")
    plt.plot(xrange,yrange)
    plt.savefig(f"SpecificPlots/pdf/InstancePositionsLine.pdf")
    plt.savefig(f"SpecificPlots/png/InstancePositionsLine.png")
    
    
    
    

def testFit(cl,rank):
    #Imports
    import numpy as np
    import shapely.geometry as geom
    global isoList
    
    cluster = clusters[cl]
    iso = cluster.iso[rank][0]
    
    isoLine = geom.LineString(tuple(zip(iso.br,[x+cluster.dist_mod for x in iso.g])))
    dist = []
    for star in cluster.condensed:
        starPt = geom.Point(star[0],star[1])
        #print(starPt.distance(isoLine))
        dist.append(np.abs(starPt.distance(isoLine)))
    isoScore = np.mean(dist[:])
    
    print(isoScore,dist)
    specificPlot(cl,iso.name)
    
    import matplotlib.pyplot as plt
    
    plt.figure(f"{cl}_{iso}")
    plt.gca().invert_yaxis()
    plt.xlabel('B-R')
    plt.ylabel('G Mag')
    plt.title(f"{cl} {iso.name}")
    plt.scatter(cluster.mag[:,0],cluster.mag[:,1],s=0.05,c='olive',label='Cluster')
    plt.plot(iso.br,[x+cluster.dist_mod for x in iso.g],c='midnightblue',label=f"Score: {isoScore}")
    plt.scatter(cluster.condensed[:,0],cluster.condensed[:,1],s=5,c='red',label='Cluster Proxy')
    plt.legend()
    
    


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

#========================================================SNIPPETS================================================================

#TurboFilter
"""
mag = np.empty((0,2))
pos_radius = 5*np.sqrt(cluster.stdev_pmra**2+cluster.stdev_pmdec**2)
for star in filtered:
    if np.less_equal(np.sqrt((star.ra-cluster.mean_ra)**2+(star.dec-cluster.mean_dec)**2),pos_radius):
        cluster.filtered.append(star)
        mag = np.r_[mag,[[star.b_r,star.g_mag]]]
cluster.mag = mag
"""

#Condensed
"""
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

#Plot
"""
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
            
            
            #Position reddening overlay
            plt.figure(f"{cluster.name}_ra_dec_reddening_overlay")
            plt.xlabel('RA')
            plt.ylabel('DEC')
            plt.title(f"{cluster.name} Reddening Overlay")
            plt.scatter(unfra[:],unfdec[:],s=0.025,c='olive')
            plt.scatter(ebr_ra[:],ebr_dec[:],s=0.75,c='midnightblue')
            plt.axis("square")
            plt.savefig(f"{cluster.imgPath}{cluster.name}_ra_dec_reddening_overlay.pdf")
            
            
"""