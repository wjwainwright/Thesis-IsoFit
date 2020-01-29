# -*- coding: utf-8 -*-

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

def toDict():
    #Imports
    global isoList
    global isochrones
    global isoIn
    
    if isoIn:
        isoName=[]
        for iso in isoList:
            isoName.append(iso.name)
        isochrones = dict(zip(isoName,isoList))


def copyOver():
    #Imports
    import numpy as np
    global isoList
    global isoList2
    
    isoList2 = np.empty((0,2))
    
    for iso in isoList:
        isoList2 = np.r_[isoList2,[[iso,1]]]
        compareInstances(iso,isoList2[-1][0])


def compareInstances(i1,i2):
    #print(f"{i1}    {i2}")
    if not len(i1.br) == len(i2.br):
        print(f"{i1.name} and {i2.name} are not identical")

def checkAll():
    for i in range(100):
        orig = isoList[i]
        copy = isoList2[i][0]
        named = isochrones[orig.name]
        
        print(f"Position: {i}   Original(Length {len(orig.br)}): {orig} {orig.name}   Copy(Length {len(copy.br)}): {copy} {copy.name}   Named(Length {len(named.br)}): {named} {named.name}   ")

def compareDict():
    for i in range(100):
        orig = isoList[i]
        named = isochrones[orig.name]
        
        print(f"Position: {i}   Original(Length {len(orig.br)}): {orig} {orig.name}  Named(Length {len(named.br)}): {named} {named.name}   ")

def fullCheck(i):
    #Imports
    import numpy as np
    
    orig = isoList[i]
    named = isochrones[orig.name]
    
    print(f"[]========Full check of Isochrone at index {i}========[]")
    print(f"Source name: {orig.name} | Source address: {orig} | Source br: {len(orig.br)} | Source g: {len(orig.g)}")
    print(f"Dict name: {named.name}   | Dict address: {named}   | Dict br: {len(named.br)}   | Dict g: {len(named.g)}")
    
    
    for iso in isoList:
        if len(iso.g) == len(named.g):
            diff = np.mean(np.subtract(iso.g,named.g))
            if diff == 0:
                print(f"The g lists match for original {iso.name} and dictionary {named.name} | {len(iso.g)} = {len(named.g)}")
        if len(iso.br) == len(named.br):
            diff = np.mean(np.subtract(iso.br,named.br))
            if diff == 0:
                print(f"The br lists match for original {iso.name} and dictionary {named.name} | {len(iso.br)} = {len(named.br)}")
                print(orig,iso,named)
    
    print(f"[]========================END========================[]")
    
