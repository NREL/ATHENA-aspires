########
#ASPIRES: Airport Shuttle Planning and Improved Routing Event-driven Simulation
## Author: Qichao Wang
## Purpose: Fast simulate and evaluate the airport shuttle operation
import simpy
import random
import numpy as np
import sys, os
import pandas as pd
import pickle
import glob
import time
from os import path
import calendar
import argparse
# input
p = argparse.ArgumentParser()

p.add_argument('--TimeToCharger')
p.add_argument('--RoutesFile')
p.add_argument('--RoutesFileFolder')
p.add_argument('--StartingDayOfWeek')
p.add_argument('--doHotShot')
p.add_argument('--SimTime')
p.add_argument('--NumEVCharger')
p.add_argument('--outputName')
p.add_argument('--randSeed')
p.add_argument('--numEV')
p.add_argument('--OnRouteCharging')
p.add_argument('--ArrivalSFactor')
p.add_argument('--maxqueue')
p.add_argument('--doNightTimeOndemand')
p.add_argument('--NightTimeOndemandInterval')
p.add_argument('--doRegularOndemand')
p.add_argument('--numOndemand')
p.add_argument('--OnDemandCapacity')


print('current directory', os.path.dirname(os.path.abspath(__file__)))
cdr = os.path.dirname(os.path.abspath(__file__))
#print('current directory',sorted(glob.glob('./*')))
args = p.parse_args()
if args.TimeToCharger == None:
    TimeToCharger = 0
else:
    TimeToCharger = int(args.TimeToCharger)

if args.OnRouteCharging == None:
    onRouteCharging = False
elif args.OnRouteCharging== 'True':
    onRouteCharging = True
else:
    onRouteCharging = False
if args.doRegularOndemand == None:
    ifDoRegularOndemand = False
elif args.doRegularOndemand== 'True':
    ifDoRegularOndemand = True
else:
    ifDoRegularOndemand = False

if args.OnDemandCapacity == None:
    OnDemandCapacity = 14
else:
    OnDemandCapacity = int(args.OnDemandCapacity)

if args.NightTimeOndemandInterval == None:
    NightTimeOndemandCheckInterval = 60*15
else:
    NightTimeOndemandCheckInterval = int(args.NightTimeOndemandInterval)

if args.numOndemand == None:
    numOndemand = 0
else:
    numOndemand = int(args.numOndemand)

if args.doNightTimeOndemand == None:
    ifDoNightTimeOndemand = False
elif args.doNightTimeOndemand== 'True':
    ifDoNightTimeOndemand = True
else:
    ifDoNightTimeOndemand = False
if args.maxqueue == None:
    MAX_QUEUE = 999
else:
    MAX_QUEUE = int(args.maxqueue)
if args.RoutesFile== None:
    RoutesFile=None
else:
    RoutesFile= args.RoutesFile
    RoutesFileFolder = args.RoutesFileFolder
if args.doHotShot== None:
    doHotShot=True
elif args.doHotShot== 'True':
    doHotShot = True
else:
    doHotShot = False
if args.ArrivalSFactor== None:
    ArrivalSFactor = 1
else:
    ArrivalSFactor = int(args.ArrivalSFactor)
if args.numEV== None:
    numEV=10
else:
    numEV = int(args.numEV)
if args.randSeed== None:
    rndSeed=0
else:
    rndSeed = int(args.randSeed)
if args.StartingDayOfWeek == None:
    StartingDayOfWeekNum=0
else:
    StartingDayOfWeekNum = int(args.StartingDayOfWeek)
if args.SimTime == None:
    simTime = 1
else:
    simTime = int(args.SimTime)
if args.NumEVCharger== None:
    numEVCharger = 3
else:
    numEVCharger = int(args.NumEVCharger)
if args.outputName== None:
    outputName = 'output'
else:
    outputName = args.outputName


NightStartHr = 24
NightEndHr = 6

Day = calendar.day_name[StartingDayOfWeekNum]
DayOfWeekNum=StartingDayOfWeekNum
def nextDay(env):
    '''
    Update the DayOfWeekNum and listRoutes for the next day
    '''
    global DayOfWeekNum, Day, listRoutes
    while True:
        yield env.timeout(3600*24)
        DayOfWeekNum += 1
        DayOfWeekNum = DayOfWeekNum %7
        Day = calendar.day_name[DayOfWeekNum]
        listRoutes=list(RoutesData[DayOfWeekNum].index)
        # print('this is day {}'.format(DayOfWeekNum))
        # print('list Routes')
        # print(listRoutes)
#Global variables
# current num bus in each route
dic_numBusInRoute={}
RoutesData={}

#
def convertTimeAndDay_time(secondsInSimulation):
    '''
    convert the current seconds in simulation into the seconds in a day
    '''
    secInDay = secondsInSimulation%(3600*24)
    return secInDay
def convertTimeAndDay_day(secondsInSimulation):# haven't been used
    '''
    conver the current seconds in simulation into the day index
    '''
    global StartingDayOfWeekNum
    dayNum = (StartingDayNum +secondsInSimulation//(3600*24))%7
    return dayNum
# Base line data
#def getOptimizedRouteData(dayOfWeekNum,mr=20,hw=20,sd=2,routes=5):
#    day = calendar.day_name[dayOfWeekNum]
#    aRoutesData = pd.read_csv('./data/{}_mr={}_hw={}_sd={}_routes={}.csv'.format(day,mr,hw,sd,routes),index_col=0)
#    return aRoutesData
def getOptimizedRouteData(dayOfWeekNum,routesFile,folder):
    '''
    Import the optimized routes data
    '''
    day = calendar.day_name[dayOfWeekNum]
    aRoutesData = pd.read_csv('{}/{}_{}'.format(folder,day,routesFile),index_col=0)
    #night time
    if ifDoNightTimeOndemand:
        lenData = len(aRoutesData)
        for currentHrIndex in range(24):
            if currentHrIndex<NightEndHr or currentHrIndex>=NightStartHr:
                aRoutesData[str(currentHrIndex)]=list(np.zeros(lenData,dtype=int))
    return aRoutesData
def getBaseLineRouteData(dayOfWeekNum):
    '''
    Used the baseline route data
    '''
    day = calendar.day_name[dayOfWeekNum]
    aRoutesData = pd.read_csv(cdr+'/data_2020/FleetSize_MixedRoute/'+day+'FleetSize.csv',index_col=0)
    #region round baseline RoutesData to integer
    for i in aRoutesData.index:
        l = []
        for e in aRoutesData.loc[i]:
            if e > 0.8:
                l.append(int(e+0.5))
            elif e > 0:
                l.append(1)
            else:
                l.append(0)
        aRoutesData.loc[i]=l
    for c in aRoutesData.columns:
        aRoutesData[c]=aRoutesData[c].astype(int)
    return aRoutesData
if RoutesFile == None:
    for dayOfWeekNum in range(7):
        RoutesData[dayOfWeekNum]=getBaseLineRouteData(dayOfWeekNum)
else:
    for dayOfWeekNum in range(7):
        RoutesData[dayOfWeekNum]=getOptimizedRouteData(dayOfWeekNum,RoutesFile,RoutesFileFolder)

#endregion
#region Get Data
#region get arrival rate data
def getArrivalRateData():
    '''
    Get the arrival rate data from the SPOT data
    '''
    filenames = sorted(glob.glob(cdr + '/data_2020/SPOT/*.csv'))
    res = [ pd.read_csv(filename,parse_dates=True) for filename in filenames ]
    df = pd.concat(res)
    df['DateTime']= pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    df['Day_of_Week'] = df['DateTime'].dt.dayofweek
    df['Hour_of_Day'] = df['DateTime'].dt.hour
    boarding = pd.pivot_table(df[df.Boardings<=43], values='Boardings', index=['Day_of_Week', 'Hour_of_Day'],columns=['StopName'], aggfunc=np.sum)
    alighting = pd.pivot_table(df[df.Alightings<=43], values='Alightings', index=['Day_of_Week', 'Hour_of_Day'],columns=['StopName'], aggfunc=np.sum)
    boarding.fillna(0)
    alighting.fillna(0)
    dic_numDay={}
    for date in df['DateTime'].dt.date.unique():
        d = date.weekday()
        if d not in dic_numDay.keys():
            dic_numDay[d]=1
        else:
            dic_numDay[d]+=1
    return [dic_numDay, alighting, boarding]
#['R', 'RA (1-15)', 'RA (16-39)', 'A (1-15)', 'A (16-39)']

arrRate =cdr + '/data_2020/arrRate.pickle'
if path.exists(arrRate):
    print('arrRate exist')
    [alighting_dic, boarding_dic]=pickle.load(open(arrRate,'rb'))
else:
    print('arrRate does not exist')
    [dic_numDay, alighting, boarding] = getArrivalRateData()
    listStopNames = alighting.columns
    alighting_dic = {}
    boarding_dic = {}
    for s in listStopNames:
        for d in range(7):
            for h in range(24):
                alighting_dic[(s,d,h)] = alighting.xs((d,h))[s]/dic_numDay[d]
                boarding_dic[(s,d,h)] = boarding.xs((d,h))[s]/dic_numDay[d]
                if np.isnan(alighting_dic[(s,d,h)]):
                    alighting_dic[(s,d,h)]=1
                if np.isnan(boarding_dic[(s,d,h)]):
                    boarding_dic[(s,d,h)]=1
    pickle.dump([alighting_dic,boarding_dic],open(arrRate,'wb'))
def getArrivalRate(StopName = 'A (16-39)',DayOfWeek=0,HourOfDay=0):
    '''
    Find the arrival rate at a certain hour of a certain day of the week for a certain stop
    '''
    # DayOfWeek = 0 #[0,6]
    # HourOfDay = 0 #[0,24]
    if StopName[0]=='R':
        _stopName = 'Term '+StopName[1:]
        # print(DayOfWeek,HourOfDay,_stopName)
        return alighting_dic[(_stopName,DayOfWeek,HourOfDay)]
    else:
        _stopName = 'Term '+StopName
        return boarding_dic[(_stopName,DayOfWeek,HourOfDay)]

#endregion
#region Trip segment data -> to be revised by Zhaocai
def collect_sumo_trips(df_sumo,trip,hour):
    '''
    get the simulation trip data from SUMO results
    '''
    df_temp = df_sumo[(df_sumo['Route']==trip) & (df_sumo['Hour']==hour)]
    l = []
    for index, r in df_temp.iterrows():
        l.append((r['Distance'],r['Duration'],r['Distance']*2,0))
    return l
def populate_with_sumo(DTED):
    '''
    Add simulated data into DTED dictionary
    '''
    df_sumo = pd.read_csv(cdr+'/data_2020/SUMO_AverageDayBusOutput.csv')
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_windows = range(int(24/4+0.5))
    terminals = ['A','B','C','D','E','R']
    ODs = []
    for i in terminals:
         for j in terminals:
            ODs.append((i,j))
    for OD in ODs:
        for day in days:
            for tw in time_windows:
                if DTED[day][OD[0]][OD[1]][tw][0][0]=='-1':
                    trip = str(OD[0]) + str(OD[1])
                    l=[]
                    for hr in range(tw*4,(tw+1)*4):
                        alist=collect_sumo_trips(df_sumo,trip,hr)
                        l.extend(alist)
                    DTED[day][OD[0]][OD[1]][tw]= l
        flatDTED = flattenDTED_dic(DTED)
    return flatDTED
def flattenDTED_dic(DTED):
    '''
    flatten the dictionary to improve performance
    '''
    newDTED={}
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_windows = range(int(24/4+0.5))
    terminals = ['A','B','C','D','E','R']
    ODs = []
    for i in terminals:
         for j in terminals:
            ODs.append((i,j))
    for OD in ODs:
        for day in days:
            for tw in time_windows:
                newDTED[(day,OD[0],OD[1],tw)]=DTED[day][OD[0]][OD[1]][tw]
    return newDTED
if path.exists(cdr+'/data_2020/Time_n_Energy_Dictionary_Nested_Full_p.npy'):
    DistanceTravelTimeEnergyDwellT=np.load(cdr+'/data_2020/Time_n_Energy_Dictionary_Nested_Full_p.npy',allow_pickle='TRUE').item()
else:
    DistanceTravelTimeEnergyDwellT=np.load(cdr+'/data_2020/Time_n_Energy_Dictionary_Nested_Full.npy',allow_pickle='TRUE').item()
    DistanceTravelTimeEnergyDwellT=populate_with_sumo(DistanceTravelTimeEnergyDwellT)
    np.save(cdr+'/data_2020/Time_n_Energy_Dictionary_Nested_Full_p.npy', DistanceTravelTimeEnergyDwellT)
def getDistanceTravelTimeEnergyDwellT(From,To,Day,SimulationTime):
    TimeOfDay = SimulationTime % (3600*24)
    hrOfDay = int(TimeOfDay//3600//4)
    if (From[0],To[0]) ==('R','P') or (From[0],To[0])==('P','R') or (From,To)==('R','RP'):
        return (0,20,0,0)
    elif From =='R':
        listDTED = DistanceTravelTimeEnergyDwellT[(Day,From,'R',hrOfDay)]
    elif From[0] == 'R' and To[0]=='R':
        return (0,0,0,0)
    else:
        From = From[0]
        To = To[0]
        listDTED = DistanceTravelTimeEnergyDwellT[(Day,From,To,hrOfDay)]
    listDTED_len = len(listDTED)
    point = random.randint(0,listDTED_len-1)
    DTED = listDTED[point]
    return DTED
#endregion
#region add idle routes (R_P)
def getTotalBusesNeeded(routesData,extraNumBus = 2):
    '''
    Get the total buses needed based on the routes file and the extra number of buses
    '''
    maxNumBusNeed={43:0,14:0}
    for aRoutesData in routesData.values():
        capacities=[]
        for r in aRoutesData.index:
            capacities.append(int(r.split('_')[-1]))
        aRoutesData['Capacity']=capacities
        for cap in [43,14]:
            totalBusNeeded = extraNumBus + max(aRoutesData.loc[aRoutesData['Capacity']==cap].sum(axis = 0, skipna = True)[0:24])
            maxNumBusNeed[cap]=max(maxNumBusNeed[cap],totalBusNeeded)
    return maxNumBusNeed
def addIdleRoutes(aRoutesData,totalBusNeeded):
    '''
    Add the idle routes for buses that are not in service
    '''
    capacities=[]
    for r in aRoutesData.index:
        capacities.append(int(r.split('_')[-1]))
    aRoutesData['Capacity']=capacities
    for cap in [43,14]:
        aRoutesData.loc['R_P_%d'%cap]=(totalBusNeeded[cap] - aRoutesData.loc[aRoutesData['Capacity']==cap].sum(axis = 0, skipna = True))
        capacities=[]
        for r in aRoutesData.index:
            capacities.append(int(r.split('_')[-1]))
        aRoutesData['Capacity']=capacities
    return aRoutesData
totalBusNeeded = getTotalBusesNeeded(RoutesData)
# print('routes data',RoutesData)
# print('total bus needed', totalBusNeeded)
for dayOfWeekNum in range(7):
    RoutesData[dayOfWeekNum]=addIdleRoutes(RoutesData[dayOfWeekNum],totalBusNeeded)
listRoutes=list(RoutesData[StartingDayOfWeekNum].index)
#initialize num bus in each route
for r in listRoutes:
    dic_numBusInRoute[r]=RoutesData[StartingDayOfWeekNum]['0'][r]
#endregion
#endregion
#region Add stops per terminal to route
StopsPerTerm={'A':[' (16-39)',' (1-15)'],
    'B':[' (21-49)',' (1-20)'],
    'C':[ ' (26-39)',' (1-25)'],
    'D':[' (23-40)',' (1-22)'],
    'E':[' (18-38)', ' (11-17)', ' (1-10)'],
    'P':['']
}
def addRccStops2Route(routeOfStops):
    # add rcc stops to route
    #  break route into segemnts
    route_pre = []
    segId=-1
    for s in routeOfStops:
        if s == 'R':
            route_pre.append([])
            segId+=1
        else:
            route_pre[segId].append(s)
    #  add rcc stops
    route_post=[]
    n = len(route_pre)
    for i in range(n):
        seg = route_pre[i]
        route_post.append('R')
        for j in range((len(seg))):#change
            route_post.append('R%s'%seg[j])
        for j in range((len(seg))):#change
            route_post.append(seg[j])
    #  add in-terminal stops
    return route_post
def addStops2Route(route):
    newRoute = []
    for stop in route:
        if stop == 'R':
            newRoute.append(stop)
        elif stop[0]=='R':
            for i in StopsPerTerm[stop[1]]:
                newRoute.append(stop+i)
        else:
            for i in StopsPerTerm[stop]:
                newRoute.append(stop+i)
    return newRoute
#endregion
def getNewRouteName(currentRouteName,currentTime,toPrint=False):
    '''
    give the shuttle a new route based on the current time and the current route
    '''
    global DayOfWeekNum, listRoutes,NightStartHr,NightEndHr
    currentHrIndex = int(currentTime//3600)%24
    currentRoute = currentRouteName
    # if currentHrIndex<NightEndHr or currentHrIndex>=NightStartHr:
    #     _capacity=currentRoute.split('_')[-1]
    #     return 'R_P_'+_capacity #parking route
    if currentRoute in RoutesData[DayOfWeekNum][str(currentHrIndex)].index:
        targetNumber = RoutesData[DayOfWeekNum][str(currentHrIndex)][currentRoute]
    else:
        targetNumber = 0
    if toPrint:
        print('targetNumber',targetNumber)
        print('list routes',listRoutes)
        print('routesdata[][]',RoutesData[DayOfWeekNum][str(currentHrIndex)])
    if dic_numBusInRoute[currentRoute]>targetNumber:
        for r in listRoutes:
            if r.split('_')[-1]==currentRoute.split('_')[-1]:
                if not (r in dic_numBusInRoute):
                    dic_numBusInRoute[r]=0
#                 if r
                if dic_numBusInRoute[r] < RoutesData[DayOfWeekNum][str(currentHrIndex)][r]:
                    dic_numBusInRoute[r]+=1
                    dic_numBusInRoute[currentRoute]-=1
                    return r
    else:
        return currentRoute
# dic_numBusInRoute

# night time on-demand shuttle
def ondemandAtNight(env,stops,stopName,checkingInterval=60*15):
    '''
    Night time on-demand shuttle operation
    '''
    global NightEndHr,NightStartHr
    NightStartTime = NightStartHr*3600
    NightEndTime = NightEndHr*3600
    while True:
        yield env.timeout(checkingInterval)
        if convertTimeAndDay_time(env.now) < NightEndTime or convertTimeAndDay_time(env.now) > NightStartTime:
            if stops[stopName].level>0:
                stops[stopName].onDemandBuses.callOndemand(stopName)
        # get the num pax waiting
        # else:
        #     yield env.timeout(60*60)
        #

#region pax arrival
def TimeDependentPassengerArrival(env,term, termName):
    '''
    Passengers arrival process
    '''
    global DayOfWeekNum
    while True:
        term.put(1)
        # print(termName+' %d has '%termI+str(term.level)+ ' people arrival at %d'% env.now)
        hr=env.now//3600%24
        ArrivalRate = getArrivalRate(StopName=termName,DayOfWeek=DayOfWeekNum,HourOfDay=hr)*ArrivalSFactor
        meanTime = 3600/ArrivalRate
        headway = min(randomExp(meanTime),3600)
        yield env.timeout(headway)
#endregion

#region distribution functions
def ecdf(data):
    ''' Compute ECDF '''
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)
def randomExp(meanTime):
    '''return a random number from the exponential districution'''
    U = random.uniform(0,1)
    return -meanTime*np.log(U)
def randomEmpirical(x,y):
    '''random number from a emperical distribution'''
    xp = x.tolist()
#     xp.insert(0,0)
    yp = y.tolist()
#     yp.insert(0,0)
    U = random.uniform(0,1)
    return np.interp(U,yp,xp,left=xp[0])
# used a pre-constructed dense data and sample from this pool
def randomQuickEmpirical(x):
    '''get a quick empirical distribution based on data and do not have interpolation'''
    U = random.randint(0,len(x)-1)
    return x[U]
#endregion

#region all the classes

class stop_class:
    def __init__(self, env, container,record=True,onDemandBuses=[]):
        self.container = container
        self.env = env
        self.event_in = []
        self.event_out = []
        self.num_in = 0
        self.num_out = 0
        self.level = 0
        self.toRecord = record
        # self.records = pd.DataFrame(columns=['time','num_in','num_out'])
        # self.busArrivalRecords=pd.DataFrame(columns=['time','peopleLeft'])
        self.recordscolumns=['time','num_in','num_out']
        self.busArrivalRecordscolumns=['time','peopleLeft']
        self.records = []
        self.unservedcolumns=['time','people']
        self.busArrivalRecords=[]
        self.unserved= []
        self._lock = False
        self.onDemandBuses=onDemandBuses
    def recordPeopleLeft(self):
        # new_row=[self.env.now,self.level]
        # self.busArrivalRecords = self.busArrivalRecords.append(new_row, ignore_index=True)
        self.busArrivalRecords.append([self.env.now,self.level])
        # self.busArrivalRecords.loc[len(self.busArrivalRecords)] = [self.env.now,self.level]
    def lock(self):
        self._lock = True
    def unlock(self):
        self._lock = False
    def put(self, number):
        if number!=0:
            self.container.put(number)
        if self.level < MAX_QUEUE:
            self.num_in += number
            self.level += number
            if self.toRecord:
                self.records.append([self.env.now,self.num_in,self.num_out])
        else:
            self.unserved.append([self.env.now, number])
    def record(self):
        # new_row=[self.env.now,self.num_in,self.num_out]
        # self.records = self.records.append(new_row, ignore_index=True)
        self.records.append([self.env.now,self.num_in,self.num_out])
    def get(self, number):
        if number !=0:
            self.container.get(number)
        self.num_out += number
        self.level -= number
        if self.toRecord:
            # self.event_out.append((self.env.now, self.num_in))
            if number <=1:
                # new_row=[self.env.now,self.num_in,self.num_out]
                # self.records = self.records.append(new_row, ignore_index=True)
                self.records.append([self.env.now,self.num_in,self.num_out])
            else:
                # lastTime = self.records['time'][len(self.records)-1]
                lastTime = self.records[-1][0]
                timeStep = (self.env.now - lastTime)/number
                for i in range(1,number+1):
                    # new_row=[lastTime+timeStep*i,self.num_in,self.num_out-number+i]
                    # self.records = self.records.append(new_row, ignore_index=True)
                    self.records.append([lastTime+timeStep*i,self.num_in,self.num_out-number+i])
        # print('%d %d %d'%(self.env.now,self.num_in,self.num_out))
        return self.level
    def get_event_records(self):
        return self.records
class bus_class:
    def __init__(self, env, ShuttleCapacity,InitialRouteName, vid,record=True, EnergyPerDistance=1.9):
        self.vehType = 'Normal bus'
        self.vid = vid
        self.RouteName=InitialRouteName
        self.container = simpy.Container(env, init=0, capacity=ShuttleCapacity)
        self.capacity = ShuttleCapacity
        self.distanceTraveled = 0
        self.env = env
        self.num_in = 0
        self.num_out = 0
        self.level = 0
        self.toRecord = record
        self.distanceRecordcolumns=['time','distance','from','to']
        self.energyRecordcolumns=['time','energy']
        self.recordscolumns=['time','num_in','num_out']
        self.routeRecordcolumns=['time','routeName']
        self.distanceRecord = []
        self.energyRecord = []
        self.records = []
        self.routeRecord = []
        self.energyConsumed=0
        self.energyPerDistance = EnergyPerDistance
    def recordRouteName(self, routeName):
        self.RouteName=routeName
        # new_row=[self.env.now,self.RouteName]
        # self.routeRecord = self.routeRecord.append(new_row, ignore_index=True)
        self.routeRecord.append([self.env.now,self.RouteName])
    def add_distance_traveled(self,distance,energy,FromStop,ToStop):
        self.distanceTraveled += distance
        # new_row=[self.env.now,self.distanceTraveled,FromStop,ToStop]
        # self.distanceRecord = self.distanceRecord.append(new_row, ignore_index=True)
        self.distanceRecord.append([self.env.now,self.distanceTraveled,FromStop,ToStop])
        if energy ==0:
            self.energyConsumed += distance*self.energyPerDistance
        else:
            self.energyConsumed += energy
        # new_row=[self.env.now,self.energyConsumed]
        # self.energyRecord = self.energyRecord.append(new_row, ignore_index=True)
        self.energyRecord.append([self.env.now,self.energyConsumed])
    def put(self, number):
        if number !=0:
            self.container.put(number)
        self.num_in += number
        self.level += number
        if self.toRecord:
            # new_row=[self.env.now,self.num_in,self.num_out]
            # self.records = self.records.append(new_row, ignore_index=True)
            self.records.append([self.env.now,self.num_in,self.num_out])
            # if number <=1:
            #     self.records.loc[len(self.records)]=[self.env.now,self.num_in,self.num_out]
            # else:
            #     lastTime = self.records['time'][len(self.records)-1]
            #     timeStep = (self.env.now - lastTime)/number
            #     for i in range(1,number+1):
            #         self.records.loc[len(self.records)]=[lastTime+timeStep*i,self.num_in-number+i,self.num_out]
    def get(self, number):
        if number!=0:
            self.container.get(number)
        self.num_out += number
        self.level -= number
        if self.toRecord:
            # new_row=[self.env.now,self.num_in,self.num_out]
            # self.records = self.records.append(new_row, ignore_index=True)
            self.records.append([self.env.now,self.num_in,self.num_out])
            # if number <=1:
            #     self.records.loc[len(self.records)]=[self.env.now,self.num_in,self.num_out]
            # else:
            #     lastTime = self.records['time'][len(self.records)-1]
            #     timeStep = (self.env.now - lastTime)/number
            #     for i in range(1,number+1):
            #         self.records.loc[len(self.records)]=[lastTime+timeStep*i,self.num_in,self.num_out-number+i]
#         print('%d %d %d'%(self.env.now,self.num_in,self.num_out))
    def get_event_records(self):
        return self.records
    def GetVehicleType(self):
        return self.vehType
EV_status={'runningEVNum':0,'charingEVNum':0,'parkingEVNum':0}
class EV_Bus_class(bus_class):
    def __init__(self, env, ShuttleCapacity,InitialRouteName,vid=0,toPrint=False, chargingStation=None, RandomBatteryPercentageLevel=(0.849,0.85), BatteryCapacity=660, dischargeRate=1.9, chargingRate = 132/3600, record=True):
        self.vehType = 'EV'
        self.vid = vid
        self.toPrint=toPrint
        self.RouteName=InitialRouteName
        self.ChargingStation = chargingStation
        self.container = simpy.Container(env, init=0, capacity=ShuttleCapacity)
        self.capacity = ShuttleCapacity
        self.distanceTraveled = 0
        self.env = env
        self.num_in = 0
        self.num_out = 0
        self.level = 0
        self.toRecord = record
        self.distanceRecordcolumns=['time','distance','from','to']
        self.recordscolumns=['time','num_in','num_out']
        self.routeRecordcolumns=['time','routeName']
        self.distanceRecord = []
        self.records = []
        self.routeRecord = []
        self.routeRecord.append([0,self.RouteName])

        self.upperPercent = 0.85

        self.BatteryRecordscolumns=['time','BatteryLevel']
        self.BatteryRecords = []
        self.dischargeRate = dischargeRate #

        self.BatteryLevel = BatteryCapacity*random.uniform(RandomBatteryPercentageLevel[0],RandomBatteryPercentageLevel[1])

        self.BatteryRecords.append([0,self.BatteryLevel])

        self.BatteryCapacity = BatteryCapacity
        self.chargingRate = chargingRate
    def add_distance_traveled(self,distance,energy,FromStop,ToStop):
        self.distanceTraveled += distance
        # new_row=[self.env.now,self.distanceTraveled]
        # self.distanceRecord = self.distanceRecord.append(new_row, ignore_index=True)
        self.distanceRecord.append([self.env.now,self.distanceTraveled,FromStop,ToStop])
        if energy ==0:
            self.BatteryLevel -= distance*self.dischargeRate
        else:
            self.BatteryLevel -= energy
        # new_row=[self.env.now,self.BatteryLevel]
        # self.BatteryRecords = self.BatteryRecords.append(new_row, ignore_index=True)
        self.BatteryRecords.append([self.env.now,self.BatteryLevel])
    def charging(self, ChargedTime):
        self.BatteryLevel += ChargedTime * self.chargingRate
        self.BatteryLevel = max(self.BatteryCapacity*self.upperPercent, self.BatteryLevel)
        # new_row=[self.env.now,self.BatteryLevel]
        # self.BatteryRecords = self.BatteryRecords.append(new_row, ignore_index=True)
        self.BatteryRecords.append([self.env.now,self.BatteryLevel])
    def chargeToFullPercentTime(self):
        global TimeToCharger
        if self.toPrint:
            print('Shuttle {0} start charing at {1}'.format(self.vid,self.env.now))
        # new_row=[self.env.now,self.BatteryLevel]
        # self.BatteryRecords = self.BatteryRecords.append(new_row, ignore_index=True)
        self.BatteryRecords.append([self.env.now,self.BatteryLevel])
        diff = self.BatteryCapacity*self.upperPercent - self.BatteryLevel
        return diff/self.chargingRate+TimeToCharger
    def chargeToFullPercent(self):
        self.BatteryLevel = self.BatteryCapacity*self.upperPercent
        if self.toPrint:
            print('Shuttle %d battery charged to %d'%(self.vid,int(self.BatteryLevel)))
        # new_row=[self.env.now,self.BatteryLevel]
        # self.BatteryRecords = self.BatteryRecords.append(new_row, ignore_index=True)
        self.BatteryRecords.append([self.env.now,self.BatteryLevel])
class onDemandBus_class(bus_class):
    def __init__(self, env, ShuttleCapacity, shuttleID,record=True,EnergyPerDistance=1.9):
        self.vehType = 'On demand bus'
        self.container = simpy.Container(env, init=0, capacity=ShuttleCapacity)
        self.capacity = ShuttleCapacity
        self.distanceTraveled = 0
        self.env = env
        self.num_in = 0
        self.num_out = 0
        self.level = 0
        self.toRecord = record
        self.distanceRecordcolumns=['time','distance','from','to']
        self.energyRecordcolumns=['time','energy']
        self.recordscolumns=['time','num_in','num_out']
        self.routeRecordcolumns=['time','routeName']
        self.distanceRecord = []
        self.energyRecord = []
        self.records = []
        self.routeRecord = []

        self.enroute= False
        self.stopList=[]
        self.shuttleID=10000+shuttleID
        self.energyConsumed = 0
        self.energyPerDistance = EnergyPerDistance
    def setRoute(self,stopList):
        self.stopList= stopList
    def run(self):
        global stops
        reactionTime = 20 #sec
        self.enroute=True
        # may have some issue here
        self.env.process(Pickup(self.env,stops,self,self.shuttleID,reactionTime,isHotShot=False,listStops=self.stopList,Looping=False))

class onDemandFleet_class:
    def __init__(self,env,capacities,record=True):
        self.buses={}
        self.busAnswers={}
        for i in range(len(capacities)):
            self.buses[i]=onDemandBus_class(env,capacities[i],i)
        for b in self.buses.values():
            self.busAnswers[b.shuttleID]=0
    def callOndemand(self,stopName):
        stopName=stopName.split(' ')[0]
        # print('stopName',stopName)
        route=[]
        if len(stopName)==1:
            route = ['R','R'+stopName[0],stopName[0]]
        else:
            sn = stopName[1]
            route = ['R','R'+sn,sn]
        # print('route',route)
        route = addStops2Route(route)
        # print('new route',route)
        #find an availabe bus
        ifFoundABus = False
        AlreadyABus = False
        for b in self.buses.values():
            if b.stopList==route and b.enroute:
                AlreadyABus=True
        if not AlreadyABus:
            for b in self.buses.values():
                if b.enroute == False:
                    ifFoundABus = True
                    # print('On demand %d is answering the call'%b.shuttleID)
                    self.busAnswers[b.shuttleID]+=1
                    b.setRoute(route)
                    b.run()
                    break
        # return ifFoundABus
        # print('found ondemand bus',ifFoundABus,AlreadyABus)
class dispatchCenter():
    def __init__(self):
        self.requestQueue=set()
    def request(self,stop):
        self.requestQueue.add(stop)
    def checkIsRequestQueued(self):
        if len(self.requestQueue)==0:
            return False
        else:
            return True
    def answer(self):
        stopAnswered = self.requestQueue.pop()
        return stopAnswered
class chargingStation():
    def __init__(self,env,numberOfChargers=3):
        self.env = env
        self.NumChargers=numberOfChargers
        self.OccupiedChargers = 0
        self.recordscolumns=['time','OccupiedChargers']
        self.records = []
    def request(self):
        if self.OccupiedChargers<self.NumChargers:
            self.records.append([self.env.now,self.OccupiedChargers])
            return True
        else:
            return False
    def enterStation(self):
        self.OccupiedChargers += 1
    def leaveStation(self):
        self.OccupiedChargers -=1
        # new_row=[self.env.now,self.OccupiedChargers]
        # self.records = self.records.append(new_row, ignore_index=True)
        self.records.append([self.env.now,self.OccupiedChargers])
#endregion
OppChargingAvailable = True
def Pickup(env, stops, shuttle, shuttleIndex, startTime,isHotShot=False,DayTimeOnDemand=False,dispatch=None,Looping=True,listStops=None,ToPrint=False,BatterLowLevel=0.2,chargingThreshold=0.0):
    '''
    The main process for shuttle operations
    '''
    global Day, EV_status, onRouteCharging, OppChargingAvailable
    yield env.timeout(startTime)
    stopNumber = 0

    if listStops ==None:
        currentRouteName = shuttle.RouteName
        termsInRoute=currentRouteName.split('_')[:-1]
        _capacity = currentRouteName.split('_')[-1]
        routes=addStops2Route(addRccStops2Route(termsInRoute))
    else:
        routes=listStops

    numStops = len(routes)
    peopleForStop={}

    tempRoute=routes

    IsShuttleFromRAC = False
    loopNumber = 1
    lastStopNumber = numStops
    # if Looping:
    #     lastStopNumber = numStops
    # else:
    #     lastStopNumber= numStops-1
    while stopNumber<lastStopNumber:
        stopNumber +=1
        if not Looping and stopNumber == 1 and loopNumber==2:
            break
        if stopNumber == numStops:
            loopNumber += 1
            tempRoute=routes
            numStops = len(routes)
            stopNumber = 0
            if isHotShot and shuttle.RouteName[2]!='P':# maybe an issue here
                # check dispatcher to see if new call is needed
                if dispatch.checkIsRequestQueued():
                    requestedStop=dispatch.answer()
                    # generate temp route
                    tempRoute=['R',requestedStop[0]]
                    tempRoute = addStops2Route(addRccStops2Route(tempRoute))
                    # update numStops
                    numStops=len(tempRoute)
                    lastStopNumber=numStops
                    hotshotRouteName = ''
                    for s in tempRoute:
                        hotshotRouteName+=s+'_'
                    shuttle.recordRouteName('Hotshot: '+hotshotRouteName)
                # if no call:
                    #resume normal route and numStops
            idleCharge=False
            skip=False
            if EV_status['parkingEVNum']>0 and shuttle.GetVehicleType()=='Normal bus' and currentRouteName.split('_')[1] ==  'P':
                skip=True
            if Looping and not skip: # not on demand. Get new route
                newRouteName = getNewRouteName(currentRouteName,env.now)
                if newRouteName == None:
                    print(env.now,currentRouteName,Day)
                    getNewRouteName(currentRouteName,env.now,toPrint=True)
                if newRouteName!=currentRouteName:
                    if currentRouteName.split('_')[1] ==  'P' and shuttle.GetVehicleType()=='EV':
                        EV_status['runningEVNum']+=1
                        EV_status['parkingEVNum']-=1
                        print(env.now,'at pick up parking to runing',shuttleIndex, EV_status)
                    if ToPrint:
                        print('Shuttle {} switch from {} to {}'.format(shuttleIndex,currentRouteName,newRouteName))
                    if newRouteName.split('_')[1] ==  'P' and shuttle.GetVehicleType()=='EV':
                        idleCharge = True
                        EV_status['parkingEVNum']+=1
                        EV_status['runningEVNum']-=1
                        print(env.now,'at pick up run to parking',shuttleIndex, EV_status)
                        if ToPrint:
                            print('Idle charge')
                    shuttle.recordRouteName(newRouteName)
                    termsInRoute = newRouteName.split('_')[:-1]
                    routes = addStops2Route(addRccStops2Route(termsInRoute))
                    tempRoute=routes
                    numStops = len(routes)
                    lastStopNumber=numStops
                    peopleForStop = {}
                    currentRouteName = newRouteName
            # charging for EV
            if shuttle.GetVehicleType()=='EV':
                toCharge = False
                wait=True
                toPrintIdleCharge = True
                inline=False
                while wait:
                    wait=False
                    if idleCharge:
                        if shuttle.ChargingStation.request():
                            if inline ==False:
                                EV_status['parkingEVNum']-=1
                                EV_status['charingEVNum']+=1
                            toCharge = True
                        else:
                            if inline==False:#into the queue
                                EV_status['parkingEVNum']-=1
                                EV_status['charingEVNum']+=1
                                inline=True
                            wait=True
                            yield env.timeout(90)
                    elif shuttle.BatteryLevel <= (BatterLowLevel+chargingThreshold)*shuttle.BatteryCapacity:
                        if shuttle.ChargingStation.request():
                            if inline==False:
                                EV_status['runningEVNum']-=1
                                EV_status['charingEVNum']+=1
                            toCharge = True
                        elif shuttle.BatteryLevel <= (BatterLowLevel)*shuttle.BatteryCapacity:
                            if inline==False:#into the queue
                                EV_status['runningEVNum']-=1
                                EV_status['charingEVNum']+=1
                                inline=True
                            wait=True
                            yield env.timeout(90)
                if toCharge:
                    print(env.now,'at pick up before charge',shuttleIndex, EV_status)
                    # print('Shuttle {0} will charge at {1}'.format(shuttleIndex,env.now))
                    dic_numBusInRoute[currentRouteName]-=1
                    shuttle.ChargingStation.enterStation()
                    newRouteName='R_P_'+_capacity
                    dic_numBusInRoute[newRouteName]+=1
                    shuttle.recordRouteName(newRouteName)
                    termsInRoute = newRouteName.split('_')[:-1]
                    routes = addStops2Route(addRccStops2Route(termsInRoute))
                    tempRoute=routes
                    numStops = len(routes)
                    lastStopNumber=numStops
                    peopleForStop = {}
                    currentRouteName = newRouteName
                    if ToPrint:
                        print('Number Chargers used {}'.format(shuttle.ChargingStation.OccupiedChargers))
                    yield env.timeout(shuttle.chargeToFullPercentTime())
                    shuttle.chargeToFullPercent()
                    yield env.timeout(5)
                    shuttle.ChargingStation.leaveStation()
                    EV_status['charingEVNum']-=1
                    EV_status['parkingEVNum']+=1
                    print(env.now,'at pick up after charge',shuttleIndex, EV_status)
                    if ToPrint:
                        print('Number Chargers used {}'.format(shuttle.ChargingStation.OccupiedChargers))
        if ToPrint:
            print('Shuttle %d reach %s at %d'%(shuttleIndex,tempRoute[stopNumber],env.now))
        stopName = tempRoute[stopNumber]
        shuttle.get(0)
        if stopName =='R':#rental car center drop off location
            # unload

            if ToPrint:
                print('drop off %d people'%shuttle.level)
            if shuttle.level>0:
                peopleOut = shuttle.level
                # yield env.timeout(peopleOut*AvgUnloadPersonTime)
                shuttle.get(shuttle.level)
            #     else:
            #         # yield env.timeout(AvgTime_RAC_DropOff_Pickup)#travel time from the RAC drop off to RAC pickup
            # else:
                # env.timeout(AvgTime_RAC_DropOff_Pickup)#travel time from the RAC drop off to RAC pickup
            # update route

        elif stopName[0] == 'R':#rental car center pickup location
            # load
            while stops[stopName]._lock:
                yield env.timeout(5)
            stops[stopName].lock()
            stops[stopName].get(0)
            peopleLoaded = min(shuttle.capacity - shuttle.level,stops[stopName].level)
            peopleForStop[stopName[1:]] = peopleLoaded
            if peopleLoaded > 0:
                # yield env.timeout(peopleLoaded*AvgLoadPersonTime)
                shuttle.put(peopleLoaded)
                stops[stopName].get(peopleLoaded)
                if ToPrint:
                    print('Shuttle %d loaded %d people'%(shuttleIndex,peopleLoaded))
            elif shuttle.capacity == shuttle.level:
                if ToPrint:
                    print('bus is full at %s'%stopName)
            elif stops[stopName].level == 0:
                if ToPrint:
                    print('no one was at %s'%stopName)
            stops[stopName].recordPeopleLeft()
            stops[stopName].unlock()
        else:#terminal
            if ToPrint:
                print('Shuttle %d reach terminal %s at %d'%(shuttleIndex, stopName, env.now))
            shuttle.get(0)
            #unload
            if peopleForStop[stopName] >0:
                # yield env.timeout(peopleForStop[stopName]*AvgUnloadPersonTime)
                shuttle.get(peopleForStop[stopName])
                if ToPrint:
                    print('%d people was dropped off at terminal %s'%(peopleForStop[stopName],tempRoute[stopNumber]))
            else:
                if ToPrint:
                    print('no one gets out')
            peopleForStop[stopName] = -1

            #load
            while stops[stopName]._lock:
                yield env.timeout(5)
            stops[stopName].lock()
            emptySeats = shuttle.capacity - shuttle.level
            stops[stopName].get(0)
            if emptySeats >0:
                if stops[stopName].level==0:
                    waitTime = random.randint(50,70)
                    # yield env.timeout(waitTime)
                    shuttle.get(0)
                if stops[stopName].level>0:
                    peopleLoaded = min(emptySeats,stops[stopName].level)
                    # yield env.timeout(peopleLoaded*AvgLoadPersonTime)
                    stops[stopName].get(peopleLoaded)
                    shuttle.put(peopleLoaded)
                    if ToPrint:
                        print('%d people was picked up at time: %d'%(peopleLoaded, env.now))
                        print('people left at terminal %s: %d'%(stopName,stops[stopName].level))
                        print('people in shuttle %d: '%shuttleIndex+str(shuttle.level))
                else:
                    if ToPrint:
                        print('no one at the stop to pick up')
            else:
                if ToPrint:
                    print('Shuttle is full')
            stops[stopName].recordPeopleLeft()
            stops[stopName].unlock()
            if stops[stopName].level>0:
                if ToPrint:
                    print('Stop '+stopName+' calling on demand buses')
                if DayTimeOnDemand:
                    stops[stopName].onDemandBuses.callOndemand(stopName)
                if dispatch != None:
                    dispatch.request(stopName)
        thisStop = stopName
        nextStop = tempRoute[(stopNumber+1)%numStops]

        (Distance, travelTime, Energy, dwell) = getDistanceTravelTimeEnergyDwellT(thisStop,nextStop,Day,env.now)
        # only charge at rental car center
        if onRouteCharging and shuttle.GetVehicleType()=='EV' and thisStop=='R' and OppChargingAvailable:
            shuttle.charging(travelTime)
            OppChargingAvailable = False
            yield env.timeout(travelTime)
            OppChargingAvailable=True
        else:
            yield env.timeout(travelTime)
        shuttle.add_distance_traveled(Distance,Energy,thisStop,nextStop)
        # if thisStop != 'R' and len(thisStop+nextStop)!=4:
        #     travelLink=thisStop[0]+nextStop
        #     shuttle.add_distance_traveled(get_distance(travelLink))
    shuttle.enroute=False
    # return True
stops = {}
def experiment(rndSeed,shuttleSpacing,recordShuttles=True,onDemandCap=14,IfRecordStops=True,doHotShot = True,dayTimeOnDemand=False,NightTimeOndemand=True,NightTimeOndemandInterval=60*15,numEV=10, NumOnDemand=3,numEVCharger=3,simTime=1,BatteryCapacity=660,chargingRate=180/3600,chargingThreshold=0.0):
    '''
    start a simulation experiment with certain parameters
    '''
    global listStops,terminalArrivals,RAC_Arrivals, stops, dic_numBusInRoute, EV_status

    RACStops = ['RA','RB','RC','RD','RE','RP']
    termStops = ['A','B','C','D','E','P']
    listStops=['RA','RB','RC','RD','RE','RP','A','B','C','D','E','P']

    RACStops = addStops2Route(RACStops)
    termStops = addStops2Route(termStops)
    listStops = addStops2Route(listStops)

    numTerminals = len(termStops)

    random.seed(rndSeed)
    env = simpy.Environment(initial_time=0)
    env.process(nextDay(env))
    # onDemandBuses = []
    #on demand buses
    ondemandCapacities = [onDemandCap for i in range(NumOnDemand)]
    onDemandFleet = onDemandFleet_class(env,ondemandCapacities,record=True)

    # stops and passenger arrivals

    for stop in listStops:
        stops[stop]= stop_class(env,simpy.Container(env,init=0),record=IfRecordStops,onDemandBuses=onDemandFleet)
        if stop != 'P' and stop !='RP':
            env.process(TimeDependentPassengerArrival(env,stops[stop], stop))
            if NightTimeOndemand:
                env.process(ondemandAtNight(env,stops,stop,checkingInterval=NightTimeOndemandInterval))
    terminals = [stops[termStop] for termStop in termStops]
    rentalCarCenters = [stops[RACStop] for RACStop in RACStops]

    #ini buses
    shuttles=[]
    countVeh=0
    dispatch = dispatchCenter()
    _chargingStation = chargingStation(env,numberOfChargers=numEVCharger)
    for (r,n) in dic_numBusInRoute.items():
        cap = int(r.split('_')[-1])
        if n !=0:
            shuttleSpacing = int(60*60/n)
        else:
            shuttleSpacing = 1800
        for i in range(n):
            if countVeh<numEV:
                if r.split('_')[1] ==  'P':
                    EV_status['parkingEVNum']+=1
                else:
                    EV_status['runningEVNum']+=1
                shuttles.append(EV_Bus_class(env, cap, r,vid=countVeh,chargingStation=_chargingStation, record=recordShuttles,BatteryCapacity=BatteryCapacity,chargingRate=chargingRate))
            else:
                shuttles.append(bus_class(env, cap,r,vid=countVeh, record=recordShuttles))
            env.process(Pickup(env,stops,shuttles[countVeh],countVeh,i*shuttleSpacing+1,DayTimeOnDemand=dayTimeOnDemand,isHotShot=doHotShot,dispatch=dispatch,Looping=True,ToPrint=False,chargingThreshold=chargingThreshold))
            countVeh+=1
    print('at experiment', EV_status)
    env.run(until=3600*24*simTime-1)
    return [shuttles,onDemandFleet, stops,_chargingStation]

shuttleSpacing = 7*60

startTime= time.time()
# numOndemand = 10
[shuttles,onDemandFleet, stops,_chargingStation]=experiment(rndSeed,shuttleSpacing, NightTimeOndemand = ifDoNightTimeOndemand,onDemandCap=OnDemandCapacity,NightTimeOndemandInterval=NightTimeOndemandCheckInterval,dayTimeOnDemand=ifDoRegularOndemand,recordShuttles=True,doHotShot = doHotShot,IfRecordStops=True,numEV=numEV, NumOnDemand=numOndemand,numEVCharger=numEVCharger,simTime=simTime)
endTime=time.time()
deltaTime=endTime-startTime
print('Time used: {}'.format(deltaTime))

EVs=[]
normalShuttles=[]
onDemandV=[]
recordStops=[]
for v in shuttles:
    if v.vehType=='EV':
        thisv={}
        thisv['vid']=v.vid
        thisv['capacity']=v.capacity
        thisv['pax']=pd.DataFrame(v.records,columns=v.recordscolumns)
        thisv['battery']=pd.DataFrame(v.BatteryRecords,columns=v.BatteryRecordscolumns)
        thisv['routes']=pd.DataFrame(v.routeRecord,columns=v.routeRecordcolumns)
        thisv['distance']=pd.DataFrame(v.distanceRecord,columns=v.distanceRecordcolumns)
        EVs.append(thisv)
    else:
        thisv={}
        thisv['vid']=v.vid
        thisv['capacity']=v.capacity
        thisv['pax']=pd.DataFrame(v.records,columns=v.recordscolumns)
        thisv['battery']=pd.DataFrame(v.energyRecord,columns=v.energyRecordcolumns)
        thisv['routes']=pd.DataFrame(v.routeRecord,columns=v.routeRecordcolumns)
        thisv['distance']=pd.DataFrame(v.distanceRecord,columns=v.distanceRecordcolumns)
        normalShuttles.append(thisv)
for v in onDemandFleet.buses.values():
    thisv={}
    thisv['vid']=v.shuttleID
    thisv['capacity']=v.capacity
    thisv['pax']=pd.DataFrame(v.records,columns=v.recordscolumns)
    thisv['battery']=pd.DataFrame(v.energyRecord,columns=v.energyRecordcolumns)
    thisv['routes']=pd.DataFrame(v.routeRecord,columns=v.routeRecordcolumns)
    thisv['distance']=pd.DataFrame(v.distanceRecord,columns=v.distanceRecordcolumns)
    onDemandV.append(thisv)
for sid, value in stops.items():
    thisv={}
    thisv['stopName']=sid
    thisv['PaxRecord']=pd.DataFrame(value.records,columns=value.recordscolumns)
    thisv['unserved']=pd.DataFrame(value.unserved,columns=value.unservedcolumns)

    recordStops.append(thisv)
results={}
results['EVShuttle']=EVs
results['normalShuttle']=normalShuttles
results['OndemandShuttle']=onDemandV
results['Stops']=recordStops
results['ChargingStation']=_chargingStation.records
f = open('{}_results.pckl'.format(outputName), 'wb')
pickle.dump(results, f)
f.close()
endTime=time.time()
deltaTime=endTime-startTime
print('Time used + save: {}'.format(deltaTime))
print(len(shuttles))
