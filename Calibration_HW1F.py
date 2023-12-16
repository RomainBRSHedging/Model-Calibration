from QuantLib import *
import QuantLib as ql
from collections import namedtuple
import math
import numpy as np


today = Date(31, December, 2022)
settlement = today
Settings.instance().evaluationDate = today
dayCounter = Actual360()
dates= np.array([Date(30,12,2022),
        Date(3,7,2023),
        Date(3,8,2023),
        Date(4,9,2023),
        Date(3,10,2023),
        Date(3,11,2023),
        Date(4,12,2023),
        Date(3,1,2024),
        Date(3,4,2024),
        Date(3,7,2024),
        Date(3,1,2025),
        Date(5,1,2026),
        Date(4,1,2027),
        Date(5,1,2028),
        Date(3,1,2029), 
        Date(3,1,2030),
        Date(3,1,2031)])
dfs =  np.array([1,
        0.986347871160595,
        0.982509140361925,
        0.978944855619828,
        0.976070800832639,
        0.973346919208213,
        0.970594977778527,
        0.968249987887003,
        0.958147063105479,
        0.951027825004006,
        0.935088886907188,
        0.906607254399848,
        0.87941101708969,
        0.852513048887646,
        0.826955512793027,
        0.802250520258904,
        0.777718328397543])
yts = DiscountCurve(dates, dfs, dayCounter)
yts.enableExtrapolation()
term_structure = YieldTermStructureHandle(yts)
# term_structure = ql.YieldTermStructureHandle(FlatForward(settlement,3.9,ql.Actual365Fixed()))
index = Euribor6M(term_structure)

# Starting tenor (exercize date in months), Months to maturity (Swap), Swaption Volatilities
CalibrationData = namedtuple("CalibrationData", "start, length, volatility")


data = [
        CalibrationData(1, 8, 0.122),
        CalibrationData(2, 7, 0.13),
        CalibrationData(3, 6, 0.123),
        CalibrationData(4, 5, 0.125),
        CalibrationData(5, 4, 0.13),
        CalibrationData(6, 3,0.119),
        CalibrationData(7, 2, 0.11),
        CalibrationData(8, 1, 0.099),
        ]

def create_swaption_helpers(data, index, term_structure, engine):
    swaptions = []
    fixed_leg_tenor = ql.Period(1, ql.Years)
    fixed_leg_daycounter = ql.Actual360()
    floating_leg_daycounter = ql.Actual360()
    for d in data:
        vol_handle = QuoteHandle(SimpleQuote(d.volatility))
        helper = SwaptionHelper(Period(d.start, Years),
                                   Period(d.length, Years),
                                   vol_handle,
                                   index,
                                   fixed_leg_tenor,
                                   fixed_leg_daycounter,
                                   floating_leg_daycounter,
                                   term_structure
                                   )
        helper.setPricingEngine(engine)
        swaptions.append(helper)
    return swaptions    

def calibration_report(swaptions, data):
    print("-"*82)
    print("%15s %15s %15s %15s %15s" % \
    ("Model Price", "Market Price", "Implied Vol", "Market Vol", "Rel Error"))
    print("-"*82)
    cum_err = 0.0
    for i, s in enumerate(swaptions):
        model_price = s.modelValue()
        market_vol = data[i].volatility
        black_price = s.blackPrice(market_vol)
        rel_error = model_price/black_price - 1.0
        implied_vol = s.impliedVolatility(model_price,
                                          1e-5, 50, 0.0, 0.50)
        rel_error2 = implied_vol/market_vol-1.0
        cum_err += rel_error2*rel_error2
        
        print("%15.5f %15.5f %15.5f %15.5f %15.5f" % \
        (model_price, black_price, implied_vol, market_vol, rel_error))
    print("-"*82)
    print("Cumulative Error : %15.5f" % math.sqrt(cum_err))
   

model = HullWhite(term_structure)
engine = JamshidianSwaptionEngine(model)
swaptions = create_swaption_helpers(data, index, term_structure, engine)

optimization_method = LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
end_criteria = ql.EndCriteria(10000, 100, 1e-6, 1e-8, 1e-8)
model.calibrate(swaptions, optimization_method, end_criteria)

a, sigma = model.params()
print("a = %6.5f, sigma = %6.5f" % (a, sigma))
calibration_report(swaptions, data)

model = HullWhite(term_structure, a, sigma)
engine = JamshidianSwaptionEngine(model)
swaptions = create_swaption_helpers(data, index, term_structure, engine)


def makeSwap(start, maturity, nominal, fixedRate, index, typ=ql.VanillaSwap.Payer):

    end = ql.TARGET().advance(start, maturity)
    fixedLegTenor = ql.Period(3,ql.Months)
    fixedLegBDC = ql.ModifiedFollowing
    fixedLegDC = ql.Thirty360(ql.Thirty360.BondBasis)
    # fixedLegDC = ql.Actual365Fixed()

    spread = 0.0
    fixedSchedule = ql.Schedule(start,
                                end, 
                                fixedLegTenor, 
                                index.fixingCalendar(), 
                                fixedLegBDC,
                                fixedLegBDC, 
                                ql.DateGeneration.Backward,
                                False)
    floatSchedule = ql.Schedule(start,
                                end,
                                index.tenor(),
                                index.fixingCalendar(),
                                index.businessDayConvention(),
                                index.businessDayConvention(),
                                ql.DateGeneration.Backward,
                                False)
    swap = ql.VanillaSwap(typ, 
                          nominal,
                          fixedSchedule,
                          fixedRate,
                          fixedLegDC,
                          floatSchedule,
                          index,
                          spread,
                          index.dayCounter())
    return swap, [index.fixingDate(x) for x in floatSchedule][:-1]

def makeSwaption(swap, callDates, settlement):
    if len(callDates) == 1:
        exercise = ql.EuropeanExercise(callDates[0])
    else:
        exercise = ql.BermudanExercise(callDates)
    return ql.Swaption(swap, exercise, settlement)


settlementDate = today + ql.Period("2D")
swaps = [makeSwap(settlementDate,ql.Period("1Y"),1e5,0.03,index)]
calldates = [index.fixingDate(ql.Date(28,2,2023))]
swaptions = [makeSwaption(swap, calldates, ql.Settlement.Physical) for swap, fd in swaps]

for swaption in swaptions:
    swaption.setPricingEngine(engine)
    print("Swaption NPV  : %.2f" % swaption.NPV())