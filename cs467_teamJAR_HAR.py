###########################################################################################################################
# Course:    CS467 - Winter, 2022
# Author:    John Carlton (carltjoh) - Team JAR
# Date:      March 7, 2022
#
#
#                        Heterogeneous Auto Regression (HAR) Model for Projecting Volatility
#
# Following is an implementation of a HAR model for projecting volatility.  It may be used in one of two modes, to be
# selected by the PREDICT_MODE value defined below.
#
# PREDICT - If PREDICT_MODE is 'True', then the code will train a HAR model to predict volatility for all the assets
# included in the pool of candidate investments and make a prediction of the expected volatility for the upcoming period.
# The intervals to be considered are defined in the "lagProfile" variable, in which the first term represents that number
# of days considered to comprise one interval.  The second variable is a list specifying the number of previously defined
# intervals which will be used for lag inputs.  In this mode, the application will compare all securities included in the 
# 'predictSecurities' list for the months included in the 'predictMonths' list of months in 2021, then provide performance
# metrics on the selections.
#
# TEST - When PREDICT_MODE is 'False', the code will train a model based on all data available between 2017 and 2021, then
# iteratively test it with a single year from that period while the remaining data is used for training.  Results are
# tabulated and the summarized in a high level table.  This method provides a more complete view on the behavior of the
# models.
#
# In each case, data is pulled from files in the 'data/' subdirectory of the current directory.
# 
# We chose an approach similar to the following implementation, although our project diverged as we developed additional
# features not included in the reference project below.
# https://github.com/nathansimonis1612/HAR-RV/blob/master/HAR-RV_forecast.ipynb
###########################################################################################################################

import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
#from statsmodels.tsa.stattools import adfuller as adf
#import statistics

PREDICT_MODE = True                     # PREDICT mode or TEST mode
lagProfile = [22, [1, 3, 12]]           # Interval in days, lags in intervals

# Summary of all test scenarios to be used in TEST mode
TEST_SUITES = [["CURE", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["DRN", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["DUSL", 2019, 2021, lagProfile[0], lagProfile[1]],
               ["FAS", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["MIDU", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["NAIL", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["SOXL", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["SPXL", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["TECL", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["TNA", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["TQQQ", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["UBOT", 2020, 2021, lagProfile[0], lagProfile[1]],
               ["UDOW", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["UMDD", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["UPRO", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["URTY", 2017, 2021, lagProfile[0], lagProfile[1]],
               ["UTSL", 2019, 2021, lagProfile[0], lagProfile[1]],
               ["WANT", 2020, 2021, lagProfile[0], lagProfile[1]]]


TRAIN_START = "2017-01-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2021-12-31"
PREDICT_DATE = "2022-02-24"


data_path = "data/"                                         # Path to folder with input files
file_suffix = "_ohlc.csv"                                   # Standard filename is 'ticker symbol' + this suffix
security_list_source = "security_list.txt"                  # List of security ticker symbols to be considered
COL_DATETIME = 0                                            # Column numbers (0 thru 6) for corresponding data fields
COL_OPEN = 1
COL_HIGH = 2
COL_LOW = 3
COL_CLOSE = 4
COL_VOLUME = 5
COL_TRADE_COUNT = 6
COL_VWAP = 7
VOLATILITY = "realized_volatility"

###########################################################################################################################
# Used to capture the attributes and methods required for analysis of a single security.
###########################################################################################################################
class Asset:
    def __init__(self, name):
        self.name = name


###########################################################################################################################
# Load data from .csv input files stored in the data/ subdirectory of the current directory.
###########################################################################################################################
    def loadData(self):
        fileIn = open(data_path + self.name + file_suffix, "r")
        inputData = []
        
        for i, line in enumerate(fileIn):
            currentValues = line.split(",")
            currentRecord = []

            for field in currentValues:
                currentRecord.append(field.rstrip())

            if i == 0:
                self.columnLabels = currentRecord
            else:
                inputData.append(currentRecord)
    
        fileIn.close()
        self.csvData = np.array(inputData)


###########################################################################################################################
# Sorts the .csv file based on a defined column
###########################################################################################################################
    def sortCsvData(self, column):
        self.csvData = self.csvData[self.csvData[:, column].argsort()]


###########################################################################################################################
# Remove records with low trading volume so they don't distort the volatility calculations.
###########################################################################################################################
    def cleanData(self, threshold):
        averageVolume = np.mean(self.csvData[:, COL_VOLUME].astype('float'))

        currentRecord = len(self.csvData) - 1

        tempRow = []
        for element in self.csvData[0]:
            tempRow.append(element)

        tempArray = []
        tempArray.append(tempRow)

        for i in range (1, len(self.csvData)):
            tempRow = []

            if (self.csvData[i][COL_VOLUME].astype('float')) > (threshold * averageVolume):
                for element in self.csvData[i]:
                    tempRow.append(element)

                tempArray.append(tempRow)


###########################################################################################################################
# Aggregate the 5-minute date into intervals of the number of days specified by the 'daysUnitPeriod' parameter.  Adds
# volatility to each of these intervals based on the 5-minute data aggregated.
###########################################################################################################################
    def aggregateTableToPeriods(self, daysUnitPeriod):
        labelRow = ["Datetime_start", "Datetime_end", "Open", "High", "Low", "Close", "Volume", "trade_count", "vwap", "active_intervals", VOLATILITY]
        self.aggregatedTable = []
        self.aggregatedTable.append(labelRow)
        currentRecord = len(self.csvData) - 1
        previousDate = self.csvData[currentRecord][COL_DATETIME]
        endSlice = currentRecord
        dayCount = 1

        while currentRecord >= 0:
            if self.csvData[currentRecord][COL_DATETIME][0:10] != previousDate[0:10]:
                if dayCount >= daysUnitPeriod:
                    newRecord = [self.csvData[currentRecord + 1][COL_DATETIME]]
                    newRecord.append(self.csvData[endSlice][COL_DATETIME])
                    newRecord.append(float(self.csvData[currentRecord + 1][COL_OPEN]))
                    newRecord.append(float(max(self.csvData[currentRecord + 1: endSlice + 1, COL_HIGH])))
                    newRecord.append(float(min(self.csvData[currentRecord + 1: endSlice + 1, COL_LOW])))
                    newRecord.append(float(self.csvData[endSlice][COL_CLOSE]))
                    newRecord.append(sum(self.csvData[currentRecord + 1: endSlice + 1, COL_VOLUME].astype('float')))
                    newRecord.append(sum(self.csvData[currentRecord + 1: endSlice + 1, COL_TRADE_COUNT].astype('int')))

                    dollarsTransacted = 0
                    for i in range (currentRecord + 1, endSlice + 1):
                        dollarsTransacted += (float(self.csvData[i][COL_VOLUME]) * float(self.csvData[i][COL_VWAP]))
                    newRecord.append(dollarsTransacted / sum(self.csvData[currentRecord + 1: endSlice + 1, COL_VOLUME].astype('float')))

                    newRecord.append(endSlice - currentRecord - 1)
                    # Add realized volatility
                    rvCount = 0
                    for i in range (currentRecord + 1, endSlice):
                        rvCount += ((float(self.csvData[i + 1][COL_CLOSE]) - float(self.csvData[i][COL_CLOSE])) ** 2)
                    if endSlice - currentRecord > 1:
                        newRecord.append((rvCount / (endSlice - currentRecord - 1)) ** 0.5)
                    else:
                        newRecord.append(0)

                    self.aggregatedTable.append(newRecord)
                    dayCount = 0
                    endSlice = currentRecord

                dayCount += 1
                previousDate = self.csvData[currentRecord][COL_DATETIME]

            currentRecord -= 1


###########################################################################################################################
# Add lag values for volatility over periods comprised of the number of intervals specified in the lagVauleList, provided
# as a parameter.
###########################################################################################################################
    def addLags(self, lagValueList):
        numberOfColumns = len(self.aggregatedTable[0])

        for i, lagValue in enumerate(lagValueList):
            self.aggregatedTable[0].append("_rv_" + str(i))

            for j in range (1, len(self.aggregatedTable) - lagValue):
                rollingSum = 0

                for k in range (j + 1, j + lagValue + 1):
                    rollingSum += self.aggregatedTable[k][numberOfColumns - 1]

                self.aggregatedTable[j].append(rollingSum / float(lagValue))

            for j in range (len(self.aggregatedTable) - lagValue, len(self.aggregatedTable)):
                self.aggregatedTable[j].append(0)

        tableRows = len(self.aggregatedTable)
        for i in range(tableRows - 1, tableRows - max(lagValueList), -1):
            self.aggregatedTable = np.delete(self.aggregatedTable, i, 0)


###########################################################################################################################
# Print selected rows from the .csv file.
###########################################################################################################################
    def printCsvTable(self, firstRecord, lastRecord):
        print(self.columnLabels)

        for i in range (firstRecord, lastRecord):
            print(self.csvData[i])


###########################################################################################################################
# Print contents of the table with data aggregated over longer intervals.
###########################################################################################################################
    def printAggregatedTable(self):
        for i in range (len(self.aggregatedTable)):
            print(self.aggregatedTable[i])


###########################################################################################################################
# Find the slice boundaries in the aggregated tablet to provide data for training, test or other as required.
###########################################################################################################################
    def getBoundaries(self, startDate, endDate):
        ptrNext = 1
        if self.aggregatedTable[len(self.aggregatedTable) - 1][0][0:10] > endDate:
            ptrEnd = len(self.aggregatedTable) - 1
        else:
            while self.aggregatedTable[ptrNext][0][0:10] > endDate:
                ptrNext += 1
            ptrEnd = ptrNext

        ptrNext = 1
        if self.aggregatedTable[len(self.aggregatedTable) - 1][0][0:10] >= startDate:
            ptrStart = len(self.aggregatedTable) - 1
        else:
            while self.aggregatedTable[ptrNext][0][0:10] >= startDate:
                ptrNext += 1
            ptrStart = ptrNext - 1

        ptrLeft = 0
        ptrRight = 0
        for i in range(len(self.aggregatedTable[0])):
            if self.aggregatedTable[0][i][0:1] == "_":
                ptrRight = i

                if ptrLeft == 0:
                    ptrLeft = i

        boundaries = {"start": ptrStart, "end": ptrEnd, "left": ptrLeft, "right": ptrRight}
        return boundaries


###########################################################################################################################
# Perform OLS HAR regression and print critical metrics
###########################################################################################################################
    def performRegression(self, startDate, endDate, startTest, endTest):
        bounds = self.getBoundaries(startDate, endDate)
        print()
        print("bounds     : ", bounds)
        X = self.aggregatedTable[bounds["end"]:bounds["start"], bounds["left"]:bounds["right"] + 1].astype('float')
        
        for i in range (len(self.aggregatedTable[0])):
            if self.aggregatedTable[0][i] == VOLATILITY:
                targetColumn = i

        Y = self.aggregatedTable[bounds["end"]:bounds["start"], targetColumn].astype('float')

        reg = LinearRegression().fit(X, Y)
        print("Coef:    ", reg.coef_)
        print("Intercept", reg.intercept_)
        
        bounds = self.getBoundaries(startTest, endTest)
        X_test = self.aggregatedTable[bounds["end"]:bounds["start"], bounds["left"]:bounds["right"] + 1].astype('float')
        Y_test = self.aggregatedTable[bounds["end"]:bounds["start"], targetColumn].astype('float')
        #print("Dims", X_test.shape(), Y_test.shape())
        print("Score:   ", reg.score(X_test, Y_test))


###########################################################################################################################
# Uses data from firstYear to lastYear to train and test a model.  Iteratively uses each year as test data, while using
# the remainder for training.  Outputs the results to a tabular format.
###########################################################################################################################
    def testByYear(self, firstYear, lastYear, targetYear):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for i in range (len(self.aggregatedTable[0])):
            if self.aggregatedTable[0][i] == VOLATILITY:
                targetColumn = i        

        for year in range(firstYear, lastYear + 1):
            bounds = self.getBoundaries(str(year) + "-01-01", str(year) + "-12-31")

            if year == targetYear:
                X_test = self.aggregatedTable[bounds["end"]:bounds["start"], bounds["left"]:bounds["right"] + 1].astype('float')
                Y_test = self.aggregatedTable[bounds["end"]:bounds["start"], targetColumn].astype('float')
            elif len(X_train) == 0:
                X_train = self.aggregatedTable[bounds["end"]:bounds["start"], bounds["left"]:bounds["right"] + 1].astype('float')
                Y_train = self.aggregatedTable[bounds["end"]:bounds["start"], targetColumn].astype('float')
            else:
                X_train = np.append(X_train, self.aggregatedTable[bounds["end"]:bounds["start"], bounds["left"]:bounds["right"] + 1].astype('float'), axis = 0)
                Y_train = np.append(Y_train, self.aggregatedTable[bounds["end"]:bounds["start"], targetColumn].astype('float'), axis = 0)
                
        print()

        
        reg = LinearRegression().fit(X_train, Y_train)
        return {"target": targetYear, "tests": len(Y_test), "training": len(Y_train), "coefficients": reg.coef_, "intercept": reg.intercept_, "score": reg.score(X_test, Y_test)}


###########################################################################################################################
# Performs data aggregation from 5-minute data to intervals of a specified number of days for selection of preferred
# security which is expected to have high volatility in the upcoming period.
###########################################################################################################################
    def aggregateForProjection(self, interval, projectionDate):
        labelRow = ["Datetime_start", "Datetime_end", "Open", "High", "Low", "Close", "Volume", "trade_count", "vwap", "active_intervals", VOLATILITY]
        self.aggregatedTable = []
        self.aggregatedTable.append(labelRow)
        currentRecord = len(self.csvData) - 1

        while self.csvData[currentRecord][COL_DATETIME][0:10] > projectionDate:
            currentRecord -= 1

        previousDate = self.csvData[currentRecord][COL_DATETIME]
        endSlice = currentRecord
        dayCount = 1

        while currentRecord >= 0:
            if self.csvData[currentRecord][COL_DATETIME][0:10] != previousDate[0:10]:
                if dayCount >= interval:
                    newRecord = [self.csvData[currentRecord + 1][COL_DATETIME]]
                    newRecord.append(self.csvData[endSlice][COL_DATETIME])
                    newRecord.append(float(self.csvData[currentRecord + 1][COL_OPEN]))
                    newRecord.append(float(max(self.csvData[currentRecord + 1: endSlice + 1, COL_HIGH])))
                    newRecord.append(float(min(self.csvData[currentRecord + 1: endSlice + 1, COL_LOW])))
                    newRecord.append(float(self.csvData[endSlice][COL_CLOSE]))
                    newRecord.append(sum(self.csvData[currentRecord + 1: endSlice + 1, COL_VOLUME].astype('float')))
                    newRecord.append(sum(self.csvData[currentRecord + 1: endSlice + 1, COL_TRADE_COUNT].astype('int')))

                    dollarsTransacted = 0
                    for i in range (currentRecord + 1, endSlice + 1):
                        dollarsTransacted += (float(self.csvData[i][COL_VOLUME]) * float(self.csvData[i][COL_VWAP]))
                    newRecord.append(dollarsTransacted / sum(self.csvData[currentRecord + 1: endSlice + 1, COL_VOLUME].astype('float')))

                    newRecord.append(endSlice - currentRecord - 1)
                    
                    rvCount = 0
                    for i in range (currentRecord + 1, endSlice):
                        rvCount += ((float(self.csvData[i + 1][COL_CLOSE]) - float(self.csvData[i][COL_CLOSE])) ** 2)
                    if endSlice - currentRecord > 1:
                        newRecord.append((rvCount / (endSlice - currentRecord - 1)) ** 0.5)
                    else:
                        newRecord.append(0)

                    self.aggregatedTable.append(newRecord)
                    dayCount = 0
                    endSlice = currentRecord

                dayCount += 1
                previousDate = self.csvData[currentRecord][COL_DATETIME]

            currentRecord -= 1


###########################################################################################################################
# Defines parameters for a single test scenario and provides methods to implement.
###########################################################################################################################
class Scenario:
    def __init__(self, ticker, startYear, endYear, testYear, interval, lags):
        self.ticker = ticker
        self.startYear = startYear
        self.endYear = endYear
        self.testYear = testYear
        self.interval = interval
        self.lags = lags


###########################################################################################################################
# Coordinate a test based on attributes already defined
###########################################################################################################################
    def runTests(self):
        currentSecurity = Asset(self.ticker)
        currentSecurity.loadData()
        currentSecurity.sortCsvData(COL_DATETIME)
        currentSecurity.cleanData(0.1)
        currentSecurity.aggregateTableToPeriods(self.interval)
        currentSecurity.addLags(self.lags)
        self.results = currentSecurity.testByYear(self.startYear, self.endYear, self.testYear)
        return self.results


###########################################################################################################################
# Printe header for tabular output
###########################################################################################################################
    def printHeader(self):
        for i in range (4):
            print()

        headers = ["Target Year", "Lag 1", "Lag 2", "Lag 3", "Intercept", "Score"]
        strLags = ""
        for value in self.lags:
            if len(strLags) == 0:
                strLags = str(value)
            else:
                strLags = strLags + ", " + str(value)

        print("Security:    ", self.ticker)
        print("Date range:  ", str(self.startYear) + " thru " + str(self.endYear))
        print("Interval:    ", str(self.interval) + " day(s)")
        print("Lags:        ", strLags + " intervals")
        print()
        
        for element in headers:
            print(f"{element: ^12}", end=" ")
        print("\n")
        

###########################################################################################################################
# Print results of a single test run in tabular format
###########################################################################################################################
    def printResults(self):
        output = [str(self.results["target"]), "{:.5f}".format(self.results["coefficients"][0]), "{:.5f}".format(self.results["coefficients"][1]),
                  "{:.5f}".format(self.results["coefficients"][2]), "{:.5f}".format(self.results["intercept"]), "{:.5f}".format(self.results["score"])]

        for element in output:
            print(f"{element: ^12}", end=" ")
        print("\n")


###########################################################################################################################
# Defines parameters for a suite of tests and provides methods to implement.
###########################################################################################################################
class TestSuite:
    def __init__(self, ticker, startYear, endYear, interval, lags):
        self.ticker = ticker
        self.startYear = startYear
        self.endYear = endYear
        self.interval = interval
        self.lags = lags
        self.sumOfCoefficients = [0, 0, 0]
        self.sumOfIntercepts = 0
        self.sumOfScores = 0
        self.scenarios = 0


###########################################################################################################################
# Coorinates a full suite of tests
###########################################################################################################################
    def runSuite(self):
        report = []

        for targetYear in range (self.startYear, self.endYear + 1):
            currentScenario = Scenario(self.ticker, self.startYear, self.endYear, targetYear, self.interval, self.lags)

            if targetYear == self.startYear:
                currentScenario.printHeader()

            results = currentScenario.runTests()
            currentScenario.printResults()

            self.sumOfCoefficients[0] += results["coefficients"][0]
            self.sumOfCoefficients[1] += results["coefficients"][1]
            self.sumOfCoefficients[2] += results["coefficients"][2]
            self.sumOfIntercepts += results["intercept"]
            self.sumOfScores += results["score"]
            self.scenarios += 1
            report.append({"ticker": self.ticker, "target": targetYear, "score": results["score"]})

        output = ["Average", "{:.5f}".format(self.sumOfCoefficients[0] / self.scenarios), "{:.5f}".format(self.sumOfCoefficients[1] / self.scenarios),
                  "{:.5f}".format(self.sumOfCoefficients[2] / self.scenarios), "{:.5f}".format(self.sumOfIntercepts / self.scenarios),
                  "{:.5f}".format(self.sumOfScores / self.scenarios)]

        for element in output:
            print(f"{element: ^12}", end=" ")
        print("\n")

        return report


###########################################################################################################################
# Provides a projection of the expected volatility for the upcoming period (as well as the actual volatility for comparison
# if it is available).
###########################################################################################################################
def getProjection(ticker, interval, predictionDate, lags):
    currentSecurity = Asset(ticker)
    currentSecurity.loadData()
    currentSecurity.sortCsvData(COL_DATETIME)
    currentSecurity.cleanData(0.1)
    currentSecurity.aggregateForProjection(interval, predictionDate)
    currentSecurity.addLags(lags)

    bounds = currentSecurity.getBoundaries("0000-00-00", predictionDate)        # Find bounds for slice through selected date

    for i in range (len(currentSecurity.aggregatedTable[0])):                   # Find column with volatility
            if currentSecurity.aggregatedTable[0][i] == VOLATILITY:
                targetColumn = i
    
    # Get slices with training data and the input that we will use to generate prediction
    X_train = currentSecurity.aggregatedTable[2:, bounds["left"]:bounds["right"] + 1].astype('float')
    Y_train = currentSecurity.aggregatedTable[2:, targetColumn].astype('float')

    X = np.array(currentSecurity.aggregatedTable[1][bounds["left"]:bounds["right"] + 1].astype('float')).reshape(1, -1)
    Y_act = currentSecurity.aggregatedTable[1][targetColumn].astype('float')

    reg = LinearRegression().fit(X_train, Y_train)                              # Establish OLS regressive model

    Y_pred = reg.predict(np.array(X))[0]                                        # Predict volatility

    return {"Y_predicted": Y_pred, "price": float(currentSecurity.aggregatedTable[2][COL_CLOSE]), "Y_actual": Y_act}


###########################################################################################################################
# Prints contents of table with volatility results
###########################################################################################################################
def printVolatilityTables(volArray, volMonths, volSecurities, strMessage):
    print(strMessage)

    for i in range (3):
        print()

    output = [" "]
    for month in volMonths:
        output.append(month)

    for element in output:
        print(f"{element: ^10}", end=" ")
    print("\n")

    for j, security in enumerate(volSecurities):
        output = [security]

        for i in range (len(volMonths)):
            output.append("{:.6f}".format(volArray[j][i]))

        for element in output:
            print(f"{element: ^10}", end=" ")
        print("\n")

    

###########################################################################################################################
#                                        EXECUTION BEGINS HERE!!!
###########################################################################################################################
if PREDICT_MODE:                                    # If in PREDICT mode (instead of TEST mode), run this
    # List of months to be included in the table with results
    predictMonths = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # List of securities to be included in the table
    predictSecurities = ["CURE", "DRN", "DUSL", "FAS", "MIDU", "NAIL", "SOXL", "SPXL", "TECL", "TNA", "TQQQ", "UBOT", "UDOW", "UMDD", "UPRO", "URTY", "UTSL", "WANT"]
    predictedVolatility = []
    actualVolatility = []

    # Get predicted volatility for table
    for j, security in enumerate(predictSecurities):
        predictedRow = []
        actualRow = []

        for i, month in enumerate(predictMonths):
            result = getProjection(security, lagProfile[0], "2021-" + str(i + 1).zfill(2) + "-01", lagProfile[1])
            predictedRow.append(result["Y_predicted"] / result["price"])
            actualRow.append(result["Y_actual"] / result["price"])

        predictedVolatility.append(predictedRow)
        actualVolatility.append(actualRow)

    # Print predicted volatility table
    printVolatilityTables(predictedVolatility, predictMonths, predictSecurities, "Predicted Relative Volatility by Month (2021)")

    for i in range (4):
        print()

    # Print actual volatility table
    printVolatilityTables(actualVolatility, predictMonths, predictSecurities, "Actual Relative Volatility by Month (2021)")

    # Find the max predicted volatility and corresponding index
    maxIndex = []
    for i in range (len(predictMonths)):
        maxVol = predictedVolatility[0][i]
        maxInd = 0

        for j in range (1, len(predictSecurities)):
            if predictedVolatility[j][i] > maxVol:
                maxVol = predictedVolatility[j][i]
                maxInd = j

        maxIndex.append(maxInd)

    # Show selected securities
    for i in range (4):
        print()
    print("Selected Security by Month (2021)")
    print()

    output = []
    for element in maxIndex:
        output.append(predictSecurities[element])

    for element in predictMonths:
        print(f"{element: ^10}", end=" ")
    print("\n")

    for element in output:
        print(f"{element: ^10}", end=" ")
    print("\n")

    # Calculate the actual rank of the selected security for each month
    ranks = []
    for i in range (len(predictMonths)):
        selActVol = actualVolatility[maxIndex[i]][i]
        currentRank = 1

        for j in range (len(predictSecurities)):
            if actualVolatility[j][i] > selActVol:
                currentRank += 1

        ranks.append(currentRank)

    for i in range (4):
        print()
    print("Selected Security's Rank by Month for 2021")
    print("(out of " + str(len(predictSecurities)) + ")")
    print()

    output = []
    for element in ranks:
        output.append(str(element))

    for element in predictMonths:
        print(f"{element: ^10}", end=" ")
    print("\n")

    for element in output:
        print(f"{element: ^10}", end=" ")
    print("\n")

else:                                       # TEST mode
    summary = []
    summaryTable = []

    for suite in TEST_SUITES:
        currentSuite = TestSuite(suite[0], suite[1], suite[2], suite[3], suite[4])
        summary.append(currentSuite.runSuite())

    for i in range(len(summary)):
        for j in range(len(summary[i])):
            if len(summaryTable) == 0:
                summaryTable.append([summary[i][j]["ticker"], 0 , 0, 0, 0, 0])
            elif summary[i][j]["ticker"] not in summaryTable[len(summaryTable) - 1]:
                summaryTable.append([summary[i][j]["ticker"], 0 , 0, 0, 0, 0])

            summaryTable[len(summaryTable) - 1][summary[i][j]["target"] - 2016] = summary[i][j]["score"]


    for i in range(5):
        print()
    print("Correlation Score for Target Year")
    print("Interval:  " + str(lagProfile[0]) + " day(s)")
    print("Lags:      " + str(lagProfile[1][0]) + ", " + str(lagProfile[1][1]) + ", " + str(lagProfile[1][2]) + " intervals")
    print()
    headers = ["Ticker", "2017", "2018", "2019", "2020", "2021"]
    for element in headers:
        print(f"{element: ^12}", end=" ")
    print("\n")

    for line in summaryTable:
        output = [line[0]]
        for j in range(1, len(line)):
            if line[j] == 0:
                output.append("N/A")
            else:
                output.append("{:.5f}".format(line[j]))
        for element in output:
            print(f"{element: ^12}", end=" ")
        print("\n")